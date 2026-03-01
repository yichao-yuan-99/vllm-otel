from __future__ import annotations

from copy import deepcopy
import logging
from typing import Any

from gateway.model_configs import ModelRegistry
from gateway.reasoning_runtime import extract_reasoning


LOGGER = logging.getLogger(__name__)


class ReasoningResponseParser:
    def __init__(self, model_registry: ModelRegistry) -> None:
        self.model_registry = model_registry

    def transform(
        self,
        request_path: str,
        request_payload: Any,
        response_payload: Any,
    ) -> Any:
        normalized_path = request_path.split("?", 1)[0].strip("/")
        if normalized_path != "v1/chat/completions":
            return response_payload
        if not isinstance(response_payload, dict):
            return response_payload
        if response_payload.get("object") != "chat.completion":
            return response_payload

        parser_name = self._resolve_parser_name(request_payload, response_payload)
        if not parser_name:
            return response_payload

        raw_choices = response_payload.get("choices")
        if not isinstance(raw_choices, list):
            return response_payload

        transformed_payload = response_payload
        transformed_choices: list[Any] | None = None
        changed = False

        for index, choice in enumerate(raw_choices):
            transformed_choice = self._transform_choice(choice, parser_name)
            if transformed_choice is choice:
                continue
            if transformed_choices is None:
                transformed_payload = deepcopy(response_payload)
                transformed_choices = list(raw_choices)
                transformed_payload["choices"] = transformed_choices
            transformed_choices[index] = transformed_choice
            changed = True

        return transformed_payload if changed else response_payload

    def _resolve_parser_name(
        self,
        request_payload: Any,
        response_payload: dict[str, Any],
    ) -> str | None:
        candidate_names: list[str | None] = []
        if isinstance(response_payload.get("model"), str):
            candidate_names.append(response_payload.get("model"))
        if isinstance(request_payload, dict) and isinstance(request_payload.get("model"), str):
            candidate_names.append(request_payload.get("model"))

        for model_name in candidate_names:
            parser_name = self.model_registry.resolve_reasoning_parser(model_name)
            if parser_name:
                return parser_name
        return None

    def _transform_choice(self, choice: Any, parser_name: str) -> Any:
        if not isinstance(choice, dict):
            return choice
        raw_message = choice.get("message")
        if not isinstance(raw_message, dict):
            return choice
        raw_content = raw_message.get("content")
        if not isinstance(raw_content, str):
            return choice

        try:
            reasoning, parsed_content = extract_reasoning(parser_name, raw_content)
        except Exception:
            LOGGER.exception("reasoning parse failed for parser=%s", parser_name)
            return choice

        if (
            raw_message.get("content") == parsed_content
            and raw_message.get("reasoning") == reasoning
            and raw_message.get("reasoning_content") == reasoning
        ):
            return choice

        transformed_choice = dict(choice)
        transformed_message = dict(raw_message)
        transformed_message["content"] = parsed_content
        transformed_message["reasoning"] = reasoning
        transformed_message["reasoning_content"] = reasoning
        transformed_choice["message"] = transformed_message
        return transformed_choice
