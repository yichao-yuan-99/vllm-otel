# SPDX-License-Identifier: Apache-2.0
"""Force-token-sequence custom logits processor for vLLM V1.

Per request, provide ``extra_args["forced_token_ids"]`` and the processor will
force generation to follow those token ids step-by-step.
"""

from __future__ import annotations

import os
from typing import Any

import torch

from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import (
    AdapterLogitsProcessor,
    RequestLogitsProcessor,
)

logger = init_logger(__name__)


class ForceSequencePerRequestProcessor:
    """Request-level processor that forces one token per generation step."""

    def __init__(
        self,
        forced_token_ids: list[int],
        eos_token_id: int | None,
        force_eos_after_sequence: bool,
    ) -> None:
        self.forced_token_ids = forced_token_ids
        self.eos_token_id = eos_token_id
        self.force_eos_after_sequence = force_eos_after_sequence

    def __call__(self, output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        step = len(output_ids)
        target_token_id: int | None = None

        if step < len(self.forced_token_ids):
            target_token_id = self.forced_token_ids[step]
        elif self.force_eos_after_sequence and self.eos_token_id is not None:
            target_token_id = self.eos_token_id
        else:
            return logits

        vocab_size = logits.shape[0]
        if target_token_id < 0 or target_token_id >= vocab_size:
            logger.warning(
                "force sequence target token %d is outside vocab range [0, %d); "
                "skipping forcing for this step.",
                target_token_id,
                vocab_size,
            )
            return logits

        value_to_keep = logits[target_token_id].item()
        logits[:] = float("-inf")
        logits[target_token_id] = value_to_keep
        return logits


class ForceSequenceAdapter(AdapterLogitsProcessor):
    """Batch-level adapter that constructs per-request force-sequence handlers.

    Requires environment variable ``VLLM_FORCE_SEQUENCE_EOS_TOKEN_ID`` at
    processor load time.

    Request parameters are read from ``SamplingParams.extra_args``:
    - ``forced_token_ids``: list[int], required to enable this processor.
    - ``force_eos_after_sequence``: bool, optional (default True).
    """

    ENV_EOS_TOKEN_ID_KEY = "VLLM_FORCE_SEQUENCE_EOS_TOKEN_ID"
    FORCED_TOKEN_IDS_KEY = "forced_token_ids"
    FORCE_EOS_AFTER_SEQUENCE_KEY = "force_eos_after_sequence"

    def __init__(
        self,
        vllm_config: Any,
        device: torch.device,
        is_pin_memory: bool,
    ):
        super().__init__(vllm_config, device, is_pin_memory)
        raw_eos_id = os.getenv(self.ENV_EOS_TOKEN_ID_KEY)
        if raw_eos_id is None or raw_eos_id == "":
            raise ValueError(
                f"{self.ENV_EOS_TOKEN_ID_KEY} must be set for "
                "ForceSequenceAdapter to load."
            )
        self.default_eos_token_id = self._coerce_non_negative_int(
            raw_eos_id,
            self.ENV_EOS_TOKEN_ID_KEY,
        )

    @staticmethod
    def _coerce_bool(value: Any, key: str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, int) and value in (0, 1):
            return bool(value)
        if isinstance(value, str):
            normalized = value.strip().lower()
            if normalized in {"true", "1"}:
                return True
            if normalized in {"false", "0"}:
                return False
        raise ValueError(
            f"{key} must be a bool (or 0/1), got {value!r}."
        )

    @staticmethod
    def _coerce_non_negative_int(value: Any, key: str) -> int:
        if isinstance(value, bool):
            raise ValueError(f"{key} must be a non-negative int, got {value!r}.")
        if isinstance(value, int):
            int_value = value
        elif isinstance(value, float) and value.is_integer():
            int_value = int(value)
        elif isinstance(value, str):
            raw = value.strip()
            if raw.startswith("-"):
                raise ValueError(
                    f"{key} must be a non-negative int, got {value!r}."
                )
            if not raw.isdigit():
                raise ValueError(
                    f"{key} must be a non-negative int, got {value!r}."
                )
            int_value = int(raw)
        else:
            raise ValueError(f"{key} must be a non-negative int, got {value!r}.")

        if int_value < 0:
            raise ValueError(f"{key} must be a non-negative int, got {value!r}.")
        return int_value

    @classmethod
    def validate_params(cls, params: SamplingParams):
        extra_args = params.extra_args or {}
        if cls.FORCED_TOKEN_IDS_KEY not in extra_args:
            return

        forced_token_ids = extra_args[cls.FORCED_TOKEN_IDS_KEY]
        if not isinstance(forced_token_ids, list) or len(forced_token_ids) == 0:
            raise ValueError(
                f"{cls.FORCED_TOKEN_IDS_KEY} must be a non-empty list of ints."
            )
        if not all(isinstance(token_id, int) for token_id in forced_token_ids):
            raise ValueError(
                f"{cls.FORCED_TOKEN_IDS_KEY} must contain only ints, "
                f"got {forced_token_ids!r}."
            )
        if any(token_id < 0 for token_id in forced_token_ids):
            raise ValueError(
                f"{cls.FORCED_TOKEN_IDS_KEY} cannot contain negative values, "
                f"got {forced_token_ids!r}."
            )

        force_eos_after_sequence_raw = extra_args.get(
            cls.FORCE_EOS_AFTER_SEQUENCE_KEY,
            True,
        )
        force_eos_after_sequence = cls._coerce_bool(
            force_eos_after_sequence_raw,
            cls.FORCE_EOS_AFTER_SEQUENCE_KEY,
        )

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(
        self,
        params: SamplingParams,
    ) -> RequestLogitsProcessor | None:
        extra_args: dict[str, Any] = params.extra_args or {}
        forced_token_ids_any = extra_args.get(self.FORCED_TOKEN_IDS_KEY)
        if forced_token_ids_any is None:
            return None

        # Re-validate in case validate_params was not called upstream.
        self.validate_params(params)
        forced_token_ids = forced_token_ids_any
        force_eos_after_sequence = self._coerce_bool(
            extra_args.get(
                self.FORCE_EOS_AFTER_SEQUENCE_KEY,
                True,
            ),
            self.FORCE_EOS_AFTER_SEQUENCE_KEY,
        )
        eos_token_id: int | None = None
        if force_eos_after_sequence:
            eos_token_id = self.default_eos_token_id

        return ForceSequencePerRequestProcessor(
            forced_token_ids=forced_token_ids,
            eos_token_id=eos_token_id,
            force_eos_after_sequence=force_eos_after_sequence,
        )
