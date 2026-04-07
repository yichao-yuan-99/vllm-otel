from __future__ import annotations

from dataclasses import dataclass
import importlib.util
import logging
from pathlib import Path
import sys
from threading import Lock
import types
from typing import Any


LOGGER = logging.getLogger(__name__)
REASONING_DIR = Path(__file__).resolve().parent / "reasoning"

_RUNTIME_LOCK = Lock()
_RUNTIME_READY = False

_PARSER_MODULES: dict[str, tuple[str, str]] = {
    "deepseek_r1": ("deepseek_r1_reasoning_parser", "DeepSeekR1ReasoningParser"),
    "deepseek_v3": ("deepseek_v3_reasoning_parser", "DeepSeekV3ReasoningParser"),
    "ernie45": ("ernie45_reasoning_parser", "Ernie45ReasoningParser"),
    "glm45": ("deepseek_v3_reasoning_parser", "DeepSeekV3ReasoningWithThinkingParser"),
    "openai_gptoss": ("gptoss_reasoning_parser", "GptOssReasoningParser"),
    "granite": ("granite_reasoning_parser", "GraniteReasoningParser"),
    "holo2": ("deepseek_v3_reasoning_parser", "DeepSeekV3ReasoningWithThinkingParser"),
    "hunyuan_a13b": ("hunyuan_a13b_reasoning_parser", "HunyuanA13BReasoningParser"),
    "kimi_k2": ("deepseek_v3_reasoning_parser", "DeepSeekV3ReasoningWithThinkingParser"),
    "minimax_m2": ("minimax_m2_reasoning_parser", "MiniMaxM2ReasoningParser"),
    "minimax_m2_append_think": (
        "minimax_m2_reasoning_parser",
        "MiniMaxM2AppendThinkReasoningParser",
    ),
    "mistral": ("mistral_reasoning_parser", "MistralReasoningParser"),
    "olmo3": ("olmo3_reasoning_parser", "Olmo3ReasoningParser"),
    "qwen3": ("qwen3_reasoning_parser", "Qwen3ReasoningParser"),
    "seed_oss": ("seedoss_reasoning_parser", "SeedOSSReasoningParser"),
    "step3": ("step3_reasoning_parser", "Step3ReasoningParser"),
    "step3p5": ("step3p5_reasoning_parser", "Step3p5ReasoningParser"),
}


@dataclass
class DeltaMessage:
    reasoning: str | None = None
    content: str | None = None


class _ToolServer:
    def has_tool(self, _name: str) -> bool:
        return False


class _MinimalTokenizer:
    def __init__(self) -> None:
        self._vocab: dict[str, int] = {
            "<think>": 1,
            "</think>": 2,
            "<response>": 3,
            "</response>": 4,
            "<0x0A>": 5,
            "[THINK]": 6,
            "[/THINK]": 7,
            "<seed:think>": 8,
            "</seed:think>": 9,
            "<|channel|>final": 10,
            "<|message|>": 11,
            "<|end|>": 12,
        }

    def get_vocab(self) -> dict[str, int]:
        return dict(self._vocab)

    @property
    def vocab(self) -> dict[str, int]:
        return self._vocab

    def encode(self, text: str) -> list[int]:
        token_id = self._vocab.get(text)
        if token_id is not None:
            return [token_id]
        encoded: list[int] = []
        for char in text:
            token_id = self._vocab.setdefault(char, len(self._vocab) + 1)
            encoded.append(token_id)
        return encoded


def _ensure_package(name: str) -> types.ModuleType:
    module = sys.modules.get(name)
    if module is None:
        module = types.ModuleType(name)
        module.__path__ = []  # type: ignore[attr-defined]
        sys.modules[name] = module
    return module


def _install_module(name: str, module: types.ModuleType) -> types.ModuleType:
    sys.modules[name] = module
    return module


def _load_file_module(module_name: str, filename: str) -> types.ModuleType:
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing

    spec = importlib.util.spec_from_file_location(module_name, REASONING_DIR / filename)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not load module {module_name} from {filename}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _install_stub_modules() -> None:
    _ensure_package("vllm")
    _ensure_package("vllm.entrypoints")
    _ensure_package("vllm.entrypoints.openai")
    _ensure_package("vllm.entrypoints.openai.chat_completion")
    _ensure_package("vllm.entrypoints.openai.engine")
    _ensure_package("vllm.entrypoints.openai.responses")
    _ensure_package("vllm.entrypoints.openai.parser")
    _ensure_package("vllm.entrypoints.mcp")
    _ensure_package("vllm.reasoning")
    _ensure_package("vllm.utils")

    logger_module = types.ModuleType("vllm.logger")
    logger_module.init_logger = logging.getLogger
    _install_module("vllm.logger", logger_module)

    tokenizers_module = types.ModuleType("vllm.tokenizers")
    tokenizers_module.TokenizerLike = _MinimalTokenizer
    _install_module("vllm.tokenizers", tokenizers_module)

    collection_module = types.ModuleType("vllm.utils.collection_utils")
    collection_module.is_list_of = (
        lambda value, item_type: isinstance(value, list)
        and all(isinstance(item, item_type) for item in value)
    )
    _install_module("vllm.utils.collection_utils", collection_module)

    import_utils_module = types.ModuleType("vllm.utils.import_utils")

    def import_from_path(path: str) -> Any:
        module_name, _, attr_name = path.partition(":")
        if not module_name or not attr_name:
            raise ValueError("plugin path must use module:attribute syntax")
        module = importlib.import_module(module_name)
        return getattr(module, attr_name)

    import importlib

    import_utils_module.import_from_path = import_from_path
    _install_module("vllm.utils.import_utils", import_utils_module)

    engine_protocol_module = types.ModuleType("vllm.entrypoints.openai.engine.protocol")
    engine_protocol_module.DeltaMessage = DeltaMessage
    _install_module("vllm.entrypoints.openai.engine.protocol", engine_protocol_module)

    chat_protocol_module = types.ModuleType("vllm.entrypoints.openai.chat_completion.protocol")
    chat_protocol_module.ChatCompletionRequest = object
    _install_module("vllm.entrypoints.openai.chat_completion.protocol", chat_protocol_module)

    responses_protocol_module = types.ModuleType("vllm.entrypoints.openai.responses.protocol")
    responses_protocol_module.ResponsesRequest = object
    _install_module("vllm.entrypoints.openai.responses.protocol", responses_protocol_module)

    tool_server_module = types.ModuleType("vllm.entrypoints.mcp.tool_server")
    tool_server_module.ToolServer = _ToolServer
    _install_module("vllm.entrypoints.mcp.tool_server", tool_server_module)

    harmony_module = types.ModuleType("vllm.entrypoints.openai.parser.harmony_utils")
    harmony_module.parse_chat_output = lambda _input_ids: (None, None, None)
    _install_module("vllm.entrypoints.openai.parser.harmony_utils", harmony_module)

    if "transformers" not in sys.modules and importlib.util.find_spec("transformers") is None:
        transformers_module = types.ModuleType("transformers")
        transformers_module.PreTrainedTokenizerBase = object
        _install_module("transformers", transformers_module)


def _ensure_runtime() -> None:
    global _RUNTIME_READY
    with _RUNTIME_LOCK:
        if _RUNTIME_READY:
            return

        _install_stub_modules()
        abs_module = _load_file_module(
            "vllm.reasoning.abs_reasoning_parsers",
            "abs_reasoning_parsers.py",
        )
        basic_module = _load_file_module(
            "vllm.reasoning.basic_parsers",
            "basic_parsers.py",
        )

        reasoning_package = _ensure_package("vllm.reasoning")
        reasoning_package.ReasoningParser = abs_module.ReasoningParser
        reasoning_package.ReasoningParserManager = abs_module.ReasoningParserManager
        reasoning_package.__path__ = [str(REASONING_DIR)]  # type: ignore[attr-defined]

        # Keep the modules reachable through the package for parsers that import from
        # `vllm.reasoning` directly.
        setattr(reasoning_package, "abs_reasoning_parsers", abs_module)
        setattr(reasoning_package, "basic_parsers", basic_module)
        _RUNTIME_READY = True


def load_reasoning_parser_class(parser_name: str) -> type[Any]:
    parser_spec = _PARSER_MODULES.get(parser_name)
    if parser_spec is None:
        raise KeyError(f"unknown reasoning parser: {parser_name}")

    _ensure_runtime()
    file_name, class_name = parser_spec
    module = _load_file_module(
        f"vllm.reasoning.{file_name}",
        f"{file_name}.py",
    )
    parser_class = getattr(module, class_name, None)
    if parser_class is None:
        raise AttributeError(f"{class_name} not found in {file_name}.py")
    return parser_class


def extract_reasoning(parser_name: str, model_output: str) -> tuple[str | None, str | None]:
    parser_class = load_reasoning_parser_class(parser_name)
    parser = parser_class(_MinimalTokenizer())
    reasoning, content = parser.extract_reasoning(model_output, request={})
    return reasoning, content
