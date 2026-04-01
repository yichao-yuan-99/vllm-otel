from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import sys
from pathlib import Path


INITIAL_SYSTEM_MESSAGE = "sys-old"
INITIAL_USER_MESSAGE = "hi"
FOLLOWUP_USER_MESSAGE = "bye"
SOURCE_MESSAGE_TOKEN_MAP = {
    INITIAL_SYSTEM_MESSAGE: [801, 802],
    INITIAL_USER_MESSAGE: [803],
    FOLLOWUP_USER_MESSAGE: [804],
}
SOURCE_RESPONSE_TOKEN_MAP = {
    "old-1": [11, 15, 17],
    "old-2": [13, 19],
}


def load_module() -> object:
    module_path = (
        Path(__file__).resolve().parents[1] / "generate_derived_single_plan.py"
    ).resolve()
    spec = importlib.util.spec_from_file_location(
        "generate_derived_single_plan",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["generate_derived_single_plan"] = module
    spec.loader.exec_module(module)
    return module


def write_source_plan(plan_path: Path) -> None:
    plan_payload = {
        "schema_version": "replay-plan.v1",
        "replay_target": {
            "model": "Test Model",
            "deterministic_required": True,
        },
        "launch_policy": {
            "strategy": "config_ordered",
            "max_concurrent": 4,
            "pattern": {"name": "eager"},
            "pattern_args": {},
        },
        "compile_options": {
            "single_trail": "profile-1/run_abc",
        },
        "workers": [
            {
                "worker_id": "trial-001",
                "trial_id": "trial-001",
                "api_token": "token-001",
                "api_token_hash": "ignored",
                "launch_priority": 0,
                "source_gateway_run_id": "run_abc",
                "source_trail_name": "profile-1/run_abc",
                "requests": [
                    {
                        "index": 0,
                        "request_id": "req-1",
                        "method": "POST",
                        "path": "v1/chat/completions",
                        "body": {
                            "model": "Test Model",
                            "messages": [
                                {"role": "system", "content": INITIAL_SYSTEM_MESSAGE},
                                {"role": "user", "content": INITIAL_USER_MESSAGE},
                            ],
                            "max_tokens": 2,
                            "vllm_xargs": {
                                "forced_token_ids": [11, 15, 17],
                                "force_eos_after_sequence": True,
                            },
                        },
                        "model_for_tokenize": "Test Model",
                        "delta_agent_action_after_s": 0.1,
                        "replay_mode": "deterministic_forced_tokens",
                        "cancel_after_s": None,
                        "expected_status_code": 200,
                        "expected_error": None,
                        "expected_response_text": "old-1",
                        "forced_token_ids": [11, 15, 17],
                        "force_eos_after_sequence": True,
                    },
                    {
                        "index": 1,
                        "request_id": "req-2",
                        "method": "POST",
                        "path": "v1/chat/completions",
                        "body": {
                            "model": "Test Model",
                            "messages": [
                                {"role": "system", "content": INITIAL_SYSTEM_MESSAGE},
                                {"role": "user", "content": INITIAL_USER_MESSAGE},
                                {"role": "assistant", "content": "old-1"},
                                {"role": "user", "content": FOLLOWUP_USER_MESSAGE},
                            ],
                            "max_tokens": 10,
                            "vllm_xargs": {
                                "forced_token_ids": [13, 19],
                                "force_eos_after_sequence": True,
                            },
                        },
                        "model_for_tokenize": "Test Model",
                        "delta_agent_action_after_s": 0.2,
                        "replay_mode": "deterministic_forced_tokens",
                        "cancel_after_s": None,
                        "expected_status_code": 200,
                        "expected_error": None,
                        "expected_response_text": "old-2",
                        "forced_token_ids": [13, 19],
                        "force_eos_after_sequence": True,
                    },
                ],
            }
        ],
    }
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        json.dumps(plan_payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )


def write_replacement_text(path: Path) -> str:
    text = (
        "Replacement corpus text with enough content to generate deterministic "
        "token windows for derived replay plans.\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return text


def fake_roundtrip_detokenize(model_name: str, token_ids: list[int]) -> str:
    return f"{model_name}|{'-'.join(str(token) for token in token_ids)}"


def fake_roundtrip_tokenize(model_name: str, text: str) -> list[int]:
    prefix = f"{model_name}|"
    if text.startswith(prefix):
        suffix = text[len(prefix) :]
        if not suffix:
            return []
        return [int(token) for token in suffix.split("-")]
    if text in SOURCE_MESSAGE_TOKEN_MAP:
        return list(SOURCE_MESSAGE_TOKEN_MAP[text])
    if text in SOURCE_RESPONSE_TOKEN_MAP:
        return list(SOURCE_RESPONSE_TOKEN_MAP[text])
    raise AssertionError(f"unexpected text for fake_roundtrip_tokenize: {text!r}")


def is_contiguous_window(token_corpus: list[int], token_window: list[int]) -> bool:
    if not token_window or len(token_window) > len(token_corpus):
        return False
    limit = len(token_corpus) - len(token_window) + 1
    for start in range(limit):
        if token_corpus[start : start + len(token_window)] == token_window:
            return True
    return False


def test_derive_plan_payload_clones_worker_and_samples_contiguous_windows() -> None:
    module = load_module()
    replacement_token_corpus = [101, 102, 103, 104, 105, 106, 107]
    source_payload = {
        "schema_version": "replay-plan.v1",
        "replay_target": {"model": "Test Model"},
        "launch_policy": {"strategy": "config_ordered", "pattern": {"name": "eager"}},
        "compile_options": {"single_trail": "profile-1/run_abc"},
        "workers": [
            {
                "worker_id": "trial-001",
                "trial_id": "trial-001",
                "api_token": "token-001",
                "requests": [
                    {
                        "index": 0,
                        "body": {
                            "model": "Test Model",
                            "messages": [
                                {"role": "system", "content": INITIAL_SYSTEM_MESSAGE},
                                {"role": "user", "content": INITIAL_USER_MESSAGE},
                            ],
                            "max_tokens": 1,
                            "vllm_xargs": {
                                "forced_token_ids": [11, 15, 17],
                            },
                        },
                        "model_for_tokenize": "Test Model",
                        "expected_response_text": "old-1",
                        "forced_token_ids": [11, 15, 17],
                        "replay_mode": "deterministic_forced_tokens",
                    },
                    {
                        "index": 1,
                        "body": {
                            "model": "Test Model",
                            "max_tokens": 3,
                            "messages": [
                                {"role": "system", "content": INITIAL_SYSTEM_MESSAGE},
                                {"role": "user", "content": INITIAL_USER_MESSAGE},
                                {"role": "assistant", "content": "old-1"},
                            ],
                            "vllm_xargs": {
                                "forced_token_ids": [13, 19],
                            },
                        },
                        "model_for_tokenize": "Test Model",
                        "expected_response_text": "old-2",
                        "forced_token_ids": [13, 19],
                        "replay_mode": "deterministic_forced_tokens",
                    },
                ],
            }
        ],
    }

    derived_payload, summary = module.derive_plan_payload(
        plan_payload=source_payload,
        seed=7,
        size=3,
        detokenize_fn=fake_roundtrip_detokenize,
        tokenize_fn=fake_roundtrip_tokenize,
        replacement_tokens_provider=lambda model_name: replacement_token_corpus,
        replacement_text_path=Path("/tmp/replacement.txt"),
    )

    assert derived_payload["is_derived"] is True
    assert len(derived_payload["workers"]) == 3
    assert [worker["launch_priority"] for worker in derived_payload["workers"]] == [0, 1, 2]
    assert len({worker["worker_id"] for worker in derived_payload["workers"]}) == 3
    assert len({worker["trial_id"] for worker in derived_payload["workers"]}) == 3
    assert len({worker["api_token"] for worker in derived_payload["workers"]}) == 3
    assert summary["replacement_text_path"] == "/tmp/replacement.txt"
    assert summary["replacement_token_count_by_model"] == {"Test Model": 7}

    for worker in derived_payload["workers"]:
        assert worker["derived_from_worker_id"] == "trial-001"
        assert worker["derived_from_trial_id"] == "trial-001"
        assert worker["api_token_hash"] == module.sha256_hex(worker["api_token"])
        requests = worker["requests"]
        assert len(requests) == 2

        first_tokens = requests[0]["forced_token_ids"]
        second_tokens = requests[1]["forced_token_ids"]
        assert len(first_tokens) == 3
        assert len(second_tokens) == 2
        assert requests[0]["body"]["vllm_xargs"]["forced_token_ids"] == first_tokens
        assert requests[1]["body"]["vllm_xargs"]["forced_token_ids"] == second_tokens
        assert requests[0]["force_eos_after_sequence"] is True
        assert requests[1]["force_eos_after_sequence"] is True
        assert requests[0]["body"]["max_tokens"] >= 4
        assert requests[1]["body"]["max_tokens"] >= 3
        assert is_contiguous_window(replacement_token_corpus, first_tokens)
        assert is_contiguous_window(replacement_token_corpus, second_tokens)
        assert requests[0]["expected_response_text"].startswith("Test Model|")
        assert requests[1]["expected_response_text"].startswith("Test Model|")
        first_request_messages = requests[0]["body"]["messages"]
        second_request_messages = requests[1]["body"]["messages"]
        assert first_request_messages[0]["content"].startswith("Test Model|")
        assert first_request_messages[1]["content"].startswith("Test Model|")
        assert len(fake_roundtrip_tokenize("Test Model", first_request_messages[0]["content"])) == len(
            SOURCE_MESSAGE_TOKEN_MAP[INITIAL_SYSTEM_MESSAGE]
        )
        assert len(fake_roundtrip_tokenize("Test Model", first_request_messages[1]["content"])) == len(
            SOURCE_MESSAGE_TOKEN_MAP[INITIAL_USER_MESSAGE]
        )
        assert second_request_messages[0]["content"] == first_request_messages[0]["content"]
        assert second_request_messages[1]["content"] == first_request_messages[1]["content"]
        assert second_request_messages[2]["content"] == requests[0]["expected_response_text"]

    assert derived_payload["derived_single_plan"]["seed"] == 7
    assert derived_payload["derived_single_plan"]["size"] == 3
    assert derived_payload["derived_single_plan"]["output_worker_count"] == 3
    assert derived_payload["derived_single_plan"]["replacement_text_path"] == "/tmp/replacement.txt"

    verification = module.verify_derived_plan_payload(
        source_plan_payload=source_payload,
        derived_plan_payload=derived_payload,
        tokenize_fn=fake_roundtrip_tokenize,
    )
    assert verification["status"] == "ok"
    assert verification["request_count_per_worker_matches"] is True
    assert verification["generation_length_distribution_matches"] is True
    assert verification["prompt_length_distribution_matches"] is True
    assert verification["cross_worker_history_unique"] is True


def test_sample_replay_stable_sequence_retries_unstable_windows() -> None:
    module = load_module()

    class FakeRandom:
        def __init__(self, values: list[int]) -> None:
            self._values = list(values)

        def randint(self, start: int, end: int) -> int:
            assert start == 0
            assert end == 3
            if not self._values:
                raise AssertionError("randint called too many times")
            return self._values.pop(0)

    def fake_detokenize(model_name: str, token_ids: list[int]) -> str:
        assert model_name == "Test Model"
        mapping = {
            (22,): "unstable-text",
            (33,): "stable-text",
        }
        return mapping[tuple(token_ids)]

    def fake_tokenize(model_name: str, text: str) -> list[int]:
        assert model_name == "Test Model"
        mapping = {
            "unstable-text": [999],
            "stable-text": [33],
        }
        return mapping[text]

    token_ids, expected_response_text = module.sample_replay_stable_sequence(
        rng=FakeRandom([1, 2]),
        token_corpus=[11, 22, 33, 44],
        token_count=1,
        model_name="Test Model",
        detokenize_fn=fake_detokenize,
        tokenize_fn=fake_tokenize,
    )

    assert token_ids == [33]
    assert expected_response_text == "stable-text"


def test_rewrite_request_message_history_replaces_duplicate_responses_in_order() -> None:
    module = load_module()
    request_payload = {
        "body": {
            "messages": [
                {"role": "assistant", "content": "repeat"},
                {"role": "user", "content": "tool output"},
                {"role": "assistant", "content": "repeat"},
            ]
        }
    }

    module.rewrite_request_message_history(
        request_payload=request_payload,
        source_expected_response_texts=["repeat", "repeat"],
        replacement_expected_response_texts=["repeat", "second replacement"],
    )

    messages = request_payload["body"]["messages"]
    assert messages[0]["content"] == "repeat"
    assert messages[2]["content"] == "second replacement"


def test_main_writes_default_prefixed_output_and_refreshes_expected_text(tmp_path: Path) -> None:
    module = load_module()
    plan_path = tmp_path / "replay-plan.trail-demo.json"
    replacement_text_path = tmp_path / "replacement.txt"
    write_source_plan(plan_path)
    replacement_text = write_replacement_text(replacement_text_path)

    seen_calls: list[tuple[str, tuple[int, ...]]] = []
    seen_tokenize_calls: list[tuple[str, str]] = []
    replacement_token_corpus = [31, 32, 33, 34, 35, 36, 37, 38]

    def fake_detokenize_tokens(
        *,
        detokenize_endpoint: str,
        model_name: str,
        forced_token_ids: list[int],
        timeout_s: float,
    ) -> str:
        seen_calls.append((model_name, tuple(forced_token_ids)))
        assert detokenize_endpoint == "http://127.0.0.1:11451/detokenize"
        assert timeout_s == 12.5
        return f"detok:{','.join(str(token) for token in forced_token_ids)}"

    def fake_tokenize_text(
        *,
        tokenize_endpoint: str,
        model_name: str,
        text: str,
        timeout_s: float,
    ) -> list[int]:
        seen_tokenize_calls.append((model_name, text))
        assert tokenize_endpoint == "http://127.0.0.1:11451/tokenize"
        assert timeout_s == 12.5
        if text == replacement_text:
            return list(replacement_token_corpus)
        if text in SOURCE_MESSAGE_TOKEN_MAP:
            return list(SOURCE_MESSAGE_TOKEN_MAP[text])
        if text in SOURCE_RESPONSE_TOKEN_MAP:
            return list(SOURCE_RESPONSE_TOKEN_MAP[text])
        prefix = "detok:"
        assert text.startswith(prefix)
        suffix = text[len(prefix) :]
        if not suffix:
            return []
        return [int(token) for token in suffix.split(",")]

    module.detokenize_tokens = fake_detokenize_tokens
    module.tokenize_text = fake_tokenize_text

    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        exit_code = module.main(
            [
                "--plan",
                str(plan_path),
                "--seed",
                "9",
                "--size",
                "2",
                "--port-profile",
                "0",
                "--replacement-text-file",
                str(replacement_text_path),
                "--request-timeout-s",
                "12.5",
            ]
        )
    assert exit_code == 0
    summary = json.loads(stdout.getvalue())

    output_path = tmp_path / "seed-9-size-2.replay-plan.trail-demo.json"
    assert output_path.is_file()
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["is_derived"] is True
    assert len(payload["workers"]) == 2
    assert len(seen_calls) == 8
    assert len(seen_tokenize_calls) == 20
    assert sum(1 for _, text in seen_tokenize_calls if text == replacement_text) == 1

    expected_lengths = [2, 1, 3, 2, 2, 1, 3, 2]
    for call, expected_length in zip(seen_calls, expected_lengths):
        model_name, token_ids = call
        assert model_name == "Test Model"
        assert len(token_ids) == expected_length
        assert is_contiguous_window(replacement_token_corpus, list(token_ids))

    first_request = payload["workers"][0]["requests"][0]
    second_request = payload["workers"][0]["requests"][1]
    assert first_request["expected_response_text"].startswith("detok:")
    assert first_request["body"]["messages"][0]["content"].startswith("detok:")
    assert first_request["body"]["messages"][1]["content"].startswith("detok:")
    assert second_request["body"]["messages"][0]["content"] == first_request["body"]["messages"][0]["content"]
    assert second_request["body"]["messages"][1]["content"] == first_request["body"]["messages"][1]["content"]
    assert second_request["body"]["messages"][2]["content"] == first_request["expected_response_text"]
    assert payload["derived_single_plan"]["source_plan_path"] == str(plan_path.resolve())
    assert payload["derived_single_plan"]["replacement_text_path"] == str(
        replacement_text_path.resolve()
    )
    assert payload["derived_single_plan"]["total_generated_initial_message_token_count"] == 6
    assert payload["derived_single_plan"]["initial_message_count"] == 2
    assert summary["verification"]["status"] == "ok"
    assert summary["verification"]["request_count_per_worker_matches"] is True
    assert summary["verification"]["generation_length_distribution_matches"] is True
    assert summary["verification"]["prompt_length_distribution_matches"] is True
    assert summary["verification"]["cross_worker_history_unique"] is True


def test_main_batch_mode_derives_all_detected_source_plans(
    tmp_path: Path,
    capsys: object,
) -> None:
    module = load_module()
    plan_root = tmp_path / "plans"
    replacement_text_path = tmp_path / "replacement.txt"
    plan_a = plan_root / "group-a" / "replay-plan.trail-a.json"
    plan_b = plan_root / "group-b" / "replay-plan.trail-b.json"
    write_source_plan(plan_a)
    write_source_plan(plan_b)
    replacement_text = write_replacement_text(replacement_text_path)

    derived_payload = {
        "schema_version": "replay-plan.v1",
        "is_derived": True,
        "workers": [{"requests": [{}]}],
    }
    derived_path = plan_root / "group-c" / "replay-plan.already-derived.json"
    derived_path.parent.mkdir(parents=True, exist_ok=True)
    derived_path.write_text(
        json.dumps(derived_payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    multi_worker_payload = {
        "schema_version": "replay-plan.v1",
        "workers": [
            {"requests": [{}]},
            {"requests": [{}]},
        ],
    }
    multi_worker_path = plan_root / "group-d" / "replay-plan.multi-worker.json"
    multi_worker_path.parent.mkdir(parents=True, exist_ok=True)
    multi_worker_path.write_text(
        json.dumps(multi_worker_payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    one_worker_non_single_trail_payload = {
        "schema_version": "replay-plan.v1",
        "workers": [
            {"requests": [{"index": 0}]},
        ],
    }
    non_single_trail_path = plan_root / "group-e" / "replay-plan.one-worker-not-single-trail.json"
    non_single_trail_path.parent.mkdir(parents=True, exist_ok=True)
    non_single_trail_path.write_text(
        json.dumps(one_worker_non_single_trail_payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )

    seen_calls: list[tuple[str, tuple[int, ...]]] = []
    seen_tokenize_calls: list[tuple[str, str]] = []
    replacement_token_corpus = [41, 42, 43, 44, 45, 46, 47, 48]

    def fake_detokenize_tokens(
        *,
        detokenize_endpoint: str,
        model_name: str,
        forced_token_ids: list[int],
        timeout_s: float,
    ) -> str:
        seen_calls.append((model_name, tuple(forced_token_ids)))
        assert detokenize_endpoint == "http://127.0.0.1:11451/detokenize"
        assert timeout_s == 10.0
        return f"detok:{','.join(str(token) for token in forced_token_ids)}"

    def fake_tokenize_text(
        *,
        tokenize_endpoint: str,
        model_name: str,
        text: str,
        timeout_s: float,
    ) -> list[int]:
        seen_tokenize_calls.append((model_name, text))
        assert tokenize_endpoint == "http://127.0.0.1:11451/tokenize"
        assert timeout_s == 10.0
        if text == replacement_text:
            return list(replacement_token_corpus)
        if text in SOURCE_MESSAGE_TOKEN_MAP:
            return list(SOURCE_MESSAGE_TOKEN_MAP[text])
        if text in SOURCE_RESPONSE_TOKEN_MAP:
            return list(SOURCE_RESPONSE_TOKEN_MAP[text])
        prefix = "detok:"
        assert text.startswith(prefix)
        suffix = text[len(prefix) :]
        if not suffix:
            return []
        return [int(token) for token in suffix.split(",")]

    module.detokenize_tokens = fake_detokenize_tokens
    module.tokenize_text = fake_tokenize_text

    exit_code = module.main(
        [
            "--plan",
            str(plan_root),
            "--seed",
            "5",
            "--size",
            "2",
            "--port-profile",
            "0",
            "--replacement-text-file",
            str(replacement_text_path),
            "--request-timeout-s",
            "10",
        ]
    )
    assert exit_code == 0

    summary = json.loads(capsys.readouterr().out)
    assert summary["mode"] == "batch"
    assert summary["status"] == "ok"
    assert summary["candidate_plan_count"] == 2
    assert summary["derived_plan_count"] == 2
    assert summary["failed_plan_count"] == 0
    assert len(summary["derived_plans"]) == 2
    assert len(seen_calls) == 16
    assert len(seen_tokenize_calls) == 41
    assert sum(1 for _, text in seen_tokenize_calls if text == replacement_text) == 1
    assert summary["replacement_text_file"] == str(replacement_text_path.resolve())
    assert all(
        plan_summary["verification"]["status"] == "ok"
        for plan_summary in summary["derived_plans"]
    )

    output_a = plan_a.parent / "seed-5-size-2.replay-plan.trail-a.json"
    output_b = plan_b.parent / "seed-5-size-2.replay-plan.trail-b.json"
    assert output_a.is_file()
    assert output_b.is_file()
    assert not (
        derived_path.parent / "seed-5-size-2.replay-plan.already-derived.json"
    ).exists()
    assert not (
        multi_worker_path.parent / "seed-5-size-2.replay-plan.multi-worker.json"
    ).exists()
    assert not (
        non_single_trail_path.parent
        / "seed-5-size-2.replay-plan.one-worker-not-single-trail.json"
    ).exists()


def test_resolve_detokenize_endpoint_uses_port_profile() -> None:
    module = load_module()
    assert module.resolve_detokenize_endpoint(0) == "http://127.0.0.1:11451/detokenize"


def test_resolve_tokenize_endpoint_uses_port_profile() -> None:
    module = load_module()
    assert module.resolve_tokenize_endpoint(0) == "http://127.0.0.1:11451/tokenize"


def test_derive_plan_payload_rejects_non_single_worker_plan() -> None:
    module = load_module()
    bad_payload = {
        "schema_version": "replay-plan.v1",
        "workers": [{}, {}],
    }
    try:
        module.derive_plan_payload(
            plan_payload=bad_payload,
            seed=1,
            size=2,
            detokenize_fn=lambda model_name, token_ids: "",
            tokenize_fn=lambda model_name, text: [],
            replacement_tokens_provider=lambda model_name: [1, 2, 3],
        )
    except ValueError as exc:
        assert "exactly 1 worker" in str(exc)
    else:
        raise AssertionError("expected ValueError for non-single-worker plan")


def test_derive_plan_payload_rejects_plan_without_compile_single_trail() -> None:
    module = load_module()
    bad_payload = {
        "schema_version": "replay-plan.v1",
        "workers": [
            {
                "requests": [
                    {
                        "body": {
                            "model": "Test Model",
                            "vllm_xargs": {
                                "forced_token_ids": [1, 2, 3],
                            },
                        },
                        "model_for_tokenize": "Test Model",
                        "forced_token_ids": [1, 2, 3],
                        "replay_mode": "deterministic_forced_tokens",
                    }
                ]
            }
        ],
    }
    try:
        module.derive_plan_payload(
            plan_payload=bad_payload,
            seed=1,
            size=2,
            detokenize_fn=lambda model_name, token_ids: "",
            tokenize_fn=lambda model_name, text: [],
            replacement_tokens_provider=lambda model_name: [1, 2, 3],
        )
    except ValueError as exc:
        assert "compile_options.single_trail" in str(exc)
    else:
        raise AssertionError("expected ValueError for plan without compile_options.single_trail")
