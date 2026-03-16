from __future__ import annotations

import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import requests
import torch

from .eval_metrics import evaluate_prediction_rows, parse_json_object, validate_prediction_schema
from .merge_lora import resolve_dtype, resolve_model_path
from .storage import load_jsonl, write_jsonl


SYSTEM_PROMPT = (
    "你是一个结构化抽取器。"
    "你必须只输出一个 JSON 对象，不要输出 markdown，不要输出解释，不要输出前后缀文本。"
    "必须包含 schema 中的所有 key，缺失字段也要显式输出 null。"
    "JSON Schema: "
    "{"
    '"title": str,'
    '"category": str|null,'
    '"price": float,'
    '"currency": "GBP",'
    '"availability": {"in_stock": bool, "stock_count": int|null},'
    '"rating": int|null,'
    '"key_attributes": {'
    '"upc": str,'
    '"product_type": str,'
    '"price_excl_tax": float,'
    '"price_incl_tax": float,'
    '"tax": float,'
    '"availability_text": str,'
    '"review_count": int'
    "}"
    "}"
)


def run_baseline(
    val_path: Path,
    pred_out_path: Path,
    report_out_path: Path,
    *,
    mode: str = "api",
    model_name: str | None = None,
    max_samples: int | None = None,
    max_tokens: int = 512,
) -> dict[str, Any]:
    rows = load_jsonl(val_path)
    if max_samples is not None:
        rows = rows[: max(0, max_samples)]
    elif mode == "local":
        rows = rows[:50]

    predictor = _build_predictor(mode=mode, model_name=model_name)
    prediction_rows: list[dict[str, Any]] = []
    eval_rows: list[dict[str, Any]] = []

    for idx, row in enumerate(rows):
        ground_truth, gt_error = parse_json_object(str(row.get("output", "")))
        if ground_truth is None or gt_error is not None:
            continue

        raw_response, request_error = predictor.predict(
            instruction=str(row.get("instruction", "")),
            input_text=str(row.get("input", "")),
            max_tokens=max_tokens,
        )
        parsed_json, parse_error = parse_json_object(raw_response)
        schema_ok, schema_error = validate_prediction_schema(parsed_json)

        error_reason = request_error or parse_error or schema_error
        prediction_rows.append(
            {
                "idx": idx,
                "model_name": predictor.model_name,
                "input": str(row.get("input", ""))[:500],
                "raw_response": (raw_response or "")[:2000],
                "parsed_json": parsed_json,
                "parse_ok": parse_error is None,
                "schema_ok": schema_ok,
                "error_reason": error_reason,
            }
        )
        eval_rows.append(
            {
                "idx": idx,
                "sample_id": _sample_id(idx, ground_truth),
                "ground_truth": ground_truth,
                "raw_response": raw_response,
                "parsed_json": parsed_json,
                "parse_ok": parse_error is None,
                "schema_ok": schema_ok,
                "error_reason": error_reason,
            }
        )

    write_jsonl(pred_out_path, prediction_rows)
    report = {
        "mode": mode,
        "model_name": predictor.model_name,
        "guided_json_enabled": getattr(predictor, "guided_json_enabled", False),
        "val_path": str(val_path),
        "pred_out": str(pred_out_path),
        "total_requested": len(rows),
        "total_evaluated": len(eval_rows),
    }
    report.update(evaluate_prediction_rows(eval_rows))

    report_out_path.parent.mkdir(parents=True, exist_ok=True)
    report_out_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return report


class _BasePredictor:
    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def predict(
        self,
        instruction: str,
        input_text: str,
        *,
        max_tokens: int,
    ) -> tuple[str, str | None]:
        raise NotImplementedError


class _ApiPredictor(_BasePredictor):
    def __init__(self, model_name: str, api_base: str, api_key: str | None) -> None:
        super().__init__(model_name)
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.session = requests.Session()
        self.guided_json_enabled = _should_use_guided_json(self.api_base)

    def predict(
        self,
        instruction: str,
        input_text: str,
        *,
        max_tokens: int,
    ) -> tuple[str, str | None]:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{instruction}\n\n{input_text}"},
        ]
        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": 0,
            "top_p": 1,
            "max_tokens": max_tokens,
        }
        guided_payload = _with_guided_json(payload, enabled=self.guided_json_enabled)
        last_error = "request_failed"

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        response_text, last_error = self._request_with_retries(
            headers=headers,
            payload=guided_payload,
        )
        if response_text is not None:
            return response_text, None

        if self.guided_json_enabled:
            self.guided_json_enabled = False
            response_text, fallback_error = self._request_with_retries(
                headers=headers,
                payload=payload,
            )
            if response_text is not None:
                return response_text, None
            last_error = fallback_error or last_error

        return "", last_error

    def _request_with_retries(
        self,
        *,
        headers: dict[str, str],
        payload: dict[str, Any],
    ) -> tuple[str | None, str | None]:
        last_error = "request_failed"
        for attempt in range(1, 4):
            time.sleep(random.uniform(0.2, 0.5))
            try:
                response = self.session.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                    timeout=60,
                )
                if response.status_code in {429, 500, 502, 503, 504} and attempt < 3:
                    last_error = f"http_{response.status_code}"
                    time.sleep((2 ** (attempt - 1)) + random.uniform(0.1, 0.3))
                    continue
                if response.status_code == 400:
                    return None, "http_400"
                response.raise_for_status()
                data = response.json()
                content = (
                    data.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                if not str(content).strip():
                    return "", "empty_output"
                return str(content), None
            except (requests.RequestException, ValueError) as exc:
                last_error = str(exc)
                if attempt < 3:
                    time.sleep((2 ** (attempt - 1)) + random.uniform(0.1, 0.3))
                    continue
        return None, last_error


class _LocalPredictor(_BasePredictor):
    def __init__(self, model_name: str) -> None:
        super().__init__(model_name)
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "Local mode requires transformers. Install it manually before using --mode local."
            ) from exc

        resolved_model_path, local_files_only = resolve_model_path(model_name)
        torch_dtype = resolve_dtype("auto")
        self.model_name = resolved_model_path
        self._tokenizer = AutoTokenizer.from_pretrained(
            resolved_model_path,
            trust_remote_code=True,
            use_fast=False,
            local_files_only=local_files_only,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._tokenizer.padding_side = "left"
        self._model = AutoModelForCausalLM.from_pretrained(
            resolved_model_path,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            device_map="auto",
            local_files_only=local_files_only,
        )
        self._model.eval()
        self.guided_json_enabled = False

    def predict(
        self,
        instruction: str,
        input_text: str,
        *,
        max_tokens: int,
    ) -> tuple[str, str | None]:
        prompt = _format_local_chat_prompt(self._tokenizer, instruction, input_text)
        inputs = self._tokenizer(prompt, return_tensors="pt")
        target_device = _infer_model_device(self._model)
        inputs = {key: value.to(target_device) for key, value in inputs.items()}
        generation_config = copy.deepcopy(self._model.generation_config)
        generation_config.do_sample = False
        generation_config.max_new_tokens = max_tokens
        generation_config.pad_token_id = self._tokenizer.pad_token_id
        generation_config.eos_token_id = self._tokenizer.eos_token_id
        generation_config.max_length = None
        with torch.no_grad():
            generated = self._model.generate(
                **inputs,
                generation_config=generation_config,
            )
        new_tokens = generated[0][inputs["input_ids"].shape[1] :]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if not text.strip():
            return "", "empty_output"
        return text, None


def _build_predictor(mode: str, model_name: str | None) -> _BasePredictor:
    if mode == "local":
        resolved_model = model_name or os.getenv("LLM_MODEL") or "Qwen/Qwen2.5-0.5B-Instruct"
        return _LocalPredictor(resolved_model)

    api_base = os.getenv("LLM_API_BASE", "https://api.openai.com/v1")
    api_key = os.getenv("LLM_API_KEY")
    resolved_model = model_name or os.getenv("LLM_MODEL") or "Qwen/Qwen2.5-7B-Instruct"
    return _ApiPredictor(resolved_model, api_base=api_base, api_key=api_key)


def _sample_id(idx: int, ground_truth: dict[str, Any]) -> str:
    key_attributes = ground_truth.get("key_attributes") or {}
    upc = key_attributes.get("upc")
    if isinstance(upc, str) and upc:
        return upc
    return str(idx)


def _should_use_guided_json(api_base: str) -> bool:
    setting = os.getenv("LLM_GUIDED_JSON", "auto").strip().casefold()
    if setting in {"1", "true", "yes", "on"}:
        return True
    if setting in {"0", "false", "no", "off"}:
        return False

    parsed = urlparse(api_base)
    host = (parsed.hostname or "").casefold()
    return host in {"localhost", "127.0.0.1", "0.0.0.0"} or "vllm" in api_base.casefold()


def _with_guided_json(payload: dict[str, Any], *, enabled: bool) -> dict[str, Any]:
    if not enabled:
        return payload

    cloned = dict(payload)
    schema = _guided_json_schema()
    cloned["guided_json"] = schema
    cloned["extra_body"] = {"guided_json": schema}
    return cloned


def _guided_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "title",
            "category",
            "price",
            "currency",
            "availability",
            "rating",
            "key_attributes",
        ],
        "properties": {
            "title": {"type": "string"},
            "category": {"type": ["string", "null"]},
            "price": {"type": "number"},
            "currency": {"type": "string", "enum": ["GBP"]},
            "availability": {
                "type": "object",
                "additionalProperties": False,
                "required": ["in_stock", "stock_count"],
                "properties": {
                    "in_stock": {"type": "boolean"},
                    "stock_count": {"type": ["integer", "null"]},
                },
            },
            "rating": {"type": ["integer", "null"]},
            "key_attributes": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "upc",
                    "product_type",
                    "price_excl_tax",
                    "price_incl_tax",
                    "tax",
                    "availability_text",
                    "review_count",
                ],
                "properties": {
                    "upc": {"type": "string"},
                    "product_type": {"type": "string"},
                    "price_excl_tax": {"type": "number"},
                    "price_incl_tax": {"type": "number"},
                    "tax": {"type": "number"},
                    "availability_text": {"type": "string"},
                    "review_count": {"type": "integer"},
                },
            },
        },
    }


def _format_local_chat_prompt(tokenizer, instruction: str, input_text: str) -> str:
    user_content = f"{instruction}\n\n{input_text}"
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:  # noqa: BLE001
            pass
    return (
        "<|im_start|>system\n"
        f"{SYSTEM_PROMPT}\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        f"{user_content}\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def _infer_model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
