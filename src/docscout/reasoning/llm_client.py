"""Provider-agnostic LLM client with a disk cache, retries with backoff, rate limiting and usage counters.

- Provider, model and key come from .env (LLM_PROVIDER, LLM_MODEL, LLM_API_KEY); other settings from the
  `llm` section of configs/config.yaml. The key is never logged or written anywhere.
- Cache: one JSON line per response in `llm.cache_path`, keyed by sha256(provider, model, messages, temperature).
- Retries: 429, 5xx and connection/timeout errors are retried `max_retries` times with exponential backoff
  (or the server's Retry-After, if larger, honoured up to max_retry_after_s). Other errors (400, 401, ...) fail immediately.
- A minimum gap (`min_interval_s`) is kept between real API calls; cache hits are not delayed.

Usage:
    client = get_client()
    data = client.complete_json("Return {\"ok\": true}", system="You answer in JSON.")
    print(client.stats)
"""
from __future__ import annotations

import hashlib
import logging
import json
import threading
import time
from collections import deque
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from docscout.config import get_config, get_secret

log = logging.getLogger(__name__)

Messages = List[Dict[str, str]]
Usage = Dict[str, int]


class LLMError(RuntimeError):
    """The LLM call failed (after retries, or with a non-retryable error)."""


class MissingKeyError(LLMError):
    """LLM_API_KEY is not set in .env."""


class RetryableError(Exception):
    """Raised by providers for errors worth retrying (429, 5xx, connection problems)."""

    def __init__(self, message: str, status: Optional[int] = None, retry_after: Optional[float] = None) -> None:
        super().__init__(message)
        self.status, self.retry_after = status, retry_after


class LLMProvider(ABC):
    """One chat-completion backend. Implement `complete` to add a provider."""

    name: str = "base"

    @abstractmethod
    def complete(self, messages: Messages, model: str, temperature: float, max_tokens: int,
                 json_mode: bool) -> Tuple[str, Usage]:
        """Return (text, usage) where usage has prompt_tokens / completion_tokens. Raise RetryableError to retry."""


class GroqProvider(LLMProvider):
    """Groq chat completions (OpenAI-compatible API)."""

    name = "groq"

    def __init__(self, api_key: str, timeout_s: float = 30.0, reasoning_effort: Optional[str] = None) -> None:
        from groq import Groq

        self.reasoning_effort = reasoning_effort  # only sent to reasoning models (e.g. openai/gpt-oss-*)

        # max_retries=0: retries are handled (and counted) by LLMClient
        self._client = Groq(api_key=api_key, timeout=timeout_s, max_retries=0)

    def complete(self, messages: Messages, model: str, temperature: float, max_tokens: int,
                 json_mode: bool) -> Tuple[str, Usage]:
        import groq

        kwargs: Dict[str, Any] = {"model": model, "messages": messages, "temperature": temperature,
                                  "max_tokens": max_tokens}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        if self.reasoning_effort and "gpt-oss" in model:
            kwargs["reasoning_effort"] = self.reasoning_effort
        try:
            resp = self._client.chat.completions.create(**kwargs)
        except (groq.APIConnectionError, groq.APITimeoutError) as e:
            raise RetryableError(f"connection error: {type(e).__name__}") from e
        except groq.APIStatusError as e:
            status = getattr(e, "status_code", None)
            if status == 429 or (status is not None and status >= 500):
                ra = None
                try:
                    ra = float(e.response.headers.get("retry-after"))  # type: ignore[union-attr]
                except Exception:  # noqa: BLE001 - header missing or not a number
                    ra = None
                raise RetryableError(f"HTTP {status}", status=status, retry_after=ra) from e
            raise LLMError(f"groq error HTTP {status}: {type(e).__name__}") from e
        u = resp.usage
        usage = {"prompt_tokens": int(getattr(u, "prompt_tokens", 0) or 0),
                 "completion_tokens": int(getattr(u, "completion_tokens", 0) or 0)}
        return resp.choices[0].message.content or "", usage


PROVIDERS: Dict[str, Callable[..., LLMProvider]] = {"groq": GroqProvider}


@dataclass
class LLMStats:
    """Counters for one client: real API calls, cache hits, retries and tokens."""

    calls: int = 0
    cache_hits: int = 0
    retries: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds_in_api: float = 0.0

    def as_dict(self) -> Dict[str, Any]:
        d = dict(self.__dict__)
        d["total_tokens"] = self.prompt_tokens + self.completion_tokens
        d["seconds_in_api"] = round(self.seconds_in_api, 2)
        return d

    def __str__(self) -> str:
        d = self.as_dict()
        return (f"LLM calls={d['calls']} cache_hits={d['cache_hits']} retries={d['retries']} "
                f"tokens={d['total_tokens']} (prompt {d['prompt_tokens']}, completion {d['completion_tokens']})")


def cache_key(provider: str, model: str, messages: Messages, temperature: float) -> str:
    """sha256 over (provider, model, prompt messages, temperature)."""
    payload = json.dumps([provider, model, messages, float(temperature)], ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class DiskCache:
    """Append-only JSONL cache: {"key", "text", "usage"} per line. Loaded fully into memory."""

    def __init__(self, path: Optional[Path]) -> None:
        self.path = Path(path) if path else None
        self._data: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        if self.path and self.path.exists():
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except json.JSONDecodeError:  # a torn last line from an interrupted run
                        continue
                    self._data[rec["key"]] = rec

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        return self._data.get(key)

    def put(self, key: str, text: str, usage: Usage, meta: Dict[str, Any]) -> None:
        rec = {"key": key, "text": text, "usage": usage, **meta}
        with self._lock:
            self._data[key] = rec
            if self.path:
                self.path.parent.mkdir(parents=True, exist_ok=True)
                with self.path.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def __len__(self) -> int:
        return len(self._data)


class LLMClient:
    """Cached, rate-limited, retrying chat client over any LLMProvider."""

    def __init__(self, provider: LLMProvider, model: str, temperature: float = 0.0, max_tokens: int = 512,
                 max_retries: int = 5, backoff_base_s: float = 2.0, backoff_max_s: float = 60.0,
                 min_interval_s: float = 0.0, cache_path: Optional[Path] = None,
                 sleep: Callable[[float], None] = time.sleep, clock: Callable[[], float] = time.monotonic,
                 tokens_per_minute: int = 0, max_retry_after_s: float = 900.0) -> None:
        self.provider, self.model = provider, model
        self.temperature, self.max_tokens = temperature, max_tokens
        self.max_retries, self.backoff_base_s, self.backoff_max_s = max_retries, backoff_base_s, backoff_max_s
        self.min_interval_s = min_interval_s
        self.cache = DiskCache(cache_path)
        self.stats = LLMStats()
        self._sleep, self._clock = sleep, clock
        self._last_call: Optional[float] = None
        self.max_retry_after_s = float(max_retry_after_s)  # a server Retry-After is honoured up to this (not backoff_max_s)
        self.tokens_per_minute = int(tokens_per_minute or 0)  # 0 = no client-side token budget
        self._token_log: "deque[Tuple[float, int]]" = deque()  # (time of real call, total tokens)

    def _wait_gap(self) -> None:
        if self._last_call is not None and self.min_interval_s > 0:
            wait = self.min_interval_s - (self._clock() - self._last_call)
            if wait > 0:
                self._sleep(wait)

    def _wait_tokens(self, est_tokens: int) -> None:
        """Keep the tokens of real calls in any rolling 60 s window under tokens_per_minute (estimate first)."""
        if self.tokens_per_minute <= 0:
            return
        while True:
            now = self._clock()
            while self._token_log and now - self._token_log[0][0] >= 60.0:
                self._token_log.popleft()
            used = sum(t for _, t in self._token_log)
            if not self._token_log or used + est_tokens <= self.tokens_per_minute:
                return
            self._sleep(max(0.1, 60.0 - (now - self._token_log[0][0])))

    @staticmethod
    def estimate_call_tokens(messages: Messages, max_tokens: int) -> int:
        """Prompt chars / 4 plus half the completion budget (gpt-oss also spends hidden reasoning tokens)."""
        return sum(len(m["content"]) for m in messages) // 4 + max_tokens // 2

    def _call_with_retries(self, messages: Messages, json_mode: bool) -> Tuple[str, Usage]:
        last: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            self._wait_tokens(self.estimate_call_tokens(messages, self.max_tokens))
            self._wait_gap()
            t0 = time.perf_counter()
            try:
                self._last_call = self._clock()
                self.stats.calls += 1
                text, usage = self.provider.complete(messages, self.model, self.temperature, self.max_tokens, json_mode)
                self.stats.seconds_in_api += time.perf_counter() - t0
                self._token_log.append((self._clock(), usage.get("prompt_tokens", 0) + usage.get("completion_tokens", 0)))
                return text, usage
            except RetryableError as e:
                self.stats.seconds_in_api += time.perf_counter() - t0
                last = e
                if attempt == self.max_retries:
                    break
                self.stats.retries += 1
                wait = min(self.backoff_max_s, self.backoff_base_s * (2 ** attempt))
                if e.retry_after is not None:
                    # wait out the server's Retry-After (rolling quota) instead of re-trying early and burning requests
                    wait = max(wait, min(self.max_retry_after_s, e.retry_after))
                log.warning("LLM retry %d/%d after %s (status %s, retry-after %s): waiting %.1fs", attempt + 1,
                            self.max_retries, e, e.status, e.retry_after, wait)
                self._sleep(wait)
        raise LLMError(f"LLM call failed after {self.max_retries} retries: {last}")

    def complete(self, prompt: str, system: Optional[str] = None, json_mode: bool = False) -> str:
        """Return the model's text for `prompt` (cached)."""
        messages: Messages = ([{"role": "system", "content": system}] if system else []) + \
                             [{"role": "user", "content": prompt}]
        key = cache_key(self.provider.name, self.model, messages, self.temperature)
        hit = self.cache.get(key)
        if hit is not None:
            self.stats.cache_hits += 1
            return hit["text"]
        text, usage = self._call_with_retries(messages, json_mode)
        self.stats.prompt_tokens += usage.get("prompt_tokens", 0)
        self.stats.completion_tokens += usage.get("completion_tokens", 0)
        self.cache.put(key, text, usage, {"provider": self.provider.name, "model": self.model, "json_mode": json_mode})
        return text

    def complete_json(self, prompt: str, system: Optional[str] = None) -> Dict[str, Any]:
        """JSON-mode completion parsed into a dict. Raises LLMError if the reply is not a JSON object."""
        text = self.complete(prompt, system=system, json_mode=True)
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise LLMError(f"LLM did not return valid JSON: {text[:200]!r}") from e
        if not isinstance(data, dict):
            raise LLMError(f"LLM JSON is not an object: {text[:200]!r}")
        return data


_CLIENT: Optional[LLMClient] = None


def get_client(cfg: Optional[dict] = None, model: Optional[str] = None) -> LLMClient:
    """Build (once per process) the configured client. Raises MissingKeyError if LLM_API_KEY is not set.

    `model` overrides llm.model (e.g. llm.demo_model for scripts.ask --demo); such a client is not shared."""
    global _CLIENT
    if _CLIENT is not None and cfg is None and model is None:
        return _CLIENT
    shared = cfg is None and model is None
    cfg = cfg or get_config()
    llm = dict(cfg["llm"])
    if model:
        llm["model"] = model
    key = get_secret("LLM_API_KEY")
    if not key:
        raise MissingKeyError("LLM_API_KEY is missing in .env")
    name = (llm.get("provider") or "groq").lower()
    if name not in PROVIDERS:
        raise LLMError(f"unknown LLM provider {name!r}; available: {sorted(PROVIDERS)}")
    provider = PROVIDERS[name](api_key=key, timeout_s=float(llm.get("timeout_s", 30)),
                               reasoning_effort=llm.get("reasoning_effort"))
    client = LLMClient(provider, model=llm["model"], temperature=float(llm.get("temperature", 0.0)),
                       max_tokens=int(llm.get("max_tokens", 512)), max_retries=int(llm.get("max_retries", 5)),
                       backoff_base_s=float(llm.get("backoff_base_s", 2.0)),
                       backoff_max_s=float(llm.get("backoff_max_s", 60.0)),
                       min_interval_s=float(llm.get("min_interval_s", 0.0)), cache_path=llm.get("cache_path"),
                       tokens_per_minute=int(llm.get("tokens_per_minute", 0) or 0),
                       max_retry_after_s=float(llm.get("max_retry_after_s", 900)))
    if shared:
        _CLIENT = client
    return client
