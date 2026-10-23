"""LLM client: disk cache hit/miss, backoff on a fake 429, non-retryable errors, min gap. No network."""
import pytest

from docscout.reasoning.llm_client import LLMClient, LLMError, LLMProvider, RetryableError, cache_key


class FakeProvider(LLMProvider):
    name = "fake"

    def __init__(self, fail_times=0, status=429, error=None, reply='{"type": "textual"}'):
        self.fail_times, self.status, self.error, self.reply, self.calls = fail_times, status, error, reply, 0

    def complete(self, messages, model, temperature, max_tokens, json_mode):
        self.calls += 1
        if self.error:
            raise self.error
        if self.calls <= self.fail_times:
            raise RetryableError(f"HTTP {self.status}", status=self.status)
        return self.reply, {"prompt_tokens": 10, "completion_tokens": 3}


def make(provider, tmp_path, **kw):
    sleeps = []
    kw.setdefault("min_interval_s", 0.0)
    c = LLMClient(provider, model="m", cache_path=tmp_path / "cache.jsonl", sleep=sleeps.append,
                  backoff_base_s=2.0, max_retries=3, **kw)
    return c, sleeps


def test_cache_miss_then_hit_and_persisted(tmp_path):
    p = FakeProvider()
    c, _ = make(p, tmp_path)
    assert c.complete_json("q1") == {"type": "textual"}
    assert c.complete_json("q1") == {"type": "textual"}
    assert p.calls == 1 and c.stats.calls == 1 and c.stats.cache_hits == 1 and c.stats.prompt_tokens == 10
    c2, _ = make(p, tmp_path)               # new process: cache read from disk
    c2.complete_json("q1")
    assert p.calls == 1 and c2.stats.cache_hits == 1 and c2.stats.calls == 0
    c2.complete_json("q2")                  # different prompt -> miss
    assert p.calls == 2


def test_cache_key_depends_on_model_prompt_temperature():
    m = [{"role": "user", "content": "x"}]
    k = cache_key("groq", "a", m, 0.0)
    assert k != cache_key("groq", "b", m, 0.0)
    assert k != cache_key("groq", "a", m, 0.5)
    assert k != cache_key("groq", "a", [{"role": "user", "content": "y"}], 0.0)
    assert k == cache_key("groq", "a", m, 0)


def test_backoff_on_429_then_success(tmp_path):
    p = FakeProvider(fail_times=2, status=429)
    c, sleeps = make(p, tmp_path)
    assert c.complete("q") == '{"type": "textual"}'
    assert p.calls == 3 and c.stats.retries == 2
    assert sleeps == [2.0, 4.0]             # exponential: base * 2**attempt


def test_gives_up_after_max_retries(tmp_path):
    p = FakeProvider(fail_times=99, status=503)
    c, sleeps = make(p, tmp_path)
    with pytest.raises(LLMError):
        c.complete("q")
    assert p.calls == 4 and sleeps == [2.0, 4.0, 8.0]


def test_non_retryable_error_is_not_retried(tmp_path):
    p = FakeProvider(error=LLMError("HTTP 401"))
    c, sleeps = make(p, tmp_path)
    with pytest.raises(LLMError):
        c.complete("q")
    assert p.calls == 1 and sleeps == []


def test_min_interval_between_real_calls(tmp_path):
    now = [100.0]
    p = FakeProvider()
    sleeps = []
    c = LLMClient(p, model="m", cache_path=tmp_path / "c.jsonl", min_interval_s=2.0, sleep=sleeps.append,
                  clock=lambda: now[0])
    c.complete("a")
    now[0] += 0.5
    c.complete("b")
    c.complete("a")                          # cache hit: no wait
    assert sleeps == [pytest.approx(1.5)]


def test_json_helper_rejects_non_json(tmp_path):
    c, _ = make(FakeProvider(reply="not json"), tmp_path)
    with pytest.raises(LLMError):
        c.complete_json("q")


def test_token_budget_per_minute_waits_for_the_window(tmp_path):
    """tokens_per_minute: a call whose estimate would exceed the rolling 60 s budget waits; cache hits never wait."""
    now = [0.0]
    p = FakeProvider()  # each real call uses 13 tokens
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        now[0] += s
    c = LLMClient(p, model="m", cache_path=tmp_path / "c.jsonl", sleep=sleep, clock=lambda: now[0],
                  max_tokens=0, tokens_per_minute=20)
    c.complete("a" * 40)  # estimate 10 tokens, log empty -> no wait; logs 13
    assert sleeps == []
    now[0] = 5.0
    c.complete("a" * 40)  # cache hit: no wait
    c.complete("b" * 40)  # 13 used + 10 > 20 -> wait until the first call leaves the window (t=60)
    assert sleeps == [55.0] and p.calls == 2


def test_long_retry_after_is_waited_out_not_capped(tmp_path):
    class P(FakeProvider):
        def complete(self, messages, model, temperature, max_tokens, json_mode):
            self.calls += 1
            if self.calls == 1:
                raise RetryableError("HTTP 429", status=429, retry_after=400.0)
            return self.reply, {"prompt_tokens": 1, "completion_tokens": 1}
    c, sleeps = make(P(), tmp_path, max_retry_after_s=900.0)
    c.complete("q")
    assert sleeps == [400.0]  # one long wait instead of several 60 s retries
