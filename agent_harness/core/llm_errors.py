"""Safe, typed terminal failures from model requests (never from web tools)."""

from __future__ import annotations


class LLMQuotaError(RuntimeError):
    """A model endpoint rejected payment/quota; immediate retries cannot help."""

    def __init__(self, status_code: int = 402):
        self.status_code = status_code
        self.error_code = "llm_quota_exhausted"
        super().__init__(f"Model API quota/payment rejected (HTTP {status_code}); request stopped")


def quota_error(exc: BaseException) -> LLMQuotaError | None:
    """Inspect structured model errors, without logging bodies or credentials."""
    if isinstance(exc, LLMQuotaError):
        return exc
    response = getattr(exc, "response", None)
    status = getattr(exc, "status_code", None) or getattr(response, "status_code", None)
    if status == 402:
        return LLMQuotaError(402)
    if status != 429:
        return None
    body = getattr(exc, "body", None)
    if not isinstance(body, dict) and response is not None:
        try:
            body = response.json()
        except Exception:
            body = None
    if not isinstance(body, dict):
        return None
    error = body.get("error", body)
    if not isinstance(error, dict):
        return None
    codes = {"insufficient_quota", "billing_hard_limit_reached", "insufficient_credits",
             "credit_balance_exhausted"}
    if any(isinstance(error.get(key), str) and error[key] in codes for key in ("code", "type")):
        return LLMQuotaError(429)
    return None
