"""Request progress and token generation tracking."""

from __future__ import annotations

import logging

logger = logging.getLogger("mlenergy.llm.lean")


class RequestTracker:
    """Tracks token generation and request completion for progress logging.

    Intentionally minimal — no steady-state detection, no events.
    The full benchmark window (prefill through final decode) is what we measure.
    """

    def __init__(self, total: int, log: bool = True) -> None:
        self._total = total
        self._log = log
        self._finished: int = 0
        self._tokens: int = 0

    def notify_tokens_generated(self, n: int) -> None:
        self._tokens += n

    def notify_finished(self) -> None:
        self._finished += 1
        if self._log:
            logger.info(
                "%d/%d requests finished — %d tokens generated so far.",
                self._finished,
                self._total,
                self._tokens,
            )

    @property
    def finished(self) -> int:
        return self._finished

    @property
    def total(self) -> int:
        return self._total

    @property
    def tokens_generated(self) -> int:
        return self._tokens
