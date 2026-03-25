# Optional rate limiter (not wired to a sidebar page yet).
import time
from collections import deque

class RateLimiter:
    # Each instance tracks call timestamps for one caller/key.
    def __init__(self, max_calls: int, period_seconds: float):
        self.max_calls = max_calls
        self.period = period_seconds
        self.calls = deque()

    def allow(self) -> bool:
        now = time.time()

        # Drop timestamps outside the active window.
        while self.calls and self.calls[0] <= now - self.period:
            self.calls.popleft()
        if len(self.calls) < self.max_calls:
            self.calls.append(now)
            return True
        return False

    def time_until_next(self) -> float:
        # Return wait time before another call is allowed (seconds).
        now = time.time()
        if len(self.calls) < self.max_calls:
            return 0.0
        oldest = self.calls[0]
        return max(0.0, (oldest + self.period) - now)