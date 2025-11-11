import time
from collections import deque

class RateLimiter:

    #Sliding-window rate limiter per instance
    def __init__(self, max_calls: int, period_seconds: float):
        self.max_calls = max_calls
        self.period = period_seconds
        self.calls = deque()

    def allow(self) -> bool:
        now = time.time()

        # Drop entries older than window
        while self.calls and self.calls[0] <= now - self.period:
            self.calls.popleft()
        if len(self.calls) < self.max_calls:
            self.calls.append(now)
            return True
        return False

    def time_until_next(self) -> float:
        
        # Seconds until next slot is available (0 if already available)
        now = time.time()
        if len(self.calls) < self.max_calls:
            return 0.0
        oldest = self.calls[0]
        return max(0.0, (oldest + self.period) - now)
