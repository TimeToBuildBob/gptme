"""Concurrent _prune_attempts() calls race on the shared module-level maps.

autocompact_hook() calls _prune_attempts() unguarded at its top, outside any
try. The server runs one hook per session thread; once more than
_MAX_TRACKED_CONVERSATIONS conversations have been tracked, two sessions that
prune at the same time snapshot the same stale keys and both `del` them; the
loser raises KeyError out of the hook.
"""

import sys
import threading

from gptme.tools.autocompact import hook


def test_concurrent_prune_does_not_raise():
    old_interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    errors: list[BaseException] = []
    try:
        for _ in range(30):
            hook._last_autocompact_attempt.clear()
            hook._failed_summarize.clear()
            n = hook._MAX_TRACKED_CONVERSATIONS * 4
            for i in range(n):
                hook._last_autocompact_attempt[(f"/conv/{i}", "main")] = (0.0, 1)
                hook._failed_summarize[(f"/conv/{i}", "main")] = 1
            barrier = threading.Barrier(4)

            def worker():
                barrier.wait()
                try:
                    hook._prune_attempts(1_000_000.0)
                except BaseException as e:  # noqa: BLE001
                    errors.append(e)

            threads = [threading.Thread(target=worker) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            if errors:
                break
    finally:
        sys.setswitchinterval(old_interval)
        hook._last_autocompact_attempt.clear()
        hook._failed_summarize.clear()
    assert not errors, f"prune raised under concurrency: {errors[0]!r}"
