"""First concurrent lookup of the context-provider registry can fail.

_init_registry() flips _registry_initialized=True *before* loading entry
points and registering "default". A second thread that calls
get_context_provider("default") during that window (the entry-point scan
takes ~10-30ms) sees the registry marked initialized but empty and gets
ValueError("Unknown context provider: 'default'"). In the server this is the
first rule-based compaction of two sessions at once; hook.py swallows the
exception and silently skips compaction.
"""

import threading

from gptme.tools.autocompact import context_provider as cp


def _run_once() -> list[BaseException]:
    cp._provider_registry.clear()
    cp._registry_initialized = False
    errors: list[BaseException] = []
    barrier = threading.Barrier(8)

    def worker():
        barrier.wait()
        try:
            cp.get_context_provider("default")
        except BaseException as e:  # noqa: BLE001
            errors.append(e)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


def test_concurrent_first_lookup_finds_default_provider():
    saved = dict(cp._provider_registry), cp._registry_initialized
    try:
        all_errors: list[BaseException] = []
        for _ in range(5):
            all_errors += _run_once()
        assert not all_errors, f"{len(all_errors)} lookups failed: {all_errors[0]!r}"
    finally:
        cp._provider_registry.clear()
        cp._provider_registry.update(saved[0])
        cp._registry_initialized = saved[1]
