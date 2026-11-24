"""Repro: preview proxy CSP sandbox is keyed on an exact MIME allowlist.

Browsers render any `*/*+xml` type as an XML document (XHTML-namespace
<script> executes), and resolve a comma-joined Content-Type to the LAST
value. Both reach the browser without the sandbox CSP -> document runs
with the gptme-server origin.
"""
import sys
sys.path.insert(0, "/tmp/bob-scratch/8fc2/webui-review")
from gptme.server.preview_proxy_api import _isolation_headers  # noqa: E402

CASES = [
    "text/html",                       # control: sandboxed
    "application/atom+xml",            # scriptable XML doc in Chrome/Firefox
    "application/foo+xml",
    "text/plain, text/html",           # comma-joined: browser picks text/html
]
fail = 0
for ct in CASES:
    hdrs = dict(_isolation_headers(ct))
    sandboxed = "Content-Security-Policy" in hdrs
    print(f"{ct!r:32} sandboxed={sandboxed}")
    if not sandboxed:
        fail += 1
print("RED" if fail else "GREEN", f"{fail} unsandboxed script-capable types")
sys.exit(1 if fail else 0)
