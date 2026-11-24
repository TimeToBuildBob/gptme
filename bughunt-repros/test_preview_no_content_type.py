"""E2E repro: upstream that omits Content-Type is served by the preview proxy
as text/html (werkzeug default mimetype) WITHOUT the sandbox CSP."""
import socket, sys, threading
sys.path.insert(0, "/tmp/bob-scratch/8fc2/webui-review")
import flask  # noqa: E402
from gptme.server import auth  # noqa: E402
from gptme.server.preview_proxy_api import preview_proxy_api  # noqa: E402

BODY = b"<script>fetch('/api/v2/conversations',{credentials:'include'}).then(r=>r.text()).then(t=>alert(localStorage.gptme_servers+t))</script>"

def upstream(sock):
    while True:
        c, _ = sock.accept()
        c.recv(65536)
        c.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: %d\r\nConnection: close\r\n\r\n" % len(BODY) + BODY)
        c.close()

s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(("127.0.0.1", 0)); s.listen(5); port = s.getsockname()[1]
threading.Thread(target=upstream, args=(s,), daemon=True).start()

auth.set_server_token("t0k")
auth._auth_enabled = True
app = flask.Flask(__name__); app.register_blueprint(preview_proxy_api)
r = app.test_client().get(f"/preview/{port}/", headers={"Authorization": "Bearer t0k"})
body = b"".join(r.response)
print("status", r.status_code)
print("Content-Type:", r.headers.get("Content-Type"))
print("CSP:", r.headers.get("Content-Security-Policy"))
bad = r.headers.get("Content-Type", "").startswith("text/html") and not r.headers.get("Content-Security-Policy")
print("RED: script-capable HTML served on server origin without sandbox" if bad else "GREEN")
sys.exit(1 if bad else 0)
