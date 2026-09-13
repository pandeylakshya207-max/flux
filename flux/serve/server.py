import json
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
import threading
import time

from flux.tensor.tensor import Tensor


class ModelServer:
    def __init__(self, model, host="0.0.0.0", port=8080):
        self.model  = model
        self.host   = host
        self.port   = port
        self._server = None
        self._thread = None
        self.request_count = 0
        self.start_time    = time.time()

    def predict(self, inputs):
        x      = Tensor(np.array(inputs, dtype=np.float32))
        logits = self.model(x)
        return logits.data.tolist()

    def predict_proba(self, inputs):
        x      = Tensor(np.array(inputs, dtype=np.float32))
        logits = self.model(x)
        d      = logits.data
        d      = d - d.max(axis=-1, keepdims=True)
        exp_d  = np.exp(d)
        return (exp_d / exp_d.sum(axis=-1, keepdims=True)).tolist()

    def predict_class(self, inputs):
        x      = Tensor(np.array(inputs, dtype=np.float32))
        logits = self.model(x)
        return logits.data.argmax(axis=-1).tolist()

    def health(self):
        uptime = time.time() - self.start_time
        return {
            "status":   "ok",
            "uptime_s": round(uptime, 2),
            "requests": self.request_count,
        }

    def _make_handler(self):
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, fmt, *args):
                pass

            def send_json(self, code, obj):
                body = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def read_json(self):
                length = int(self.headers.get("Content-Length", 0))
                return json.loads(self.rfile.read(length))

            def do_GET(self):
                path = urlparse(self.path).path
                if path == "/health":
                    self.send_json(200, server.health())
                else:
                    self.send_json(404, {"error": "not found"})

            def do_POST(self):
                path = urlparse(self.path).path
                try:
                    body   = self.read_json()
                    inputs = body.get("inputs")
                    if inputs is None:
                        self.send_json(400, {"error": "missing inputs"})
                        return
                    server.request_count += 1
                    if path == "/predict":
                        self.send_json(200, {"logits": server.predict(inputs)})
                    elif path == "/predict_proba":
                        self.send_json(200, {"proba": server.predict_proba(inputs)})
                    elif path == "/predict_class":
                        self.send_json(200, {"classes": server.predict_class(inputs)})
                    else:
                        self.send_json(404, {"error": "not found"})
                except Exception as e:
                    self.send_json(500, {"error": str(e)})

        return Handler

    def start(self, background=False):
        handler      = self._make_handler()
        self._server = HTTPServer((self.host, self.port), handler)
        print("flux serve http://" + self.host + ":" + str(self.port))
        if background:
            self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
            self._thread.start()
        else:
            self._server.serve_forever()

    def stop(self):
        if self._server:
            self._server.shutdown()
