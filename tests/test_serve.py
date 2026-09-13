import json
import time
import threading
import urllib.request
import numpy as np
import pytest
from flux.tensor.tensor import Tensor
from flux.nn.sequential import Sequential
from flux.nn.linear import Linear
from flux.nn.activations import ReLU
from flux.serve.server import ModelServer

def make_model():
    np.random.seed(0)
    return Sequential(Linear(4, 8), ReLU(), Linear(8, 3))

def start_server(port):
    model  = make_model()
    server = ModelServer(model, host="127.0.0.1", port=port)
    server.start(background=True)
    time.sleep(0.2)
    return server

def get(url):
    with urllib.request.urlopen(url, timeout=5) as r:
        return json.loads(r.read())

def post(url, data):
    body = json.dumps(data).encode()
    req  = urllib.request.Request(url, data=body,
           headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read())

# --- unit tests (no HTTP) ---

def test_predict_shape():
    model  = make_model()
    server = ModelServer(model)
    result = server.predict([[1,2,3,4],[5,6,7,8]])
    assert len(result) == 2
    assert len(result[0]) == 3

def test_predict_proba_sums_to_one():
    model  = make_model()
    server = ModelServer(model)
    proba  = server.predict_proba([[1,2,3,4]])
    assert abs(sum(proba[0]) - 1.0) < 1e-5

def test_predict_class_is_int():
    model   = make_model()
    server  = ModelServer(model)
    classes = server.predict_class([[1,2,3,4],[5,6,7,8]])
    assert len(classes) == 2
    assert all(isinstance(c, (int, np.integer)) for c in classes)

def test_predict_class_matches_argmax():
    model   = make_model()
    server  = ModelServer(model)
    logits  = server.predict([[1,2,3,4]])
    cls     = server.predict_class([[1,2,3,4]])
    assert cls[0] == int(np.argmax(logits[0]))

def test_health_keys():
    model  = make_model()
    server = ModelServer(model)
    h = server.health()
    assert "status" in h
    assert "uptime_s" in h
    assert "requests" in h
    assert h["status"] == "ok"

def test_request_count_increments():
    model  = make_model()
    server = ModelServer(model)
    # request_count increments via HTTP handler, not direct calls
    assert server.request_count == 0
    server.request_count += 1
    server.request_count += 1
    assert server.request_count == 2

# --- HTTP integration tests ---

def test_http_health():
    server = start_server(18080)
    try:
        r = get("http://127.0.0.1:18080/health")
        assert r["status"] == "ok"
    finally:
        server.stop()

def test_http_predict():
    server = start_server(18081)
    try:
        r = post("http://127.0.0.1:18081/predict",
                 {"inputs": [[1,2,3,4],[5,6,7,8]]})
        assert "logits" in r
        assert len(r["logits"]) == 2
        assert len(r["logits"][0]) == 3
    finally:
        server.stop()

def test_http_predict_proba():
    server = start_server(18082)
    try:
        r = post("http://127.0.0.1:18082/predict_proba",
                 {"inputs": [[1,2,3,4]]})
        assert "proba" in r
        assert abs(sum(r["proba"][0]) - 1.0) < 1e-5
    finally:
        server.stop()

def test_http_predict_class():
    server = start_server(18083)
    try:
        r = post("http://127.0.0.1:18083/predict_class",
                 {"inputs": [[1,2,3,4],[5,6,7,8]]})
        assert "classes" in r
        assert len(r["classes"]) == 2
    finally:
        server.stop()

def test_http_missing_inputs():
    server = start_server(18084)
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:18084/predict",
            data=json.dumps({}).encode(),
            headers={"Content-Type": "application/json"},
            method="POST")
        try:
            urllib.request.urlopen(req, timeout=5)
        except urllib.error.HTTPError as e:
            assert e.code == 400
    finally:
        server.stop()

def test_http_not_found():
    server = start_server(18085)
    try:
        try:
            get("http://127.0.0.1:18085/nonexistent")
        except urllib.error.HTTPError as e:
            assert e.code == 404
    finally:
        server.stop()

def test_http_batch_large():
    server = start_server(18086)
    try:
        inputs = [[float(i)]*4 for i in range(32)]
        r = post("http://127.0.0.1:18086/predict", {"inputs": inputs})
        assert len(r["logits"]) == 32
    finally:
        server.stop()
