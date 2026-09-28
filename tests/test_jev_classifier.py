import json
import os
import sys
import types
import pytest

from auto_router import jev
from auto_router.jev import (
    normalize_jev_url,
    JevClassifier,
    classifier_from_config,
    Classification,
    Judgement,
)


def test_normalize_jev_url_variants():
    # 1. 用户输入带 /api/v1/ 的标准地址
    u1 = normalize_jev_url("https://jev-ai.pro/api/v1/")
    assert u1["base_url"] == "https://jev-ai.pro/api/v1"
    assert u1["systemone_url"] == "https://jev-ai.pro/api/v1/systemone"
    assert u1["models_url"] == "https://jev-ai.pro/api/v1/models"

    # 2. 用户输入末尾不带斜杠的 /api/v1
    u2 = normalize_jev_url("https://jev-ai.pro/api/v1")
    assert u2["base_url"] == "https://jev-ai.pro/api/v1"
    assert u2["systemone_url"] == "https://jev-ai.pro/api/v1/systemone"
    assert u2["models_url"] == "https://jev-ai.pro/api/v1/models"

    # 3. 类似官方 SDK 文档建议填写的 baseURL https://jev-ai.pro/api
    u3 = normalize_jev_url("https://jev-ai.pro/api")
    assert u3["base_url"] == "https://jev-ai.pro/api/v1"
    assert u3["systemone_url"] == "https://jev-ai.pro/api/v1/systemone"
    assert u3["models_url"] == "https://jev-ai.pro/api/v1/models"

    # 4. 用户直接输入完整的 systemone 端点
    u4 = normalize_jev_url("https://jev-ai.pro/api/v1/systemone")
    assert u4["base_url"] == "https://jev-ai.pro/api/v1"
    assert u4["systemone_url"] == "https://jev-ai.pro/api/v1/systemone"
    assert u4["models_url"] == "https://jev-ai.pro/api/v1/models"

    # 5. TypeSafe 原生域名
    u5 = normalize_jev_url("https://api.typesafe.ai")
    assert u5["base_url"] == "https://api.typesafe.ai/v1"
    assert u5["systemone_url"] == "https://api.typesafe.ai/v1/systemone"
    assert u5["models_url"] == "https://api.typesafe.ai/v1/models"

    # 6. 空值默认兜底
    u6 = normalize_jev_url(None)
    assert u6["systemone_url"].endswith("/systemone")
    assert u6["models_url"].endswith("/models")


def test_jev_classifier_from_config():
    cfg = {
        "backend": "jev",
        "jev": {
            "base_url": "https://jev-ai.pro/api/v1/",
            "api_key": "test_jev_key_123",
            "model": "jev-custom",
        }
    }
    clf = classifier_from_config({"classifier": cfg})
    assert isinstance(clf, JevClassifier)
    assert clf.base_url == "https://jev-ai.pro/api/v1"
    assert clf.systemone_url == "https://jev-ai.pro/api/v1/systemone"
    assert clf.api_key == "test_jev_key_123"
    assert clf.model == "jev-custom"


def test_jev_classifier_classify_mock(monkeypatch):
    mock_payload = {
        "model": "jev-1.13.0",
        "answers": {
            "category": {"type": "choice", "choice": "coding", "probabilities": {"coding": 0.95}, "confidence": 0.9},
            "difficulty": {"type": "score", "score": 2.0, "confidence": 0.85},
            "needs_tools": {"type": "noul", "noul": 0.88},
            "needs_vision": {"type": "noul", "noul": 0.05},
            "needs_long_context": {"type": "noul", "noul": 0.12},
            "follow_up": {"type": "noul", "noul": 0.2},
            "stakes": {"type": "score", "score": 1.0, "confidence": 0.75},
        },
        "usage": {"input_tokens": 120, "output_tokens": 40},
    }

    client = JevClassifier(base_url="https://jev-ai.pro/api/v1/", api_key="sk-test", model="jev-latest")

    def mock_post(self, state, questions, timeout=None):
        assert "request" in state
        return mock_payload, 0.045

    monkeypatch.setattr(JevClassifier, "_post", mock_post)

    res = client("帮我写一个快速排序算法")
    assert isinstance(res, Classification)
    assert res.category == "coding"
    assert res.category_confidence == 0.9
    assert res.difficulty == 0.5  # 2 / (5 - 1)
    assert res.needs_tools == 0.88
    assert res.input_tokens == 120
    assert res.output_tokens == 40
    assert res.model == "jev-1.13.0"
    assert res.source == "jev[jev-latest]"


def test_jev_classifier_judge_mock(monkeypatch):
    mock_payload = {
        "model": "jev-1.13.0",
        "answers": {
            "adequate": {"type": "noul", "noul": 0.92},
            "failure": {"type": "choice", "choice": "fine", "probabilities": {"fine": 0.95}},
        },
        "usage": {"input_tokens": 80, "output_tokens": 20},
    }

    client = JevClassifier(base_url="https://jev-ai.pro/api/v1/", api_key="sk-test", model="jev-latest")

    def mock_post(self, state, questions, timeout=None):
        assert "request" in state and "response" in state
        return mock_payload, 0.035

    monkeypatch.setattr(JevClassifier, "_post", mock_post)

    res = client.judge("计算 1 + 1", "答案是 2", category="math")
    assert isinstance(res, Judgement)
    assert res.p_adequate == 0.92
    assert res.failure == "fine"
    assert not res.failed
    assert res.model == "jev-1.13.0"


def test_build_judge_with_jev():
    from server import build_judge

    mock_config = types.SimpleNamespace(
        catalog=types.SimpleNamespace(models=[]),
        providers={},
    )
    raw_policy = {
        "classifier": {
            "backend": "jev",
            "jev": {
                "base_url": "https://jev-ai.pro/api/v1/",
                "api_key": "sk-jev-verify",
                "model": "jev-latest",
            }
        },
        "verify": {
            "enabled": True,
            "judge_model": "jev",
        }
    }

    judge_fn = build_judge(mock_config, raw_policy)
    assert judge_fn is not None
    assert "jev_judge" in judge_fn.__name__


def test_classifier_two_way_selection():
    # 1. 明确选 Laya
    clf_laya = classifier_from_config({"classifier": {"backend": "laya"}})
    assert isinstance(clf_laya, jev.LocalLayaClassifier)

    # 2. 明确选 Jev
    clf_jev = classifier_from_config({"classifier": {"backend": "jev", "jev": {"base_url": "https://jev-ai.pro/api/v1/"}}})
    assert isinstance(clf_jev, JevClassifier)
    assert clf_jev.base_url == "https://jev-ai.pro/api/v1"


def test_api_jev_endpoints(monkeypatch):
    from fastapi.testclient import TestClient
    from server import app

    client = TestClient(app)

    # 1. 模拟免推理探活 test_connection 成功
    def mock_test_connection_success(self):
        return {"status": "ok", "latency_ms": 32.5, "models": ["jev-1.13.0", "jev-fast"]}

    monkeypatch.setattr(JevClassifier, "test_connection", mock_test_connection_success)

    resp = client.post("/api/jev/test", json={
        "base_url": "https://jev-ai.pro/api/v1/",
        "api_key": "test-key-abc",
        "model": "jev-latest",
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["latency_ms"] == 32.5
    assert "jev-1.13.0" in data["models"]

    # 2. 模拟真实决策实时测试 test-decision 成功
    def mock_classify(self, text, context=None):
        return Classification(
            category="coding",
            category_probs={"coding": 0.96},
            category_confidence=0.96,
            difficulty=0.45,
            difficulty_confidence=0.88,
            needs_tools=0.2,
            needs_vision=0.0,
            needs_long_context=0.1,
            follow_up=0.0,
            stakes=0.3,
            input_tokens=50,
            output_tokens=15,
            model="jev-1.13.0",
            source_name="jev[jev-latest]",
        )

    def mock_judge(self, prompt, response, category=None):
        return Judgement(
            p_adequate=0.94,
            failure=None,
            model="jev-1.13.0",
        )

    monkeypatch.setattr(JevClassifier, "__call__", mock_classify)
    monkeypatch.setattr(JevClassifier, "judge", mock_judge)

    resp_dec = client.post("/api/jev/test-decision", json={
        "base_url": "https://jev-ai.pro/api/v1/",
        "api_key": "test-key-abc",
        "model": "jev-latest",
        "prompt": "编写二分查找算法",
        "test_judge": True,
    })
    assert resp_dec.status_code == 200
    dec_data = resp_dec.json()
    assert dec_data["status"] == "ok", dec_data.get("error")
    assert dec_data["classification"]["category"] == "coding"
    assert dec_data["classification"]["difficulty"] == 0.45
    assert dec_data["classification"]["needs_tools"] == 0.2
    assert dec_data["classification"]["stakes"] == 0.3
    assert dec_data["judge"]["p_adequate"] == 0.94


def test_api_status_adaptive_to_jev(monkeypatch):
    from fastapi.testclient import TestClient
    from server import app, ar_server

    client = TestClient(app)

    # 模拟 router.classifier 为 JevClassifier
    jev_clf = JevClassifier(base_url="https://jev-ai.pro/api/v1/", api_key="sk-test", model="jev-pro-v1")
    dummy_router = types.SimpleNamespace(classifier=jev_clf)
    monkeypatch.setattr(ar_server, "router", dummy_router)

    resp = client.get("/api/status")
    assert resp.status_code == 200
    st = resp.json()
    assert st["is_jev"] is True
    assert st["classifier_backend"] == "jev"
    assert "Jev AI" in st["classifier_name"]
    assert "jev-pro-v1" in st["engine_display"]
    assert "https://jev-ai.pro/api/v1" in st["endpoint_display"]

