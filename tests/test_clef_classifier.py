import json
import os
import urllib.error
import pytest

from auto_router.clef import (
    DEFAULT_CLEF_MODEL,
    ClefClassifier,
    build_clef_endpoint,
)
from auto_router.jev import (
    FALLBACK,
    Classification,
    Judgement,
    classifier_from_config,
)


def test_build_clef_endpoint():
    # 1. 官方标准端点构建
    ep1 = build_clef_endpoint(account_id="test_acc_123", model="@cf/cloudflare/clef-flash")
    assert ep1 == "https://api.cloudflare.com/client/v4/accounts/test_acc_123/ai/run/@cf/cloudflare/clef-flash"

    # 2. 官方标准端点切换为 27B Clef
    ep2 = build_clef_endpoint(account_id="test_acc_123", model="@cf/cloudflare/clef")
    assert ep2 == "https://api.cloudflare.com/client/v4/accounts/test_acc_123/ai/run/@cf/cloudflare/clef"

    # 3. 自定义 AI Gateway 或反向代理 base_url
    ep3 = build_clef_endpoint(account_id="test_acc_123", model="@cf/cloudflare/clef-flash", base_url="https://gateway.ai.cloudflare.com/v1/test_acc_123/my-gw/workers-ai/run")
    assert ep3 == "https://gateway.ai.cloudflare.com/v1/test_acc_123/my-gw/workers-ai/run/@cf/cloudflare/clef-flash"

    # 4. 自定义 SystemOne 兼容网关
    ep4 = build_clef_endpoint(account_id="test_acc_123", base_url="https://custom-gateway.local/v1/systemone")
    assert ep4 == "https://custom-gateway.local/v1/systemone"


def test_clef_classifier_from_config(monkeypatch):
    monkeypatch.delenv("CLOUDFLARE_ACCOUNT_ID", raising=False)
    monkeypatch.delenv("CLOUDFLARE_API_TOKEN", raising=False)

    cfg = {
        "backend": "clef",
        "clef": {
            "account_id": "acc_abc123",
            "api_token": "token_xyz789",
            "model": "@cf/cloudflare/clef-flash",
            "timeout": 12.0,
        },
    }
    clf = classifier_from_config({"classifier": cfg})
    assert isinstance(clf, ClefClassifier)
    assert clf.account_id == "acc_abc123"
    assert clf.api_token == "token_xyz789"
    assert clf.model == "@cf/cloudflare/clef-flash"
    assert clf.timeout == 12.0
    assert clf.endpoint == "https://api.cloudflare.com/client/v4/accounts/acc_abc123/ai/run/@cf/cloudflare/clef-flash"


def test_clef_classifier_env_fallback(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "env_acc_456")
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "env_token_789")

    clf = classifier_from_config({"classifier": {"backend": "cloudflare"}})
    assert isinstance(clf, ClefClassifier)
    assert clf.account_id == "env_acc_456"
    assert clf.api_token == "env_token_789"
    assert clf.model == DEFAULT_CLEF_MODEL


def test_clef_classify_mock(monkeypatch):
    mock_response = {
        "result": {
            "answers": {
                "category": {
                    "type": "choice",
                    "choice": "coding",
                    "probabilities": {"coding": 0.96, "agentic": 0.04},
                    "confidence": 0.96,
                },
                "difficulty": {"type": "score", "score": 2.0, "confidence": 0.88},
                "needs_tools": {"type": "noul", "noul": 0.15},
                "needs_vision": {"type": "noul", "noul": 0.02},
                "needs_long_context": {"type": "noul", "noul": 0.05},
                "follow_up": {"type": "noul", "noul": 0.1},
                "stakes": {"type": "score", "score": 1.0, "confidence": 0.8},
            },
        },
        "success": True,
        "usage": {"input_tokens": 150, "output_tokens": 0},
    }

    client = ClefClassifier(account_id="test_acc", api_token="test_token")

    def mock_post(self, state, questions, timeout=None):
        assert "request" in state
        return mock_response, 0.038  # ~38ms 极速推理

    monkeypatch.setattr(ClefClassifier, "_post", mock_post)

    res = client("请帮我编写一个二分查找函数")
    assert isinstance(res, Classification)
    assert res.category == "coding"
    assert res.category_confidence == 0.96
    assert res.difficulty == 0.5  # 2.0 / (5 - 1)
    assert res.needs_tools == 0.15
    assert res.needs_vision == 0.02
    assert res.source_name == "clef[clef-flash]"
    assert res.failed is False


def test_clef_judge_mock(monkeypatch):
    mock_judge_response = {
        "result": {
            "answers": {
                "adequate": {"type": "noul", "noul": 0.95},
                "failure": {
                    "type": "choice",
                    "choice": "fine",
                    "probabilities": {"fine": 0.95, "wrong": 0.05},
                },
            },
        },
        "success": True,
    }

    client = ClefClassifier(account_id="test_acc", api_token="test_token")

    def mock_post(self, state, questions, timeout=None):
        assert "response" in state
        return mock_judge_response, 0.035

    monkeypatch.setattr(ClefClassifier, "_post", mock_post)

    j = client.judge("计算 1+1", "2")
    assert isinstance(j, Judgement)
    assert j.p_adequate == 0.95
    assert j.failure == "fine"
    assert j.failed is False


def test_clef_failover_to_fallback(monkeypatch):
    client = ClefClassifier(account_id="test_acc", api_token="test_token")

    def mock_post_fail(self, state, questions, timeout=None):
        raise urllib.error.URLError("Connection timed out")

    monkeypatch.setattr(ClefClassifier, "_post", mock_post_fail)

    # 遇到异常必须平滑返回 FALLBACK，不能抛出未捕获异常打崩主流程
    res = client("测试异常情况")
    assert res is FALLBACK
    assert res.failed is True
    assert res.source == "fallback"

    j = client.judge("测试异常", "无回答")
    assert j.failed is True
    assert j.p_adequate == 0.5


def test_clef_test_connection_no_token():
    client = ClefClassifier(account_id="test_acc", api_token="")
    res = client.test_connection()
    assert res["status"] == "error"
    assert "未配置 Cloudflare API Token" in res["error"]
