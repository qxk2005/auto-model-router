import pytest
from fastapi.testclient import TestClient
from server import app
from auto_router.config import normalize_provider_url

client = TestClient(app)

def test_normalize_provider_url():
    # 验证各种 base_url 格式的规范化及 /v1 自动补齐
    assert normalize_provider_url("http://8.148.249.98:88") == "http://8.148.249.98:88/v1"
    assert normalize_provider_url("http://8.148.249.98:88/") == "http://8.148.249.98:88/v1"
    assert normalize_provider_url("http://8.148.249.98:88/v1") == "http://8.148.249.98:88/v1"
    assert normalize_provider_url("http://8.148.249.98:88/v1/") == "http://8.148.249.98:88/v1"
    assert normalize_provider_url("http://8.148.249.98:88/chat/completions") == "http://8.148.249.98:88/v1"
    assert normalize_provider_url("http://8.148.249.98:88/v1/chat/completions") == "http://8.148.249.98:88/v1"
    assert normalize_provider_url("http://8.148.249.98:88/models") == "http://8.148.249.98:88/v1"
    assert normalize_provider_url("https://api.openai.com") == "https://api.openai.com/v1"
    assert normalize_provider_url("http://localhost:1234") == "http://localhost:1234/v1"
    assert normalize_provider_url("http://localhost:1234/v1") == "http://localhost:1234/v1"
    assert normalize_provider_url("${JUSHENG_BASE_URL}") == "http://8.148.249.98:88/v1"
    assert normalize_provider_url("https://api.anthropic.com", api="anthropic") == "https://api.anthropic.com/v1"
    assert normalize_provider_url("http://custom-proxy:9999", api="generic") == "http://custom-proxy:9999"


def test_fetch_provider_models_with_env_vars():
    # 测试提供商模型获取接口，验证环境变量能正确展开并自动补齐 /v1 获取模型
    res = client.post("/api/provider/models", json={
        "base_url": "${JUSHENG_BASE_URL}",
        "api_key": "${JUSHENG_API_KEY}"
    })
    assert res.status_code == 200
    data = res.json()
    assert data.get("status") == "ok"
    assert data.get("resolved_base_url") == "http://8.148.249.98:88/v1"
    assert data.get("count") >= 2
    model_ids = [m["id"] for m in data.get("models", [])]
    assert "deepseek-v4-flash" in model_ids
    assert "deepseek-v4-pro" in model_ids


def test_fetch_provider_models_missing_base_url():
    # 测试缺少 base_url 时，应返回合法的错误 JSON，而非未捕获 500
    res = client.post("/api/provider/models", json={
        "base_url": "",
        "api_key": ""
    })
    assert res.status_code == 200
    data = res.json()
    assert data.get("status") == "error"
    assert "缺少端点 Base URL" in data.get("error")


def test_provider_test_endpoint_with_env_vars():
    # 测试提供商连通性探测接口，验证自动补齐 /v1 且连通成功
    res = client.post("/api/provider/test", json={
        "base_url": "${JUSHENG_BASE_URL}",
        "api_key": "${JUSHENG_API_KEY}"
    })
    assert res.status_code == 200
    data = res.json()
    assert data.get("status") == "ok"
    assert data.get("resolved_base_url") == "http://8.148.249.98:88/v1"
    assert "deepseek-v4-flash" in data.get("available_models", [])


def test_probe_model_capability_auto_completes_v1():
    # 测试模型可用性与能力探测接口（重点验证具生涌动 deepseek-v4-pro 能自动补齐 /v1 路径，不再报 405）
    res = client.post("/api/model/probe", json={
        "provider_id": "具生涌动",
        "upstream_id": "deepseek-v4-pro",
        "prompt": "请用Python写一个快速排序函数，并用一句话解释其原理。"
    })
    assert res.status_code == 200
    data = res.json()
    assert data.get("status") == "ok"
    assert data.get("model") == "deepseek-v4-pro"
    assert data.get("resolved_base_url") == "http://8.148.249.98:88/v1"
    assert data.get("latency_ms") > 0
    assert "def quicksort" in data.get("reply_snippet", "") or len(data.get("reply_snippet", "")) > 10
