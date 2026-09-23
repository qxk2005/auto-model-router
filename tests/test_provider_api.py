import pytest
from fastapi.testclient import TestClient
from server import app

client = TestClient(app)

def test_fetch_provider_models_with_env_vars():
    # 测试提供商模型获取接口，验证环境变量能正确展开并获取模型
    res = client.post("/api/provider/models", json={
        "base_url": "${JUSHENG_BASE_URL}",
        "api_key": "${JUSHENG_API_KEY}"
    })
    assert res.status_code == 200
    data = res.json()
    assert data.get("status") == "ok"
    assert data.get("resolved_base_url") == "http://8.148.249.98:88"
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
    # 测试提供商连通性探测接口
    res = client.post("/api/provider/test", json={
        "base_url": "${JUSHENG_BASE_URL}",
        "api_key": "${JUSHENG_API_KEY}"
    })
    assert res.status_code == 200
    data = res.json()
    assert data.get("status") == "ok"
    assert data.get("resolved_base_url") == "http://8.148.249.98:88"
    assert "deepseek-v4-flash" in data.get("available_models", [])
