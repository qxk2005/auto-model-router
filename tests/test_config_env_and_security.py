import os
import json
import tempfile
import pytest
from pathlib import Path
from auto_router.config import Provider, RouterConfig, expand_env_vars, load_config

def test_expand_env_vars():
    os.environ["TEST_AMRA_KEY"] = "secret_12345"
    os.environ["TEST_AMRA_HOST"] = "https://api.test.com"
    
    # ${VAR} 格式
    assert expand_env_vars("${TEST_AMRA_KEY}") == "secret_12345"
    assert expand_env_vars("${TEST_AMRA_HOST}/v1") == "https://api.test.com/v1"
    
    # $VAR 格式
    assert expand_env_vars("$TEST_AMRA_KEY") == "secret_12345"
    
    # 不存在的环境变量返回空字符串
    assert expand_env_vars("${NOT_EXISTING_XYZ}") == ""
    
    # None 或 非字符串保持原样
    assert expand_env_vars(None) is None
    assert expand_env_vars(123) == 123

def test_provider_resolves_env_vars():
    os.environ["TEST_P_KEY"] = "sk-test-abc"
    os.environ["TEST_P_URL"] = "http://my-endpoint:8080/v1"
    
    p = Provider(
        name="test-prov",
        base_url="${TEST_P_URL}",
        api_key_literal="${TEST_P_KEY}"
    )
    
    assert p.api_key == "sk-test-abc"
    assert p.resolved_base_url == "http://my-endpoint:8080/v1"

def test_load_config_with_env_interpolation(tmp_path):
    os.environ["TEST_CONFIG_KEY"] = "sk-interp-999"
    os.environ["TEST_CONFIG_URL"] = "http://env-host:5000/v1"
    
    cfg_file = tmp_path / "router_config.json"
    data = {
        "providers": {
            "demo": {
                "name": "demo",
                "base_url": "${TEST_CONFIG_URL}",
                "api_key": "${TEST_CONFIG_KEY}"
            }
        },
        "models": []
    }
    cfg_file.write_text(json.dumps(data), encoding="utf-8")
    
    cfg = load_config(cfg_file, use_bench=False)
    prov = cfg.providers["demo"]
    assert prov.api_key == "sk-interp-999"
    assert prov.resolved_base_url == "http://env-host:5000/v1"
