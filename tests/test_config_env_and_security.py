import os
import json
import re
import tempfile
import pytest
from pathlib import Path
from auto_router.config import (
    Provider, RouterConfig, expand_env_vars, load_config, save_config,
    has_plain_secrets, generate_env_var_name, update_local_env_file, sanitize_config_for_public_repo
)

def test_expand_env_vars():
    os.environ["TEST_AMRA_KEY"] = "secret_12345"
    os.environ["TEST_AMRA_HOST"] = "https://api.test.com"
    
    # ${VAR} 格式
    assert expand_env_vars("${TEST_AMRA_KEY}") == "secret_12345"
    assert expand_env_vars("${TEST_AMRA_HOST}/v1") == "https://api.test.com/v1"
    
    # ${VAR:-default} 默认值格式
    assert expand_env_vars("${TEST_NOT_SET_VAR:-http://fallback:8080}") == "http://fallback:8080"
    assert expand_env_vars("${TEST_AMRA_HOST:-http://fallback:8080}") == "https://api.test.com"
    
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

def test_has_plain_secrets():
    # 占位符不应判定为明文密钥
    safe_data = {
        "providers": {
            "jusheng": {
                "name": "具生涌动",
                "base_url": "${JUSHENG_BASE_URL}",
                "api_key": "${JUSHENG_API_KEY}"
            }
        }
    }
    assert not has_plain_secrets(safe_data)
    
    # 包含真实明文 sk- 密钥必须被捕获
    leak_data = {
        "providers": {
            "jusheng": {
                "name": "具生涌动",
                "base_url": "http://1.2.3.4:80",
                "api_key": "sk-mock-provider-secret-key-1234567890abcdef"
            }
        }
    }
    assert has_plain_secrets(leak_data)

def test_save_config_auto_routes_plain_secrets_to_local(tmp_path, monkeypatch):
    # 当包含明文密钥且未指定绝对路径时，应自动路由到 local.json
    monkeypatch.chdir(tmp_path)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    
    leak_cfg = {
        "providers": {
            "jusheng": {
                "name": "具生涌动",
                "base_url": "http://1.2.3.4:80",
                "api_key": "sk-real-live-secret-key-1234567890"
            }
        }
    }
    saved = save_config(leak_cfg)
    assert "router_config.local.json" in saved
    assert (config_dir / "router_config.local.json").exists()
    assert not (config_dir / "router_config.json").exists()

def test_actual_router_config_is_clean_of_secrets():
    # 验证项目中当前的 config/router_config.json 绝不含有任何明文 sk- 密钥
    root = Path(__file__).resolve().parents[1]
    cfg_path = root / "config" / "router_config.json"
    if not cfg_path.exists():
        pytest.skip("config/router_config.json does not exist")
    
    text = cfg_path.read_text(encoding="utf-8")
    assert "${JUSHENG_API_KEY}" in text
    assert "${ANTIGRAVITY_API_KEY}" in text
    
    # 正则校验不含实际 sk- 密钥与明文 Token
    sk_matches = re.findall(r"sk-[a-zA-Z0-9]{20,}", text)
    assert not sk_matches, f"Found leaked keys in config/router_config.json: {sk_matches}"


def test_generate_env_var_name():
    assert generate_env_var_name("deepseek") == "DEEPSEEK_API_KEY"
    assert generate_env_var_name("openai") == "OPENAI_API_KEY"
    assert generate_env_var_name("具生涌动") == "JUSHENG_API_KEY"
    assert generate_env_var_name("月之暗面") == "MOONSHOT_API_KEY"
    assert generate_env_var_name("silicon-flow") == "SILICON_FLOW_API_KEY"


def test_update_local_env_file(tmp_path):
    env_file = tmp_path / ".env"
    update_local_env_file("CUSTOM_AMRA_KEY", "sk-test-secret-value-12345", env_path=env_file)
    content = env_file.read_text(encoding="utf-8")
    assert "CUSTOM_AMRA_KEY=sk-test-secret-value-12345" in content
    assert os.environ.get("CUSTOM_AMRA_KEY") == "sk-test-secret-value-12345"

    # 测试重复更新同一变量不会产生多份
    update_local_env_file("CUSTOM_AMRA_KEY", "sk-new-secret-value-67890", env_path=env_file)
    content2 = env_file.read_text(encoding="utf-8")
    assert "CUSTOM_AMRA_KEY=sk-new-secret-value-67890" in content2
    assert "sk-test-secret-value-12345" not in content2


def test_sanitize_config_for_public_repo():
    raw_payload = {
        "providers": {
            "minimax": {
                "name": "minimax",
                "base_url": "https://api.minimax.chat/v1",
                "api_key": "sk-live-plain-minimax-key-123456789"
            },
            "lm-studio": {
                "name": "lm-studio",
                "base_url": "http://localhost:1234/v1",
                "api_key": "lm-studio"
            },
            "jusheng": {
                "name": "具生涌动",
                "base_url": "http://1.2.3.4:88",
                "api_key": "${JUSHENG_API_KEY}"
            }
        }
    }

    sanitized, extracted = sanitize_config_for_public_repo(raw_payload)
    # 真实明文 Key 应被提取
    assert "MINIMAX_API_KEY" in extracted
    assert extracted["MINIMAX_API_KEY"] == "sk-live-plain-minimax-key-123456789"

    # 脱敏配置中应该被替换为环境变量引用
    assert sanitized["providers"]["minimax"]["api_key"] == "${MINIMAX_API_KEY}"

    # 原有的 lm-studio 和环境变量应原样保留
    assert sanitized["providers"]["lm-studio"]["api_key"] == "lm-studio"
    assert sanitized["providers"]["jusheng"]["api_key"] == "${JUSHENG_API_KEY}"

