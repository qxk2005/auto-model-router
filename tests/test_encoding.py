import json
import pytest
from pathlib import Path
from auto_router.config import _load_file, load_config
from evaluator import BenchmarkSummary
from reporter import ReportGenerator, fix_mojibake


def test_config_load_file_utf8_chinese(tmp_path):
    cfg_file = tmp_path / "router_config.json"
    data = {
        "providers": {
            "具生涌动": {
                "name": "具生涌动",
                "base_url": "http://localhost:8000/v1"
            }
        },
        "models": [
            {
                "name": "deepseek-v4-flash",
                "provider": "具生涌动",
                "upstream_id": "deepseek-v4-flash"
            }
        ]
    }
    # 模拟以标准 UTF-8 写入
    cfg_file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")

    loaded = _load_file(cfg_file)
    assert "具生涌动" in loaded["providers"]
    assert loaded["providers"]["具生涌动"]["name"] == "具生涌动"
    assert loaded["models"][0]["provider"] == "具生涌动"


def test_fix_mojibake_unscrambler():
    # 典型 UTF-8 被误按 GBK 解码的乱码
    mojibake = "\u934f\u98ce\u6553\u5a11\u5c7d\u59e9"  # 鍏风敂璇曞畾
    assert fix_mojibake(mojibake) == "具生涌动"

    # 正常中英文不应受影响
    assert fix_mojibake("具生涌动") == "具生涌动"
    assert fix_mojibake("lm-studio") == "lm-studio"
    assert fix_mojibake("") == ""
    assert fix_mojibake(None) == ""


def test_reporter_renders_clean_provider_badge(tmp_path):
    snapshot = {
        "router_params": {"policy_name": "B_naive"},
        "laya_params": {"device_display": "CUDA"},
        "active_models": [
            {
                "name": "deepseek-v4-flash",
                "provider": "具生涌动",
                "upstream_id": "deepseek-v4-flash",
                "context_tokens": 65536,
                "is_free": False,
                "input_cny": 1.0,
                "output_cny": 2.0,
                "input_usd": 0.1471,
                "output_usd": 0.2941,
            }
        ]
    }
    summary = BenchmarkSummary(
        timestamp="2026-09-23 23:45:00",
        mode="sim",
        total_cases=1,
        successful_cases=1,
        total_tokens=100,
        cost_router_total=0.001,
        cost_expensive_total=0.005,
        cost_cheap_total=0.0,
        total_savings_usd=0.004,
        total_savings_pct=80.0,
        avg_classifier_latency_ms=10.0,
        avg_total_latency_s=0.2,
        alignment_rate=1.0,
        model_distribution={"deepseek-v4-flash": 1},
        category_breakdown={},
        results=[],
        config_snapshot=snapshot,
    )
    generator = ReportGenerator(output_dir=str(tmp_path))
    out_file = generator.generate(summary, filename="report_test.html")
    html_content = Path(out_file).read_text(encoding="utf-8")

    assert '<span class="snapshot-badge snapshot-badge-blue">具生涌动</span>' in html_content
    assert "\u934f\u98ce" not in html_content
    assert "鍏风敂璇曞畾" not in html_content
