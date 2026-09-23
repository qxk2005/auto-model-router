"""HTML Report Generator for auto-llm-router-laya benchmark runs.

Produces completely self-contained, standalone, beautifully-styled HTML reports
with interactive filtering, embedded SVG charts, and comprehensive metrics.
"""

from __future__ import annotations

import html
import json
import os
import time
from pathlib import Path
from typing import Any

from evaluator import BenchmarkSummary, CaseResult


class ReportGenerator:
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate(self, summary: BenchmarkSummary, filename: str | None = None) -> str:
        """Generate a standalone HTML report file and return its absolute path."""
        ts_slug = time.strftime("%Y%m%d_%H%M%S")
        fname = filename or f"report_{summary.mode}_{ts_slug}.html"
        out_path = self.output_dir / fname

        html_content = self.render_html(summary)
        out_path.write_text(html_content, encoding="utf-8")
        return str(out_path.resolve())

    def render_html(self, summary: BenchmarkSummary) -> str:
        # JSON data for client-side filtering
        results_json = json.dumps([r.__dict__ for r in summary.results], ensure_ascii=False)
        model_dist_json = json.dumps(summary.model_distribution, ensure_ascii=False)
        cat_stats_json = json.dumps(summary.category_breakdown, ensure_ascii=False)

        # Precompute chart data
        exp_cost = summary.cost_expensive_total
        r_cost = summary.cost_router_total
        chp_cost = summary.cost_cheap_total
        max_cost = max(exp_cost, 0.0001)

        exp_bar_pct = 100.0
        r_bar_pct = round((r_cost / max_cost) * 100.0, 1)
        chp_bar_pct = round((chp_cost / max_cost) * 100.0, 1)

        cur_sym = getattr(summary, "currency_symbol", "¥")
        usd_rate = getattr(summary, "usd_cny_rate", 7.20)

        # Build Config Snapshot Section
        snapshot = getattr(summary, "config_snapshot", {}) or {}
        router_params = snapshot.get("router_params", {})
        laya_params = snapshot.get("laya_params", {})
        active_models = snapshot.get("active_models", [])

        snapshot_html = ""
        if router_params or laya_params or active_models:
            policy_disp = router_params.get("policy_display", router_params.get("policy_name", "F_expected"))
            rate_val = router_params.get("usd_cny_rate", usd_rate)
            stakes_val = router_params.get("stakes_usd", 2.0)
            detect_val = router_params.get("detect_probability", 0.6)
            mult_val = router_params.get("failure_cost_multiplier", 1.0)
            turns_val = router_params.get("remaining_turns_horizon", 3)

            dev_disp = laya_params.get("device_display", "Apple Metal (MPS) GPU 加速 [推荐 M4 Max]")
            backend_disp = laya_params.get("classifier_backend_display", "local (本地 Laya 引擎)")
            ckpt_val = laya_params.get("checkpoint", "convaiinnovations/laya")
            subfolder_val = laya_params.get("subfolder", "multilingual")
            chars_val = laya_params.get("request_chars_cap", 6000)

            models_rows = []
            for m in active_models:
                m_name = m.get("name", "")
                m_prov = m.get("provider", "none")
                m_up = m.get("upstream_id", m_name)
                is_free = m.get("is_free", False)
                p_in_cny = m.get("input_cny", 0.0)
                p_out_cny = m.get("output_cny", 0.0)
                p_in_usd = m.get("input_usd", 0.0)
                p_out_usd = m.get("output_usd", 0.0)
                ctx_k = int(m.get("context_tokens", 128000) / 1000)

                if is_free:
                    price_str = '<span style="color:#059669; font-weight:600;">¥0.00 / 免费 (0 成本)</span>'
                else:
                    price_str = f"输入: ¥{p_in_cny} (${p_in_usd:.4f}) / 输出: ¥{p_out_cny} (${p_out_usd:.4f})"

                models_rows.append(f"""
                <tr>
                  <td><strong style="color:var(--primary); font-family:var(--font-mono);">{m_name}</strong></td>
                  <td><span class="snapshot-badge snapshot-badge-blue">{m_prov}</span></td>
                  <td style="font-family:var(--font-mono); color:var(--text-muted);">{m_up}</td>
                  <td>{ctx_k}k</td>
                  <td>{price_str}</td>
                </tr>
                """)

            models_table_body = "".join(models_rows)

            snapshot_html = f"""
    <!-- Runtime Configuration & Environment Snapshot -->
    <section class="snapshot-panel">
      <div class="snapshot-header">
        <div class="panel-title">
          <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z"></path><path d="M15 12a3 3 0 11-6 0 3 3 0 016 0z"></path></svg>
          <span>评测运行配置与环境快照 (Runtime Configuration Snapshot)</span>
        </div>
        <span style="font-size: 12px; color: var(--text-muted);">以下参数与模型矩阵在本次测试运行前已实时固化</span>
      </div>

      <div class="snapshot-grid">
        <!-- Box 1: Router Policy Params -->
        <div class="snapshot-box">
          <div class="snapshot-box-title">
            <span>🧠 Router 路由决策策略与经济学超参数</span>
          </div>
          <div class="snapshot-kv-list">
            <div class="snapshot-kv-item" style="grid-column: span 2;">
              <span class="snapshot-k">当前生效决策策略 (Policy Name)</span>
              <span class="snapshot-v" style="color: var(--primary);">{policy_disp}</span>
            </div>
            <div class="snapshot-kv-item">
              <span class="snapshot-k">基准换算汇率 (USD/CNY)</span>
              <span class="snapshot-v">1 USD = {rate_val} CNY</span>
            </div>
            <div class="snapshot-kv-item">
              <span class="snapshot-k">未捕获错误故障惩罚 (Stakes)</span>
              <span class="snapshot-v">${stakes_val} USD</span>
            </div>
            <div class="snapshot-kv-item">
              <span class="snapshot-k">错误被及时发现概率 (Detect Prob)</span>
              <span class="snapshot-v">{detect_val}</span>
            </div>
            <div class="snapshot-kv-item">
              <span class="snapshot-k">失败成本倍率 (Multiplier)</span>
              <span class="snapshot-v">{mult_val}x</span>
            </div>
            <div class="snapshot-kv-item">
              <span class="snapshot-k">会话预期剩余轮次 (Turns)</span>
              <span class="snapshot-v">{turns_val} 轮</span>
            </div>
          </div>
        </div>

        <!-- Box 2: Laya Engine & Hardware Params -->
        <div class="snapshot-box">
          <div class="snapshot-box-title">
            <span>⚡ Laya 分类器与硬件加速配置</span>
          </div>
          <div class="snapshot-kv-list">
            <div class="snapshot-kv-item" style="grid-column: span 2;">
              <span class="snapshot-k">硬件加速设备 (Hardware Device)</span>
              <span class="snapshot-v"><span class="snapshot-badge snapshot-badge-green">{dev_disp}</span></span>
            </div>
            <div class="snapshot-kv-item">
              <span class="snapshot-k">分类器后端 (Backend)</span>
              <span class="snapshot-v">{backend_disp}</span>
            </div>
            <div class="snapshot-kv-item">
              <span class="snapshot-k">Checkpoint 权重分支</span>
              <span class="snapshot-v">{ckpt_val} ({subfolder_val})</span>
            </div>
            <div class="snapshot-kv-item">
              <span class="snapshot-k">提示词截断字符上限 (Cap)</span>
              <span class="snapshot-v">{chars_val} 字符</span>
            </div>
            <div class="snapshot-kv-item">
              <span class="snapshot-k">推理加速架构</span>
              <span class="snapshot-v">Metal MPS Non-Autoregressive</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Box 3: Active Routing Candidates Catalog Matrix -->
      <div class="snapshot-models-section">
        <div class="snapshot-models-title">
          <span>📋 本次测试参与路由的候选模型目录矩阵 (Active Candidates: {len(active_models)} 个)</span>
        </div>
        <div style="overflow-x: auto;">
          <table class="snapshot-models-table">
            <thead>
              <tr>
                <th>模型名称</th>
                <th>所属提供商</th>
                <th>上游端点模型 ID</th>
                <th>上下文上限</th>
                <th>计费单价 (每百万 Token)</th>
              </tr>
            </thead>
            <tbody>
              {models_table_body}
            </tbody>
          </table>
        </div>
      </div>
    </section>
    """

        template = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Auto-LLM-Router (Laya on Apple M4 Max) 评估报告</title>
  <style>
    :root {{
      --bg-main: #f8fafc;
      --card-bg: #ffffff;
      --card-border: #e2e8f0;
      --card-hover: #cbd5e1;
      --card-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.03);
      --primary: #0284c7;
      --primary-light: #e0f2fe;
      --primary-glow: rgba(2, 132, 199, 0.2);
      --success: #059669;
      --success-light: #ecfdf5;
      --warning: #d97706;
      --danger: #dc2626;
      --text-main: #0f172a;
      --text-muted: #475569;
      --text-dim: #64748b;
      --font-mono: 'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background-color: var(--bg-main);
      color: var(--text-main);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      min-height: 100vh;
      padding: 32px 24px;
      line-height: 1.5;
      -webkit-font-smoothing: antialiased;
    }}
    .container {{ max-width: 1360px; margin: 0 auto; }}
    
    /* Header */
    .header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 28px;
      padding-bottom: 20px;
      border-bottom: 1px solid var(--card-border);
      flex-wrap: wrap;
      gap: 16px;
    }}
    .title-group h1 {{
      font-size: 26px;
      font-weight: 700;
      background: linear-gradient(135deg, #0284c7, #4f46e5);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      margin-bottom: 6px;
    }}
    .title-group p {{
      color: var(--text-muted);
      font-size: 14px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      padding: 6px 14px;
      border-radius: 9999px;
      font-size: 12px;
      font-weight: 600;
      background: var(--primary-light);
      color: var(--primary);
      border: 1px solid rgba(2, 132, 199, 0.25);
    }}
    .badge-chip {{
      width: 8px; height: 8px; border-radius: 50%; background: var(--success);
      box-shadow: 0 0 6px var(--success);
    }}

    /* KPI Grid */
    .kpi-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 18px;
      margin-bottom: 28px;
    }}
    .kpi-card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 20px;
      box-shadow: var(--card-shadow);
      position: relative;
      overflow: hidden;
      transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }}
    .kpi-card:hover {{
      transform: translateY(-2px);
      border-color: var(--card-hover);
      box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }}
    .kpi-label {{
      font-size: 13px;
      color: var(--text-muted);
      margin-bottom: 8px;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .kpi-value {{
      font-size: 28px;
      font-weight: 700;
      font-family: var(--font-mono);
      color: var(--text-main);
    }}
    .kpi-highlight {{ color: var(--success); }}
    .kpi-sub {{
      font-size: 12px;
      color: var(--text-muted);
      margin-top: 6px;
    }}

    /* Snapshot Configuration Panel */
    .snapshot-panel {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 22px;
      box-shadow: var(--card-shadow);
      margin-bottom: 28px;
    }}
    .snapshot-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 18px;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .snapshot-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
      margin-bottom: 20px;
    }}
    @media (max-width: 860px) {{
      .snapshot-grid {{ grid-template-columns: 1fr; }}
    }}
    .snapshot-box {{
      background: #f8fafc;
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 16px 18px;
    }}
    .snapshot-box-title {{
      font-size: 13.5px;
      font-weight: 600;
      color: var(--text-main);
      margin-bottom: 12px;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .snapshot-kv-list {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px 14px;
      font-size: 12.5px;
    }}
    @media (max-width: 600px) {{
      .snapshot-kv-list {{ grid-template-columns: 1fr; }}
    }}
    .snapshot-kv-item {{
      display: flex;
      flex-direction: column;
      gap: 2px;
    }}
    .snapshot-k {{
      color: var(--text-muted);
      font-size: 11.5px;
    }}
    .snapshot-v {{
      font-weight: 600;
      color: var(--text-main);
      font-family: var(--font-mono);
      word-break: break-all;
    }}
    .snapshot-badge {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 11.5px;
      font-weight: 600;
      font-family: var(--font-mono);
    }}
    .snapshot-badge-blue {{
      background: #e0f2fe;
      color: #0369a1;
      border: 1px solid #bae6fd;
    }}
    .snapshot-badge-green {{
      background: #ecfdf5;
      color: #047857;
      border: 1px solid #a7f3d0;
    }}
    .snapshot-models-section {{
      background: #ffffff;
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 14px 16px;
    }}
    .snapshot-models-title {{
      font-size: 13px;
      font-weight: 600;
      color: var(--text-main);
      margin-bottom: 10px;
      display: flex;
      align-items: center;
      justify-content: space-between;
    }}
    .snapshot-models-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    .snapshot-models-table th, .snapshot-models-table td {{
      padding: 8px 12px;
      text-align: left;
      border-bottom: 1px solid #f1f5f9;
    }}
    .snapshot-models-table th {{
      background: #f8fafc;
      color: var(--text-muted);
      font-weight: 600;
      font-size: 11.5px;
    }}

    /* Charts Section */
    .charts-row {{
      display: grid;
      grid-template-columns: 1.2fr 1fr;
      gap: 20px;
      margin-bottom: 32px;
    }}
    @media (max-width: 900px) {{
      .charts-row {{ grid-template-columns: 1fr; }}
    }}
    .panel {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 22px;
      box-shadow: var(--card-shadow);
    }}
    .panel-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 18px;
    }}
    .panel-title {{
      font-size: 16px;
      font-weight: 600;
      color: var(--text-main);
      display: flex;
      align-items: center;
      gap: 8px;
    }}

    /* Cost Bar Comparison */
    .bar-group {{
      display: flex;
      flex-direction: column;
      gap: 16px;
      padding-top: 10px;
    }}
    .bar-item {{ display: flex; flex-direction: column; gap: 6px; }}
    .bar-meta {{ display: flex; justify-content: space-between; font-size: 13px; }}
    .bar-label {{ font-weight: 500; color: var(--text-main); }}
    .bar-cost {{ font-family: var(--font-mono); color: var(--text-main); font-weight: 600; }}
    .bar-track {{
      background: #f1f5f9;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      height: 22px;
      overflow: hidden;
      display: flex;
    }}
    .bar-fill {{
      height: 100%;
      border-radius: 5px;
      transition: width 0.8s cubic-bezier(0.16, 1, 0.3, 1);
    }}
    .bar-fill.expensive {{ background: linear-gradient(90deg, #ef4444, #f87171); }}
    .bar-fill.router {{ background: linear-gradient(90deg, #059669, #10b981); }}
    .bar-fill.cheap {{ background: linear-gradient(90deg, #0284c7, #38bdf8); }}

    /* Model Distribution Tag List */
    .model-dist-list {{
      display: flex;
      flex-direction: column;
      gap: 12px;
    }}
    .dist-row {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      padding: 10px 14px;
      background: #f8fafc;
      border: 1px solid var(--card-border);
      border-radius: 8px;
      font-size: 14px;
    }}
    .dist-model-name {{ font-family: var(--font-mono); font-weight: 600; color: var(--primary); }}
    .dist-count {{ font-weight: 700; color: var(--text-main); }}

    /* Table & Controls */
    .table-controls {{
      display: flex;
      gap: 12px;
      margin-bottom: 16px;
      flex-wrap: wrap;
    }}
    .search-box, .filter-select {{
      background: #ffffff;
      border: 1px solid #cbd5e1;
      border-radius: 8px;
      padding: 8px 14px;
      color: var(--text-main);
      font-size: 13px;
      outline: none;
      transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }}
    .search-box {{ flex: 1; min-width: 220px; }}
    .search-box:focus, .filter-select:focus {{
      border-color: var(--primary);
      box-shadow: 0 0 0 3px rgba(2, 132, 199, 0.15);
    }}
    
    .table-container {{
      overflow-x: auto;
      border: 1px solid var(--card-border);
      border-radius: 12px;
      background: var(--card-bg);
      box-shadow: var(--card-shadow);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      text-align: left;
      font-size: 13px;
    }}
    th {{
      background: #f8fafc;
      color: var(--text-muted);
      font-weight: 600;
      padding: 12px 14px;
      border-bottom: 1px solid var(--card-border);
      white-space: nowrap;
    }}
    td {{
      padding: 12px 14px;
      border-bottom: 1px solid #f1f5f9;
      color: var(--text-main);
    }}
    tr:hover td {{
      background: #f8fafc;
    }}
    .prompt-cell {{
      max-width: 320px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }}
    .tag {{
      display: inline-block;
      padding: 2px 8px;
      border-radius: 4px;
      font-size: 11px;
      font-weight: 600;
      text-transform: uppercase;
    }}
    .tag-easy {{ background: #ecfdf5; color: #059669; border: 1px solid rgba(5, 150, 105, 0.2); }}
    .tag-moderate {{ background: #fffbeb; color: #d97706; border: 1px solid rgba(217, 119, 6, 0.2); }}
    .tag-hard {{ background: #fef2f2; color: #dc2626; border: 1px solid rgba(220, 38, 38, 0.2); }}
    
    .model-badge {{
      font-family: var(--font-mono);
      font-size: 12px;
      font-weight: 600;
      color: var(--primary);
    }}
    .saving-cell {{
      font-family: var(--font-mono);
      color: var(--success);
      font-weight: 600;
    }}
    
    /* Footer */
    .footer {{
      margin-top: 36px;
      text-align: center;
      font-size: 12px;
      color: var(--text-dim);
      border-top: 1px solid var(--card-border);
      padding-top: 20px;
    }}
  </style>
</head>
<body>
  <div class="container">
    <!-- Header -->
    <header class="header">
      <div class="title-group">
        <h1>Auto-LLM-Router 评估报告 (Laya 驱动)</h1>
        <p>执行时间: {summary.timestamp} | 模式: {summary.mode} | 底座引擎: Laya (Apple Silicon M4 Max MPS 加速)</p>
      </div>
      <div class="badge">
        <span class="badge-chip"></span>
        <span>M4 Max Metal 加速已激活</span>
      </div>
    </header>

    <!-- Top KPI Cards -->
    <section class="kpi-grid">
      <div class="kpi-card">
        <div class="kpi-label"><span>💰 降本总幅度</span><span>%</span></div>
        <div class="kpi-value kpi-highlight">{summary.total_savings_pct}%</div>
        <div class="kpi-sub">节省金额: {cur_sym}{summary.total_savings_usd:.4f}</div>
      </div>
      
      <div class="kpi-card">
        <div class="kpi-label"><span>⚡ 智能路由总支出</span><span>{cur_sym}</span></div>
        <div class="kpi-value">{cur_sym}{summary.cost_router_total:.4f}</div>
        <div class="kpi-sub">全量昂贵对比: {cur_sym}{summary.cost_expensive_total:.4f}</div>
      </div>

      <div class="kpi-card">
        <div class="kpi-label"><span>🧠 Laya 决策平均耗时</span><span>MPS</span></div>
        <div class="kpi-value">{summary.avg_classifier_latency_ms} <span style="font-size: 16px; font-weight: normal; color: var(--text-muted);">ms</span></div>
        <div class="kpi-sub">单次前向极速反射分类</div>
      </div>

      <div class="kpi-card">
        <div class="kpi-label"><span>🎯 难度与模型匹配率</span><span>Accuracy</span></div>
        <div class="kpi-value" style="color: #38bdf8;">{summary.alignment_rate}%</div>
        <div class="kpi-sub">总测试用例数: {summary.total_cases} 条 (汇率: 1 USD = {usd_rate} CNY)</div>
      </div>
    </section>

    {snapshot_html}

    <!-- Visual Charts Row -->
    <div class="charts-row">
      <!-- Cost Comparison -->
      <div class="panel">
        <div class="panel-header">
          <div class="panel-title">
            <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
            <span>总成本对比 (Cost Comparison)</span>
          </div>
          <span style="font-size: 12px; color: var(--text-muted);">计费单位: {cur_sym} (基准汇率 1 USD = {usd_rate} CNY)</span>
        </div>
        
        <div class="bar-group">
          <div class="bar-item">
            <div class="bar-meta">
              <span class="bar-label" style="color: #dc2626; font-weight: 600;">全量使用主力旗舰模型 (Baseline)</span>
              <span class="bar-cost">{cur_sym}{exp_cost:.4f} (100%)</span>
            </div>
            <div class="bar-track">
              <div class="bar-fill expensive" style="width: {exp_bar_pct}%;"></div>
            </div>
          </div>

          <div class="bar-item">
            <div class="bar-meta">
              <span class="bar-label" style="color: #059669; font-weight: 600;">Laya 智能分级路由 (Auto-Router)</span>
              <span class="bar-cost">{cur_sym}{r_cost:.4f} ({r_bar_pct}%)</span>
            </div>
            <div class="bar-track">
              <div class="bar-fill router" style="width: {r_bar_pct}%;"></div>
            </div>
          </div>

          <div class="bar-item">
            <div class="bar-meta">
              <span class="bar-label" style="color: #0284c7; font-weight: 600;">全量使用本地/零成本模型</span>
              <span class="bar-cost">{cur_sym}{chp_cost:.4f} ({chp_bar_pct}%)</span>
            </div>
            <div class="bar-track">
              <div class="bar-fill cheap" style="width: {chp_bar_pct}%;"></div>
            </div>
          </div>
        </div>
      </div>

      <!-- Route Distribution -->
      <div class="panel">
        <div class="panel-header">
          <div class="panel-title">
            <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M11 3.055A9.001 9.001 0 1020.945 13H11V3.055z"></path><path d="M20.488 9H15V3.512A9.025 9.025 0 0120.488 9z"></path></svg>
            <span>路由模型分流分布 (Routing Distribution)</span>
          </div>
          <span style="font-size: 12px; color: var(--text-muted);">{summary.total_cases} 次调用</span>
        </div>

        <div class="model-dist-list" id="distList">
          <!-- Populated by JS -->
        </div>
      </div>
    </div>

    <!-- Test Case Ledger Panel -->
    <div class="panel">
      <div class="panel-header">
        <div class="panel-title">
          <svg width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2"></path></svg>
          <span>详细用例执行台账 (Case-by-Case Ledger)</span>
        </div>
        <span style="font-size: 13px; color: var(--text-muted);" id="tableCount">共 {summary.total_cases} 条</span>
      </div>

      <!-- Table Filter Bar -->
      <div class="table-controls">
        <input type="text" class="search-box" id="searchInput" placeholder="搜索 Prompt 或 Case ID..." oninput="filterTable()">
        <select class="filter-select" id="catFilter" onchange="filterTable()">
          <option value="all">全部分类 (Category: All)</option>
        </select>
        <select class="filter-select" id="diffFilter" onchange="filterTable()">
          <option value="all">全部难度 (Difficulty: All)</option>
          <option value="easy">Easy / Trivial</option>
          <option value="moderate">Moderate</option>
          <option value="hard">Hard / Frontier</option>
        </select>
        <select class="filter-select" id="modelFilter" onchange="filterTable()">
          <option value="all">全部路由目标 (Model: All)</option>
        </select>
      </div>

      <!-- Table -->
      <div class="table-container">
        <table id="casesTable">
          <thead>
            <tr>
              <th>ID</th>
              <th>分类</th>
              <th>难度</th>
              <th>Prompt 摘要</th>
              <th>Laya 分类 (难度/耗时)</th>
              <th>选定路由模型</th>
              <th>路由成本 ({cur_sym})</th>
              <th>节省幅度</th>
              <th>决策状态</th>
            </tr>
          </thead>
          <tbody id="tableBody">
            <!-- Populated by JS -->
          </tbody>
        </table>
      </div>
    </div>

    <!-- Footer -->
    <footer class="footer">
      <p>Auto-LLM-Router-Laya • 跑在 Apple Silicon M4 Max 上的低延迟智能模型分流系统</p>
    </footer>
  </div>

  <script>
    const results = {results_json};
    const modelDist = {model_dist_json};
    const catStats = {cat_stats_json};

    // Render Model Distribution List
    const distContainer = document.getElementById("distList");
    const totalCases = results.length;
    for (const [model, count] of Object.entries(modelDist)) {{
      const pct = totalCases > 0 ? ((count / totalCases) * 100).toFixed(1) : 0;
      const row = document.createElement("div");
      row.className = "dist-row";
      row.innerHTML = `
        <span class="dist-model-name">${{model}}</span>
        <div>
          <span class="dist-count">${{count}} 次</span>
          <span style="color: var(--text-muted); font-size: 12px; margin-left: 6px;">(${{pct}}%)</span>
        </div>
      `;
      distContainer.appendChild(row);
    }}

    // Populate Filters
    const catFilter = document.getElementById("catFilter");
    const categories = [...new Set(results.map(r => r.category))];
    categories.forEach(cat => {{
      const opt = document.createElement("option");
      opt.value = cat;
      opt.textContent = cat;
      catFilter.appendChild(opt);
    }});

    const modelFilter = document.getElementById("modelFilter");
    const models = [...new Set(results.map(r => r.chosen_model))];
    models.forEach(m => {{
      const opt = document.createElement("option");
      opt.value = m;
      opt.textContent = m;
      modelFilter.appendChild(opt);
    }});

    // Render Table
    function renderTable(data) {{
      const tbody = document.getElementById("tableBody");
      tbody.innerHTML = "";
      document.getElementById("tableCount").textContent = `显示 ${{data.length}} / ${{results.length}} 条`;

      data.forEach(r => {{
        const tr = document.createElement("tr");
        const diffClass = r.difficulty_tag.includes("hard") ? "tag-hard" : (r.difficulty_tag.includes("mod") ? "tag-moderate" : "tag-easy");
        const statusBadge = r.is_aligned 
          ? `<span style="color: var(--success); font-weight: 600;">✓ 匹配</span>` 
          : `<span style="color: var(--warning); font-weight: 600;">⚠ 偏置</span>`;

        tr.innerHTML = `
          <td style="font-family: var(--font-mono); font-size: 11px; color: var(--text-dim);">${{r.case_id}}</td>
          <td><span class="tag" style="background: #f1f5f9; color: var(--text-muted); border: 1px solid #e2e8f0;">${{r.category}}</span></td>
          <td><span class="tag ${{diffClass}}">${{r.difficulty_tag}}</span></td>
          <td class="prompt-cell" title="${{r.prompt}}">${{r.prompt}}</td>
          <td>
            <span style="font-family: var(--font-mono); font-size: 12px; color: var(--text-main);">diff: ${{r.detected_difficulty}}</span>
            <span style="color: var(--text-muted); font-size: 11px; margin-left: 4px;">(${{r.classifier_latency_ms}}ms)</span>
          </td>
          <td><span class="model-badge">${{r.chosen_model}}</span></td>
          <td style="font-family: var(--font-mono); color: var(--text-main);">${{r.cost_router > 0 ? '{cur_sym}'+r.cost_router.toFixed(5) : '{cur_sym}0.00 (免费)'}}</td>
          <td class="saving-cell">${{r.savings_pct > 0 ? '+'+r.savings_pct+'%' : '0%'}}</td>
          <td>${{statusBadge}}</td>
        `;
        tbody.appendChild(tr);
      }});
    }}

    function filterTable() {{
      const search = document.getElementById("searchInput").value.toLowerCase();
      const cat = document.getElementById("catFilter").value;
      const diff = document.getElementById("diffFilter").value;
      const model = document.getElementById("modelFilter").value;

      const filtered = results.filter(r => {{
        const matchesSearch = !search || r.prompt.toLowerCase().includes(search) || r.case_id.toLowerCase().includes(search);
        const matchesCat = cat === "all" || r.category === cat;
        const matchesDiff = diff === "all" || r.difficulty_tag.toLowerCase().includes(diff);
        const matchesModel = model === "all" || r.chosen_model === model;
        return matchesSearch && matchesCat && matchesDiff && matchesModel;
      }});

      renderTable(filtered);
    }}

    // Initial render
    renderTable(results);
  </script>
</body>
</html>
"""
        return template
