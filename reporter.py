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


def fix_mojibake(s: Any) -> str:
    """Detect and repair UTF-8 bytes mistakenly decoded as GBK/CP936 (e.g. 鍏风敂璇曞畾 -> 具生涌动)."""
    if not s or not isinstance(s, str):
        return str(s) if s is not None else ""
    try:
        repaired = s.encode("gbk").decode("utf-8")
        if len(repaired) < len(s):
            return repaired
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    return s


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
        trace_topology_json = json.dumps(getattr(summary, "trace_topology_stats", {}), ensure_ascii=False)

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
            policy_disp = fix_mojibake(router_params.get("policy_display", router_params.get("policy_name", "F_expected")))
            rate_val = router_params.get("usd_cny_rate", usd_rate)
            stakes_val = router_params.get("stakes_usd", 2.0)
            detect_val = router_params.get("detect_probability", 0.6)
            mult_val = router_params.get("failure_cost_multiplier", 1.0)
            turns_val = router_params.get("remaining_turns_horizon", 3)

            dev_disp = fix_mojibake(laya_params.get("device_display", "Apple Metal (MPS) GPU 加速 [推荐 M4 Max]"))
            backend_disp = fix_mojibake(laya_params.get("classifier_backend_display", "local (本地 Laya 引擎)"))
            ckpt_val = laya_params.get("checkpoint", "convaiinnovations/laya")
            subfolder_val = laya_params.get("subfolder", "multilingual")
            chars_val = laya_params.get("request_chars_cap", 6000)

            models_rows = []
            for m in active_models:
                m_name = fix_mojibake(m.get("name", ""))
                m_prov = fix_mojibake(m.get("provider", "none"))
                m_up = fix_mojibake(m.get("upstream_id", m_name))
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

    /* Route Trace Topology Panel */
    .trace-topology-panel {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 14px;
      padding: 22px;
      box-shadow: var(--card-shadow);
      margin-bottom: 28px;
    }}
    .topology-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 16px;
      flex-wrap: wrap;
      gap: 10px;
    }}
    .topology-container {{
      display: grid;
      grid-template-columns: 260px 1fr 280px;
      gap: 20px;
      align-items: center;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 12px;
      padding: 20px;
      position: relative;
    }}
    @media (max-width: 960px) {{
      .topology-container {{ grid-template-columns: 1fr; }}
    }}
    .topo-col {{
      display: flex;
      flex-direction: column;
      gap: 12px;
      z-index: 2;
    }}
    .topo-col-title {{
      font-size: 12px;
      font-weight: 700;
      color: var(--text-dim);
      text-transform: uppercase;
      letter-spacing: 0.5px;
      margin-bottom: 4px;
      display: flex;
      align-items: center;
      gap: 6px;
    }}
    .topo-node {{
      background: #ffffff;
      border: 1px solid var(--card-border);
      border-radius: 10px;
      padding: 12px 14px;
      box-shadow: 0 1px 3px rgba(0,0,0,0.04);
      cursor: pointer;
      transition: all 0.2s ease;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}
    .topo-node:hover {{
      transform: translateY(-2px);
      border-color: var(--primary);
      box-shadow: 0 4px 10px rgba(2, 132, 199, 0.12);
    }}
    .topo-node.active-filter {{
      border-color: var(--primary);
      background: #f0f9ff;
    }}
    .topo-node-title {{
      font-size: 13px;
      font-weight: 700;
      color: var(--text-main);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .topo-node-meta {{
      font-size: 11.5px;
      color: var(--text-muted);
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .topo-badge {{
      font-size: 11px;
      font-weight: 600;
      padding: 2px 6px;
      border-radius: 4px;
    }}

    /* ========================================================
       Jaeger / OpenTelemetry Trace Timeline (Pixel-grade APM)
       ======================================================== */
    .trace-btn {{
      background: #e0f2fe;
      color: #0369a1;
      border: 1px solid #bae6fd;
      padding: 4px 10px;
      border-radius: 6px;
      font-size: 11.5px;
      cursor: pointer;
      font-weight: 600;
      transition: all 0.15s ease;
      display: inline-flex;
      align-items: center;
      gap: 4px;
      user-select: none;
    }}
    .trace-btn:hover {{
      background: #bae6fd;
    }}
    .trace-drawer-tr {{
      background: #f8fafc !important;
    }}
    .trace-drawer-content {{
      padding: 16px 20px;
      border-top: 1px dashed #cbd5e1;
      border-bottom: 2px solid #cbd5e1;
    }}

    .jaeger-container {{
      background: #ffffff;
      border: 1px solid #e2e8f0;
      border-radius: 8px;
      overflow: hidden;
      box-shadow: 0 2px 10px rgba(0, 0, 0, 0.04);
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
      margin-bottom: 6px;
    }}
    
    /* Top Trace Bar */
    .jaeger-topbar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      padding: 10px 16px;
      background: #ffffff;
      border-bottom: 1px solid #f1f5f9;
    }}
    .jaeger-title-group {{
      display: flex;
      align-items: center;
      gap: 10px;
    }}
    .jaeger-icon-btn {{
      background: none;
      border: none;
      cursor: pointer;
      color: #64748b;
      display: inline-flex;
      align-items: center;
      padding: 4px;
      border-radius: 4px;
      transition: background 0.15s;
    }}
    .jaeger-icon-btn:hover {{
      background: #f1f5f9;
      color: #0f172a;
    }}
    .jaeger-service-name {{
      font-weight: 700;
      font-size: 15px;
      color: #0f172a;
    }}
    .jaeger-trace-id {{
      font-family: var(--font-mono);
      font-size: 12px;
      color: #64748b;
      background: #f1f5f9;
      padding: 2px 6px;
      border-radius: 4px;
      border: 1px solid #e2e8f0;
    }}
    .jaeger-top-actions {{
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .jaeger-find-box {{
      border: 1px solid #cbd5e1;
      border-radius: 4px;
      padding: 4px 10px;
      font-size: 12px;
      color: #334155;
      outline: none;
      width: 140px;
      background: #ffffff;
    }}
    .jaeger-find-box:focus {{
      border-color: #00a396;
    }}
    .jaeger-view-badge {{
      display: inline-flex;
      align-items: center;
      gap: 4px;
      background: #f1f5f9;
      color: #475569;
      border: 1px solid #e2e8f0;
      border-radius: 4px;
      padding: 4px 10px;
      font-size: 12px;
      font-weight: 600;
      cursor: default;
    }}

    /* Trace Meta Stats Row */
    .jaeger-meta-bar {{
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 20px;
      padding: 6px 16px;
      background: #f8fafc;
      border-bottom: 1px solid #e2e8f0;
      font-size: 12px;
      color: #64748b;
    }}
    .jaeger-meta-item strong {{
      color: #0f172a;
      font-weight: 600;
    }}

    /* Minimap Overview Timeline */
    .jaeger-minimap {{
      padding: 8px 16px;
      background: #fdfdfd;
      border-bottom: 1px solid #e2e8f0;
    }}
    .jaeger-minimap-ruler {{
      display: flex;
      justify-content: space-between;
      font-size: 10.5px;
      color: #94a3b8;
      font-family: var(--font-mono);
      margin-bottom: 4px;
      padding: 0 2px;
    }}
    .jaeger-minimap-track {{
      height: 18px;
      background: #f1f5f9;
      border: 1px solid #e2e8f0;
      border-radius: 4px;
      position: relative;
      overflow: hidden;
    }}
    .jaeger-minimap-bar {{
      position: absolute;
      top: 2px;
      bottom: 2px;
      border-radius: 2px;
      background: #00a396;
      opacity: 0.7;
    }}
    .jaeger-minimap-bar.classifier {{ background: #0284c7; }}
    .jaeger-minimap-bar.escalate {{ background: #f59e0b; }}
    .jaeger-minimap-bar.error {{ background: #ef4444; }}

    /* Split Grid Header */
    .jaeger-grid-header {{
      display: flex;
      background: #f8fafc;
      border-bottom: 1px solid #e2e8f0;
      font-size: 11.5px;
      color: #64748b;
      font-weight: 600;
      user-select: none;
    }}
    .jaeger-col-tree-head {{
      width: 320px;
      min-width: 320px;
      padding: 8px 14px;
      border-right: 1px solid #e2e8f0;
      display: flex;
      justify-content: space-between;
      align-items: center;
    }}
    .jaeger-col-timeline-head {{
      flex: 1;
      padding: 8px 14px;
      position: relative;
    }}
    .jaeger-ruler-ticks {{
      display: flex;
      justify-content: space-between;
      font-family: var(--font-mono);
      font-size: 11px;
      color: #64748b;
      width: 100%;
    }}

    /* Span Rows */
    .jaeger-span-wrapper {{
      border-bottom: 1px solid #f1f5f9;
      background: #ffffff;
    }}
    .jaeger-span-wrapper:hover {{
      background: #fafafa;
    }}
    .jaeger-span-row {{
      display: flex;
      align-items: center;
      min-height: 36px;
      cursor: pointer;
      position: relative;
    }}
    .jaeger-span-tree-cell {{
      width: 320px;
      min-width: 320px;
      padding: 6px 12px;
      border-right: 1px solid #e2e8f0;
      display: flex;
      align-items: center;
      gap: 6px;
      overflow: hidden;
      white-space: nowrap;
      text-overflow: ellipsis;
      box-sizing: border-box;
    }}
    .jaeger-indent-guide {{
      display: inline-block;
      width: 18px;
      height: 100%;
      border-left: 2px solid #e2e8f0;
      margin-right: 2px;
      flex-shrink: 0;
    }}
    .jaeger-color-strip {{
      width: 4px;
      height: 18px;
      border-radius: 2px;
      flex-shrink: 0;
    }}
    .jaeger-color-strip.svc-amra {{ background: #7c3aed; }}
    .jaeger-color-strip.svc-laya {{ background: #00a396; }}
    .jaeger-color-strip.svc-primary {{ background: #00a396; }}
    .jaeger-color-strip.svc-escalate {{ background: #f59e0b; }}
    .jaeger-color-strip.svc-error {{ background: #ef4444; }}

    .jaeger-svc-label {{
      font-size: 11.5px;
      color: #64748b;
      font-weight: 500;
      margin-right: 4px;
    }}
    .jaeger-op-label {{
      font-size: 12px;
      font-weight: 600;
      color: #0f172a;
    }}

    .jaeger-span-timeline-cell {{
      flex: 1;
      padding: 6px 14px;
      position: relative;
      height: 36px;
      display: flex;
      align-items: center;
      box-sizing: border-box;
    }}
    /* Vertical background grid reference lines */
    .jaeger-bg-grid {{
      position: absolute;
      top: 0;
      bottom: 0;
      left: 14px;
      right: 14px;
      display: flex;
      justify-content: space-between;
      pointer-events: none;
    }}
    .jaeger-bg-grid-line {{
      width: 1px;
      height: 100%;
      background: #f1f5f9;
    }}

    /* Span Bar */
    .jaeger-bar-item {{
      position: absolute;
      top: 8px;
      height: 20px;
      border-radius: 3px;
      display: flex;
      align-items: center;
      padding: 0 6px;
      font-size: 10.5px;
      font-weight: 600;
      color: #ffffff;
      box-shadow: 0 1px 2px rgba(0,0,0,0.1);
      transition: opacity 0.15s ease, filter 0.15s ease;
      z-index: 2;
      overflow: visible;
      white-space: nowrap;
    }}
    .jaeger-bar-item:hover {{
      filter: brightness(1.08);
      box-shadow: 0 2px 5px rgba(0,0,0,0.15);
    }}
    .jaeger-bar-item.svc-amra {{ background: #7c3aed; }}
    .jaeger-bar-item.svc-laya {{ background: #00a396; }}
    .jaeger-bar-item.svc-primary {{ background: #00a396; }}
    .jaeger-bar-item.svc-escalate {{ background: #f59e0b; }}
    .jaeger-bar-item.svc-error {{ background: #ef4444; }}

    .jaeger-bar-label-outside {{
      position: absolute;
      left: calc(100% + 6px);
      top: 50%;
      transform: translateY(-50%);
      font-family: var(--font-mono);
      font-size: 11px;
      font-weight: 600;
      color: #475569;
      pointer-events: none;
      white-space: nowrap;
    }}

    /* Span Detail Expanded Drawer (Pixel Jaeger Card) */
    .jaeger-detail-box {{
      background: #ffffff;
      border-top: 1px solid #f1f5f9;
      border-bottom: 2px solid #e2e8f0;
      padding: 14px 18px;
      display: flex;
      flex-direction: column;
      gap: 10px;
    }}
    .jaeger-detail-header {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      border-bottom: 1px solid #f1f5f9;
      padding-bottom: 6px;
    }}
    .jaeger-detail-title {{
      font-size: 13.5px;
      font-weight: 700;
      color: #0f172a;
      display: flex;
      align-items: center;
      gap: 8px;
    }}
    .jaeger-detail-meta {{
      font-size: 11.5px;
      color: #64748b;
      display: flex;
      gap: 14px;
    }}
    .jaeger-detail-meta strong {{
      color: #0f172a;
    }}

    .jaeger-tags-group {{
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}
    .jaeger-section-title {{
      font-size: 11.5px;
      font-weight: 700;
      color: #475569;
      cursor: pointer;
      display: flex;
      align-items: center;
      gap: 4px;
      user-select: none;
    }}
    .jaeger-tags-grid {{
      display: flex;
      flex-wrap: wrap;
      gap: 6px;
      margin-top: 3px;
    }}
    .jaeger-tag-badge {{
      display: inline-flex;
      align-items: center;
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 4px;
      padding: 2px 7px;
      font-size: 11px;
      font-family: var(--font-mono);
    }}
    .jaeger-tag-key {{
      color: #64748b;
      margin-right: 4px;
    }}
    .jaeger-tag-val {{
      color: #0f172a;
      font-weight: 600;
    }}

    .jaeger-logs-box {{
      background: #f8fafc;
      border: 1px solid #e2e8f0;
      border-radius: 6px;
      padding: 8px 12px;
      font-size: 11.5px;
      display: flex;
      flex-direction: column;
      gap: 4px;
    }}
    .jaeger-log-item {{
      display: flex;
      gap: 10px;
      align-items: flex-start;
      font-size: 11px;
    }}
    .jaeger-log-time {{
      font-family: var(--font-mono);
      color: #00a396;
      font-weight: 600;
      min-width: 50px;
    }}
    .jaeger-log-event {{
      font-weight: 600;
      color: #334155;
    }}
    .jaeger-log-payload {{
      color: #64748b;
      word-break: break-all;
    }}

    .jaeger-span-footer {{
      display: flex;
      justify-content: flex-end;
      align-items: center;
      font-size: 11px;
      color: #94a3b8;
      gap: 6px;
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

    <!-- Route Trace Flow Topology Panel (Macro View) -->
    <section class="trace-topology-panel" id="traceTopologySection">
      <div class="topology-header">
        <div class="panel-title">
          <svg width="20" height="20" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg>
          <span>全链路路由追踪拓扑流向图 (Route Trace Topology Flow)</span>
        </div>
        <div style="display: flex; gap: 8px; align-items: center;">
          <span style="font-size: 12px; color: var(--text-muted);">点击节点可过滤下方表格</span>
          <button type="button" onclick="resetTopoFilter()" style="padding: 2px 8px; font-size: 11.5px; border-radius: 4px; border: 1px solid #cbd5e1; background: #fff; cursor: pointer;">重置筛选</button>
        </div>
      </div>

      <div class="topology-container" id="topologyContainer">
        <!-- Col 1: Laya Decision Hub -->
        <div class="topo-col">
          <div class="topo-col-title">🧠 阶段 1: 硬件加速决策中枢</div>
          <div class="topo-node active-filter" id="nodeLaya" onclick="filterByTraceStage('all')">
            <div class="topo-node-title">
              <span>Laya 意图与特征分类</span>
              <span class="topo-badge" style="background:#e0f2fe; color:#0284c7;">{summary.avg_classifier_latency_ms} ms</span>
            </div>
            <div class="topo-node-meta">
              <span>全量请求统一极速仲裁</span>
              <strong>{summary.total_cases} 次</strong>
            </div>
          </div>
        </div>

        <!-- Col 2: Primary Dispatched Models -->
        <div class="topo-col" id="topoPrimaryCol">
          <div class="topo-col-title">🚀 阶段 2: 首次模型调度池</div>
          <!-- Populated by JS -->
        </div>

        <!-- Col 3: Final Outcomes & Escalation -->
        <div class="topo-col">
          <div class="topo-col-title">🎯 阶段 3: 质量裁决与终局流向</div>
          <div class="topo-node" id="nodeDirectSuccess" onclick="filterByTraceStage('direct')">
            <div class="topo-node-title">
              <span style="color:#059669;">✓ 一跳直接解决 (Direct)</span>
              <span class="topo-badge" style="background:#ecfdf5; color:#059669;" id="badgeDirectCount">-- 次</span>
            </div>
            <div class="topo-node-meta">
              <span>首选模型生成达标验收通过</span>
              <strong id="badgeDirectPct">--%</strong>
            </div>
          </div>

          <div class="topo-node" id="nodeEscalated" onclick="filterByTraceStage('escalated')">
            <div class="topo-node-title">
              <span style="color:#7c3aed;">🔄 重新转发升级/容灾 (Multi-hop)</span>
              <span class="topo-badge" style="background:#f5f3ff; color:#7c3aed;" id="badgeEscalateCount">-- 次</span>
            </div>
            <div class="topo-node-meta">
              <span>质量不满足升级或故障容灾</span>
              <strong id="badgeEscalatePct">--%</strong>
            </div>
          </div>
        </div>
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
              <th>全链路追踪 (Trace)</th>
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
      <p>Auto-LLM-Router-Laya • 跑在本地加速硬件上的低延迟智能模型分流系统</p>
    </footer>
  </div>

  <script>
    const results = {results_json};
    const modelDist = {model_dist_json};
    const catStats = {cat_stats_json};
    const traceTopology = {trace_topology_json};

    let activeTraceFilter = "all";

    // 1. Initialize Route Trace Topology Panel (Macro View)
    function initTopologyPanel() {{
      const totalCases = results.length;
      const directCases = results.filter(r => !r.escalated && !r.error).length;
      const escalatedCases = results.filter(r => r.escalated).length;

      const directPct = totalCases > 0 ? ((directCases / totalCases) * 100).toFixed(1) : 0;
      const escPct = totalCases > 0 ? ((escalatedCases / totalCases) * 100).toFixed(1) : 0;

      const badgeDirectCount = document.getElementById("badgeDirectCount");
      const badgeDirectPct = document.getElementById("badgeDirectPct");
      const badgeEscCount = document.getElementById("badgeEscalateCount");
      const badgeEscPct = document.getElementById("badgeEscalatePct");

      if (badgeDirectCount) badgeDirectCount.textContent = `${{directCases}} 次`;
      if (badgeDirectPct) badgeDirectPct.textContent = `${{directPct}}%`;
      if (badgeEscCount) badgeEscCount.textContent = `${{escalatedCases}} 次`;
      if (badgeEscPct) badgeEscPct.textContent = `${{escPct}}%`;

      const primaryCol = document.getElementById("topoPrimaryCol");
      if (primaryCol) {{
        for (const [model, count] of Object.entries(modelDist)) {{
          const pct = totalCases > 0 ? ((count / totalCases) * 100).toFixed(1) : 0;
          const node = document.createElement("div");
          node.className = "topo-node";
          node.id = `nodeModel_${{model.replace(/[^a-zA-Z0-9]/g, '_')}}`;
          node.onclick = () => filterByTraceStage(`model:${{model}}`);
          node.innerHTML = `
            <div class="topo-node-title">
              <span style="font-family: var(--font-mono); color: var(--primary);">${{model}}</span>
              <span class="topo-badge" style="background:#e0f2fe; color:#0369a1;">${{count}} 次</span>
            </div>
            <div class="topo-node-meta">
              <span>初选用例占比</span>
              <strong>${{pct}}%</strong>
            </div>
          `;
          primaryCol.appendChild(node);
        }}
      }}
    }}

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

    // Format duration adaptively (µs, ms, s)
    function formatJaegerDuration(ms) {{
      if (ms === undefined || ms === null || isNaN(ms)) return "0ms";
      if (ms < 1.0) {{
        return `${{Math.round(ms * 1000)}}µs`;
      }} else if (ms < 1000.0) {{
        return `${{ms < 10 ? ms.toFixed(2) : ms.toFixed(1)}}ms`;
      }} else {{
        return `${{(ms / 1000.0).toFixed(2)}}s`;
      }}
    }}

    // Toggle specific span detail drawer
    function toggleJaegerSpanDetail(spanKey) {{
      const d = document.getElementById(`detail_${{spanKey}}`);
      const arrow = document.getElementById(`arrow_${{spanKey}}`);
      if (!d) return;
      const isHidden = d.style.display === "none";
      d.style.display = isHidden ? "flex" : "none";
      if (arrow) arrow.textContent = isHidden ? "▼" : "▶";
    }}

    // Toggle subsection (Tags, Process, Logs)
    function toggleSubSection(id, titleEl) {{
      const el = document.getElementById(id);
      if (!el) return;
      const isHidden = el.style.display === "none";
      el.style.display = isHidden ? "" : "none";
      const span = titleEl.querySelector("span");
      if (span) {{
        const text = span.textContent.replace(/^[▼▶]\s*/, "");
        span.textContent = (isHidden ? "▼ " : "▶ ") + text;
      }}
    }}

    // Expand all spans in a trace
    function expandAllSpans(caseId) {{
      const container = document.getElementById(`spans_container_${{caseId}}`);
      if (!container) return;
      container.querySelectorAll(".jaeger-detail-box").forEach(b => b.style.display = "flex");
      container.querySelectorAll("[id^='arrow_']").forEach(a => a.textContent = "▼");
    }}

    // Collapse all spans in a trace
    function collapseAllSpans(caseId) {{
      const container = document.getElementById(`spans_container_${{caseId}}`);
      if (!container) return;
      container.querySelectorAll(".jaeger-detail-box").forEach(b => b.style.display = "none");
      container.querySelectorAll("[id^='arrow_']").forEach(a => a.textContent = "▶");
    }}

    function toggleAllSpans(caseId) {{
      const container = document.getElementById(`spans_container_${{caseId}}`);
      if (!container) return;
      const anyVisible = Array.from(container.querySelectorAll(".jaeger-detail-box")).some(b => b.style.display !== "none");
      if (anyVisible) {{
        collapseAllSpans(caseId);
      }} else {{
        expandAllSpans(caseId);
      }}
    }}

    // Find / Filter spans within a trace
    function findJaegerSpans(caseId, query) {{
      const container = document.getElementById(`spans_container_${{caseId}}`);
      if (!container) return;
      const q = (query || "").trim().toLowerCase();
      const wrappers = container.querySelectorAll(".jaeger-span-wrapper");
      wrappers.forEach(w => {{
        if (!q) {{
          w.style.display = "";
          w.style.opacity = "1";
        }} else {{
          const txt = w.textContent.toLowerCase();
          if (txt.includes(q)) {{
            w.style.display = "";
            w.style.opacity = "1";
          }} else {{
            w.style.opacity = "0.2";
          }}
        }}
      }});
    }}

    // Toggle Waterfall Trace Drawer for a single case
    function toggleTraceRow(caseId, btnEl) {{
      const drawer = document.getElementById(`trace_drawer_${{caseId}}`);
      if (!drawer) return;
      const isVisible = drawer.style.display !== "none";
      if (isVisible) {{
        drawer.style.display = "none";
        btnEl.innerHTML = "<span>🔍 展开链路 ▾</span>";
        btnEl.style.background = "#e0f2fe";
      }} else {{
        drawer.style.display = "table-row";
        btnEl.innerHTML = "<span>收起链路 ▴</span>";
        btnEl.style.background = "#bae6fd";
      }}
    }}

    // Render Table with Waterfall Drawer
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

        const isEscalated = r.escalated || false;
        const hopBadge = isEscalated
          ? `<span class="badge" style="background:#f5f3ff; color:#7c3aed; font-size:11px; padding:2px 8px;">🔄 2跳 (升级/容灾)</span>`
          : `<span class="badge" style="background:#ecfdf5; color:#059669; font-size:11px; padding:2px 8px;">✓ 1跳直达</span>`;

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
          <td style="white-space: nowrap;">
            <div style="display:flex; align-items:center; gap:6px;">
              ${{hopBadge}}
              <button type="button" class="trace-btn" onclick="toggleTraceRow('${{r.case_id}}', this)">
                <span>🔍 展开链路 ▾</span>
              </button>
            </div>
          </td>
        `;
        tbody.appendChild(tr);

        // Build Trace Drawer Row (Pixel-grade Jaeger APM Timeline)
        const trDrawer = document.createElement("tr");
        trDrawer.className = "trace-drawer-tr";
        trDrawer.id = `trace_drawer_${{r.case_id}}`;
        trDrawer.style.display = "none";

        const chain = r.trace_chain || [];
        const rootSpan = chain.find(s => s.depth === 1) || chain[0] || {{}};
        const totDur = Math.max(0.1, rootSpan.duration_ms || (chain.length > 0 ? chain.reduce((acc, s) => Math.max(acc, (s.start_ms || 0) + (s.duration_ms || 0)), 0) : 1));
        const traceId = r.trace_id || rootSpan.span_id || `tr_${{r.case_id}}`;
        const servicesSet = new Set(chain.map(s => s.service).filter(Boolean));
        const serviceCount = Math.max(1, servicesSet.size);

        // 5-Tick Ruler Labels
        const tick0 = formatJaegerDuration(0);
        const tick25 = formatJaegerDuration(totDur * 0.25);
        const tick50 = formatJaegerDuration(totDur * 0.50);
        const tick75 = formatJaegerDuration(totDur * 0.75);
        const tick100 = formatJaegerDuration(totDur);

        // Minimap bars
        let minimapBarsHtml = "";
        chain.forEach((sp) => {{
          const stStart = sp.start_ms || 0;
          const stDur = sp.duration_ms || 0.1;
          const leftPct = Math.min(98, Math.max(0, (stStart / totDur) * 100));
          const widthPct = Math.min(100 - leftPct, Math.max(1, (stDur / totDur) * 100));
          let bCls = "primary";
          if (sp.stage_type === "classifier") bCls = "classifier";
          else if (sp.stage_type === "escalate_model" || sp.status === "escalated" || sp.status === "escalate_recommended") bCls = "escalate";
          else if (sp.status === "error" || sp.status === "timeout") bCls = "error";

          minimapBarsHtml += `<div class="jaeger-minimap-bar ${{bCls}}" style="left: ${{leftPct.toFixed(1)}}%; width: ${{widthPct.toFixed(1)}}%;" title="${{sp.operation}}: +${{formatJaegerDuration(stDur)}}"></div>`;
        }});

        // Spans List Rows & Detail Drawers
        let spansListHtml = "";
        chain.forEach((sp, sIdx) => {{
          const stStart = sp.start_ms || 0;
          const stDur = sp.duration_ms || 0.1;
          const leftPct = Math.min(96, Math.max(0, (stStart / totDur) * 100));
          const widthPct = Math.min(100 - leftPct, Math.max(2, (stDur / totDur) * 100));

          const isError = sp.status === "error" || sp.status === "timeout";
          const isEscalate = sp.status === "escalated" || sp.status === "escalate_recommended" || sp.stage_type === "escalate_model";
          
          let colorClass = "svc-primary";
          if (isError) colorClass = "svc-error";
          else if (isEscalate) colorClass = "svc-escalate";
          else if (sp.service === "amra") colorClass = "svc-amra";
          else if (sp.service === "laya") colorClass = "svc-laya";

          const indentGuide = sp.depth === 2 ? `<span class="jaeger-indent-guide"></span>` : "";
          const opLabel = sp.operation || sp.name || "operation";
          const svcLabel = sp.service || "service";
          const spanKey = `${{r.case_id}}_${{sp.span_id || sIdx}}`;

          // Format Tags Badges
          const tags = sp.tags || {{}};
          const tagBadges = Object.entries(tags).map(([k, v]) => `
            <span class="jaeger-tag-badge">
              <span class="jaeger-tag-key">${{k}} =</span>
              <span class="jaeger-tag-val">${{v}}</span>
            </span>
          `).join("");

          // Format Process Badges
          const proc = sp.process || {{ "env": "benchmark", "runtime": "python3.11" }};
          const procBadges = Object.entries(proc).map(([k, v]) => `
            <span class="jaeger-tag-badge">
              <span class="jaeger-tag-key">${{k}} =</span>
              <span class="jaeger-tag-val">${{v}}</span>
            </span>
          `).join("");

          // Format Logs
          const logs = sp.logs || [];
          let logsHtml = "";
          if (logs.length > 0) {{
            logsHtml = logs.map(l => `
              <div class="jaeger-log-item">
                <span class="jaeger-log-time">${{formatJaegerDuration(l.time_ms || 0)}}</span>
                <span class="jaeger-log-event">${{l.event || 'event'}}:</span>
                <span class="jaeger-log-payload">${{l.payload || l.message || ''}}</span>
              </div>
            `).join("");
          }} else if (sp.snippet) {{
            logsHtml = `
              <div class="jaeger-log-item">
                <span class="jaeger-log-time">${{formatJaegerDuration(stDur)}}</span>
                <span class="jaeger-log-event">payload_snippet:</span>
                <span class="jaeger-log-payload">${{sp.snippet}}</span>
              </div>
            `;
          }}

          spansListHtml += `
            <div class="jaeger-span-wrapper" id="span_wrap_${{spanKey}}">
              <div class="jaeger-span-row" onclick="toggleJaegerSpanDetail('${{spanKey}}')">
                <div class="jaeger-span-tree-cell">
                  ${{indentGuide}}
                  <span style="font-size: 10px; color: #94a3b8; margin-right: 2px;" id="arrow_${{spanKey}}">▼</span>
                  <span class="jaeger-color-strip ${{colorClass}}"></span>
                  <span class="jaeger-svc-label">${{svcLabel}}</span>
                  <span class="jaeger-op-label">${{opLabel}}</span>
                </div>
                <div class="jaeger-span-timeline-cell">
                  <div class="jaeger-bg-grid">
                    <span class="jaeger-bg-grid-line"></span>
                    <span class="jaeger-bg-grid-line"></span>
                    <span class="jaeger-bg-grid-line"></span>
                    <span class="jaeger-bg-grid-line"></span>
                    <span class="jaeger-bg-grid-line"></span>
                  </div>
                  <div class="jaeger-bar-item ${{colorClass}}" style="left: ${{leftPct.toFixed(1)}}%; width: ${{widthPct.toFixed(1)}}%;">
                    ${{widthPct > 15 ? formatJaegerDuration(stDur) : ''}}
                    <span class="jaeger-bar-label-outside">${{widthPct <= 15 ? formatJaegerDuration(stDur) : ''}}</span>
                  </div>
                </div>
              </div>

              <!-- Span Detail Card (Expanded by default for Root or clicked) -->
              <div class="jaeger-detail-box" id="detail_${{spanKey}}" style="${{sIdx === 0 ? '' : 'display: none;'}}">
                <div class="jaeger-detail-header">
                  <div class="jaeger-detail-title">
                    <span class="jaeger-color-strip ${{colorClass}}"></span>
                    <span>${{opLabel}}</span>
                  </div>
                  <div class="jaeger-detail-meta">
                    <span>Service: <strong>${{svcLabel}}</strong></span>
                    <span>Duration: <strong>${{formatJaegerDuration(stDur)}}</strong></span>
                    <span>Start Time: <strong>${{formatJaegerDuration(stStart)}}</strong></span>
                  </div>
                </div>

                <div class="jaeger-tags-group">
                  <div class="jaeger-section-title" onclick="toggleSubSection('tags_${{spanKey}}', this)">
                    <span>▼ Tags (${{Object.keys(tags).length}})</span>
                  </div>
                  <div class="jaeger-tags-grid" id="tags_${{spanKey}}">
                    ${{tagBadges}}
                  </div>
                </div>

                <div class="jaeger-tags-group">
                  <div class="jaeger-section-title" onclick="toggleSubSection('proc_${{spanKey}}', this)">
                    <span>▼ Process (${{Object.keys(proc).length}})</span>
                  </div>
                  <div class="jaeger-tags-grid" id="proc_${{spanKey}}">
                    ${{procBadges}}
                  </div>
                </div>

                ${{logsHtml ? `
                <div class="jaeger-tags-group">
                  <div class="jaeger-section-title" onclick="toggleSubSection('logs_${{spanKey}}', this)">
                    <span>▼ Logs (${{logs.length || 1}})</span>
                  </div>
                  <div class="jaeger-logs-box" id="logs_${{spanKey}}">
                    ${{logsHtml}}
                  </div>
                </div>
                ` : ''}}

                <div class="jaeger-span-footer">
                  <span>SpanID: <strong style="font-family: var(--font-mono); color: #475569;">${{sp.span_id || 'span_' + sIdx}}</strong></span>
                  <svg width="12" height="12" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M10 13a5 5 0 007.54.54l3-3a5 5 0 00-7.07-7.07l-1.72 1.71"></path><path d="M14 11a5 5 0 00-7.54-.54l-3 3a5 5 0 007.07 7.07l1.71-1.71"></path></svg>
                </div>
              </div>
            </div>
          `;
        }});

        trDrawer.innerHTML = `
          <td colspan="10" class="trace-drawer-content">
            <div class="jaeger-container" id="jaeger_container_${{r.case_id}}">
              <!-- Header Topbar -->
              <div class="jaeger-topbar">
                <div class="jaeger-title-group">
                  <button type="button" class="jaeger-icon-btn" title="收起链路" onclick="toggleTraceRow('${{r.case_id}}', document.querySelector('#casesTable button[onclick*=\\'${{r.case_id}}\\''))">
                    <svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M10 19l-7-7m0 0l7-7m-7 7h18"></path></svg>
                  </button>
                  <button type="button" class="jaeger-icon-btn" onclick="toggleAllSpans('${{r.case_id}}')">
                    <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M19 9l-7 7-7-7"></path></svg>
                  </button>
                  <span class="jaeger-service-name">${{rootSpan.operation || 'amra: /v1/chat/completions'}}</span>
                  <span class="jaeger-trace-id" title="点击复制 Trace ID" onclick="navigator.clipboard.writeText('${{traceId}}')">${{traceId}}</span>
                </div>
                <div class="jaeger-top-actions">
                  <input type="text" class="jaeger-find-box" placeholder="Find in trace..." oninput="findJaegerSpans('${{r.case_id}}', this.value)">
                  <div class="jaeger-view-badge">
                    <svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M4 6h16M4 12h16M4 18h7"></path></svg>
                    <span>Trace Timeline ▾</span>
                  </div>
                </div>
              </div>

              <!-- Meta Stats Bar -->
              <div class="jaeger-meta-bar">
                <div class="jaeger-meta-item">Trace Start: <strong>00:00:00.000</strong></div>
                <div class="jaeger-meta-item">Duration: <strong style="color: #00a396;">${{formatJaegerDuration(totDur)}}</strong></div>
                <div class="jaeger-meta-item">Services: <strong>${{serviceCount}}</strong></div>
                <div class="jaeger-meta-item">Depth: <strong>2</strong></div>
                <div class="jaeger-meta-item">Total Spans: <strong>${{chain.length}}</strong></div>
              </div>

              <!-- Minimap Overview Timeline -->
              <div class="jaeger-minimap">
                <div class="jaeger-minimap-ruler">
                  <span>${{tick0}}</span>
                  <span>${{tick25}}</span>
                  <span>${{tick50}}</span>
                  <span>${{tick75}}</span>
                  <span>${{tick100}}</span>
                </div>
                <div class="jaeger-minimap-track">
                  ${{minimapBarsHtml}}
                </div>
              </div>

              <!-- Split Grid Header -->
              <div class="jaeger-grid-header">
                <div class="jaeger-col-tree-head">
                  <span>Service & Operation</span>
                  <div style="display: flex; gap: 6px;">
                    <span style="cursor: pointer;" title="全部展开" onclick="expandAllSpans('${{r.case_id}}')">∨</span>
                    <span style="cursor: pointer;" title="全部收起" onclick="collapseAllSpans('${{r.case_id}}')">&gt;</span>
                  </div>
                </div>
                <div class="jaeger-col-timeline-head">
                  <div class="jaeger-ruler-ticks">
                    <span>${{tick0}}</span>
                    <span>${{tick25}}</span>
                    <span>${{tick50}}</span>
                    <span>${{tick75}}</span>
                    <span>${{tick100}}</span>
                  </div>
                </div>
              </div>

              <!-- Spans Rows List -->
              <div class="jaeger-spans-container" id="spans_container_${{r.case_id}}">
                ${{spansListHtml}}
              </div>
            </div>
          </td>
        `;
        tbody.appendChild(trDrawer);
      }});
    }}

    // Filter by Topo Stage Node Click
    function filterByTraceStage(stage) {{
      activeTraceFilter = stage;
      document.querySelectorAll(".topo-node").forEach(n => n.classList.remove("active-filter"));

      if (stage === "all") {{
        const node = document.getElementById("nodeLaya");
        if (node) node.classList.add("active-filter");
      }} else if (stage === "direct") {{
        const node = document.getElementById("nodeDirectSuccess");
        if (node) node.classList.add("active-filter");
      }} else if (stage === "escalated") {{
        const node = document.getElementById("nodeEscalated");
        if (node) node.classList.add("active-filter");
      }} else if (stage.startsWith("model:")) {{
        const m = stage.split(":")[1];
        const node = document.getElementById(`nodeModel_${{m.replace(/[^a-zA-Z0-9]/g, '_')}}`);
        if (node) node.classList.add("active-filter");
      }}

      filterTable();
    }}

    function resetTopoFilter() {{
      filterByTraceStage("all");
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

        let matchesStage = true;
        if (activeTraceFilter === "direct") {{
          matchesStage = !r.escalated && !r.error;
        }} else if (activeTraceFilter === "escalated") {{
          matchesStage = r.escalated === true;
        }} else if (activeTraceFilter.startsWith("model:")) {{
          const targetM = activeTraceFilter.split(":")[1];
          matchesStage = r.chosen_model === targetM || (r.trace_chain && r.trace_chain.some(s => s.name === targetM));
        }}

        return matchesSearch && matchesCat && matchesDiff && matchesModel && matchesStage;
      }});

      renderTable(filtered);
    }}

    // Initial render
    initTopologyPanel();
    renderTable(results);
  </script>
</body>
</html>
"""
        return template
