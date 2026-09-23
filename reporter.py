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
