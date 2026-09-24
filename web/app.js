// State Management
let currentConfig = null;
let currentBenchmarkMode = "simulation";
let discoveredProviderModels = {}; // { [providerId]: string[] }
let currentModalModels = []; // [{ id, created, owned_by }] for provider modal
let quickSyncProviderId = null;
let quickSyncModels = []; // [{ id, created, owned_by }] for quick sync modal
let modelProbeCache = {}; // { [modelName]: { status, latency_ms, reply_snippet, usage, error } }
let activeProbeModelIdx = null;

// Initialize
document.addEventListener("DOMContentLoaded", () => {
  initLeaderboardPagination();
  initLeaderboardColumns();
  refreshStatus();
  loadConfig();
  loadReportsList();
  setInterval(refreshStatus, 15000);

  // Close column picker on outside click
  document.addEventListener("click", () => {
    const dd = document.getElementById("lbColumnsDropdown");
    if (dd) dd.style.display = "none";
  });

  // End-to-end vs Stream checkbox synchronization
  const chkEndToEnd = document.getElementById("chkEndToEndTest");
  const chkStream = document.getElementById("chkStreamTest");
  const lblStream = document.getElementById("lblStreamTest");
  if (chkEndToEnd && chkStream && lblStream) {
    chkEndToEnd.addEventListener("change", () => {
      chkStream.disabled = !chkEndToEnd.checked;
      lblStream.style.opacity = chkEndToEnd.checked ? "1" : "0.5";
      lblStream.style.pointerEvents = chkEndToEnd.checked ? "auto" : "none";
    });
  }
});

// Toast notification helper
function showToast(message, type = "info") {
  const toast = document.getElementById("toast");
  toast.textContent = message;
  toast.className = `toast show ${type}`;
  setTimeout(() => {
    toast.className = "toast";
  }, 3500);
}

// Tab Switching
function switchTab(tabId) {
  const tabs = ["monitor", "models", "policy", "benchmark", "leaderboard"];
  tabs.forEach(t => {
    const pane = document.getElementById(`pane${t.charAt(0).toUpperCase() + t.slice(1)}`);
    const nav = document.getElementById(`nav${t.charAt(0).toUpperCase() + t.slice(1)}`);
    if (pane) pane.classList.toggle("active", t === tabId);
    if (nav) nav.classList.toggle("active", t === tabId);
  });

  const titles = {
    monitor: ["网关状态与实时监控", "监测硬件加速本地 Laya 决策引擎与网关实时运行指标"],
    models: ["提供商与模型管理", "配置 OpenAI 兼容提供商矩阵及各级模型 Token 计费单价与可用性测试"],
    policy: ["Router 策略与 Laya 参数", "微调经济成本模型风险权重、缓存生命周期与 Laya 运行设备"],
    benchmark: ["测试中心与评估报告", "运行典型降本测试集，测算智能分流降本幅度并导出自包含 HTML 报告"],
    leaderboard: ["大模型权威评分查询", "基于 LMSYS Chatbot Arena 权威评测集查询多学科基准，横向对比当前路由候选模型"],
  };
  if (titles[tabId]) {
    document.getElementById("pageTitleText").textContent = titles[tabId][0];
    document.getElementById("pageSubtitleText").textContent = titles[tabId][1];
  }
  if (tabId === "leaderboard") {
    loadLeaderboardData();
  }
}

// Status & Metrics
async function refreshStatus() {
  try {
    const res = await fetch("/api/status");
    if (!res.ok) return;
    const data = await res.json();

    const hwName = data.device_hardware || "本地运算";
    document.getElementById("hardwareName").textContent = hwName;

    const memVal = data.memory_rss_mb;
    const memStr = (typeof memVal === "number" && memVal > 0) ? `${memVal} MB` : "-- MB";
    const uptimeStr = `${Math.round(data.uptime_seconds || 0)}s`;

    let vramRowHtml = "";
    if (data.cuda_available && data.cuda_vram_total_mb) {
      const usedGb = (data.cuda_vram_used_mb / 1024).toFixed(1);
      const totalGb = (data.cuda_vram_total_mb / 1024).toFixed(1);
      vramRowHtml = `
        <div class="meta-row vram-row" style="margin-top: 3px;">
          <span>显存: ${usedGb}G/${totalGb}G</span>
          <span>CUDA</span>
        </div>
      `;
    }

    document.getElementById("sidebarMeta").innerHTML = `
      <div class="meta-row">
        <span>内存: ${memStr}</span>
        <span>运行: ${uptimeStr}</span>
      </div>
      ${vramRowHtml}
    `;

    // Dynamic Title & Badge Adaptation
    document.title = `Auto-LLM-Router 控制台 | Laya on ${hwName}`;
    const sideTag = document.getElementById("sidebarVersionTag");
    if (sideTag) {
      sideTag.textContent = data.accelerator_type ? `Laya ${data.accelerator_type}` : "Laya Local";
    }
    const latTitle = document.getElementById("layaLatencyTitle");
    if (latTitle) {
      latTitle.textContent = `Laya 决策延迟 (${hwName})`;
    }
    const latBadge = document.getElementById("layaLatencyBadge");
    if (latBadge) {
      latBadge.textContent = data.accelerator_type ? `${data.accelerator_type} 加速` : "硬件加速";
    }
    const resDevSub = document.getElementById("resLayaDeviceSub");
    if (resDevSub) {
      resDevSub.textContent = hwName;
    }

    document.getElementById("kpiActivePolicy").textContent = data.active_policy || "F_expected";
    document.getElementById("kpiProvidersCount").innerHTML = `${data.providers_count} <span class="unit">个服务商</span>`;
    document.getElementById("kpiModelsCount").innerHTML = `${data.models_count} <span class="unit">个路由模型</span>`;
  } catch (err) {
    console.error("Failed to fetch status:", err);
  }
}

// Config Loader
async function loadConfig() {
  try {
    const res = await fetch("/api/config");
    if (!res.ok) return;
    currentConfig = await res.json();
    renderProviders();
    renderModels();
    renderPolicyForm();
    // Cache already known upstream models
    if (currentConfig.models) {
      currentConfig.models.forEach(m => {
        if (!discoveredProviderModels[m.provider]) {
          discoveredProviderModels[m.provider] = [];
        }
        if (!discoveredProviderModels[m.provider].includes(m.upstream_id)) {
          discoveredProviderModels[m.provider].push(m.upstream_id);
        }
      });
    }
  } catch (err) {
    console.error("Failed to load config:", err);
  }
}

// Render Providers List
function renderProviders() {
  const container = document.getElementById("providersList");
  if (!container || !currentConfig || !currentConfig.providers) return;
  container.innerHTML = "";

  for (const [id, p] of Object.entries(currentConfig.providers)) {
    const card = document.createElement("div");
    card.className = "provider-card";
    const isLocal = p.base_url.includes("localhost") || p.base_url.includes("127.0.0.1");

    // Count assigned models for this provider
    const assignedCount = (currentConfig.models || []).filter(m => m.provider === id).length;

    card.innerHTML = `
      <div class="prov-head">
        <span class="prov-name">${id}</span>
        <span class="badge ${isLocal ? 'badge-green' : 'badge-blue'}">${isLocal ? '本地 LM Studio' : '云端 API'}</span>
      </div>
      <div class="prov-url">${p.base_url}</div>
      <div style="font-size: 12px; color: var(--text-dim); margin-top: 2px;">
        已配置 <strong>${assignedCount}</strong> 个路由模型
      </div>
      <div class="prov-actions" style="margin-top: 10px;">
        <button class="btn btn-secondary btn-sm" onclick="testProvider('${id}', '${p.base_url}', '${p.api_key || ''}')">⚡ 测试端点</button>
        <button class="btn btn-primary btn-sm" onclick="openQuickSyncModal('${id}')">🔍 获取/选择模型</button>
        <button class="btn btn-secondary btn-sm" style="color:var(--danger);" onclick="deleteProvider('${id}')">删除</button>
      </div>
    `;
    container.appendChild(card);
  }
}

// Test Provider
async function testProvider(id, baseUrl, apiKey) {
  showToast(`正在探测提供商 [${id}] 连通性...`, "info");
  try {
    const res = await fetch("/api/provider/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ base_url: baseUrl, api_key: apiKey }),
    });
    const data = await res.json();
    if (data.status === "ok") {
      const modelsCount = data.available_models ? data.available_models.length : 0;
      showToast(`[${id}] 连通正常！延迟: ${data.latency_ms}ms，发现 ${modelsCount} 个模型`, "success");
      if (data.available_models && data.available_models.length > 0) {
        discoveredProviderModels[id] = data.available_models;
      }
    } else {
      showToast(`[${id}] 连接失败: ${data.error || '无法访问'}`, "danger");
    }
  } catch (e) {
    showToast(`探测异常: ${e.message}`, "danger");
  }
}

// Currency & Exchange Rate Helpers
function getUsdCnyRate() {
  if (currentConfig && currentConfig.policy && currentConfig.policy.currency && currentConfig.policy.currency.usd_cny_rate) {
    return parseFloat(currentConfig.policy.currency.usd_cny_rate) || 7.20;
  }
  const rateInput = document.getElementById("usdCnyRateInput");
  return rateInput ? (parseFloat(rateInput.value) || 7.20) : 7.20;
}

function onUsdRateChanged() {
  updatePricingUsdDisplay();
  renderModels();
}

function formatDualPrice(cnyPrice) {
  const p = parseFloat(cnyPrice) || 0;
  const rate = getUsdCnyRate();
  const usd = (p / rate).toFixed(4);
  return `<span class="dual-price">¥${p.toFixed(3)} <span class="sub-price">($${usd})</span></span>`;
}

function updatePricingUsdDisplay() {
  const rate = getUsdCnyRate();
  const inCny = parseFloat(document.getElementById("modelInputPrice")?.value) || 0;
  const outCny = parseFloat(document.getElementById("modelOutputPrice")?.value) || 0;
  const cacheCny = parseFloat(document.getElementById("modelCachePrice")?.value) || 0;

  const inUsd = (inCny / rate).toFixed(4);
  const outUsd = (outCny / rate).toFixed(4);
  const cacheUsd = (cacheCny / rate).toFixed(4);

  const inTag = document.getElementById("inputPriceUsdTag");
  const outTag = document.getElementById("outputPriceUsdTag");
  const cacheTag = document.getElementById("cachePriceUsdTag");

  if (inTag) inTag.textContent = `折合 $${inUsd} / 1M`;
  if (outTag) outTag.textContent = `折合 $${outUsd} / 1M`;
  if (cacheTag) cacheTag.textContent = `折合 $${cacheUsd} / 1M`;
}

// Render Models Table
function renderModels() {
  const tbody = document.getElementById("modelsTableBody");
  if (!tbody || !currentConfig || !currentConfig.models) return;
  tbody.innerHTML = "";
  populateJudgeModelSelect();

  currentConfig.models.forEach((m, idx) => {
    const tr = document.createElement("tr");
    const p = m.prices || { input: 0, output: 0, cache_read: 0 };
    const probe = modelProbeCache[m.name];

    let probeHtml = "";
    if (!probe) {
      probeHtml = `<button class="btn btn-secondary btn-sm" id="btnProbe_${idx}" onclick="probeModelCapability(${idx})">⚡ 测试可用性</button>`;
    } else if (probe.loading) {
      probeHtml = `<span class="status-pill loading"><span class="spinner" style="width:10px;height:10px;border-width:2px;"></span> 探测中...</span>`;
    } else if (probe.status === "ok") {
      probeHtml = `<span class="status-pill ok" onclick="showProbeDetail(${idx})" title="点击查看生成回复与指标详情">✓ 可用 (${probe.latency_ms}ms)</span>`;
    } else {
      probeHtml = `<span class="status-pill error" onclick="showProbeDetail(${idx})" title="点击查看诊断报错详情">⚠ 异常 (${probe.latency_ms}ms)</span>`;
    }

    tr.innerHTML = `
      <td><strong style="color:var(--primary); font-family:var(--font-mono);">${m.name}</strong></td>
      <td><span class="badge badge-purple">${m.provider}</span></td>
      <td style="font-family:var(--font-mono); font-size:12px;">${m.upstream_id || 'default'}</td>
      <td>${m.free ? '<span style="color:var(--success); font-weight:600;">✓ 免费(0成本)</span>' : '<span style="color:var(--text-dim);">计费</span>'}</td>
      <td>${formatDualPrice(p.input)}</td>
      <td>${formatDualPrice(p.output)}</td>
      <td>
        ${(m.context_tokens || 32768) / 1024}k
        <div style="font-size:11px; margin-top:2px;">
          ${m.timeout_seconds ? `<span class="badge badge-purple" style="font-size:10px; padding:2px 5px;">⏱️ ${m.timeout_seconds}s</span>` : `<span style="color:var(--text-dim); font-size:10px;">⏱️ 继承(${currentConfig?.policy?.request_timeout_seconds || 60}s)</span>`}
        </div>
      </td>
      <td>${probeHtml}</td>
      <td>
        <div class="action-btns-group">
          <button class="btn btn-outline btn-sm" onclick="openEditModelModal(${idx})">✏️ 编辑</button>
          <button class="btn btn-secondary btn-sm" style="color:var(--danger);" onclick="deleteModel(${idx})">删除</button>
        </div>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

// Populate Judge Model Dropdown
function populateJudgeModelSelect(selectedModel) {
  const sel = document.getElementById("verifyJudgeModelSelect");
  if (!sel) return;
  const currentVal = selectedModel || sel.value || (currentConfig?.policy?.verify?.judge_model) || "deepseek-v4-flash";
  sel.innerHTML = "";

  // 1. Laya Local Engine option
  const layaOpt = document.createElement("option");
  layaOpt.value = "laya";
  layaOpt.textContent = "Laya 本地分类裁决引擎 (Local Engine)";
  sel.appendChild(layaOpt);

  // 2. Configured Models
  if (currentConfig && Array.isArray(currentConfig.models)) {
    currentConfig.models.forEach(m => {
      const opt = document.createElement("option");
      opt.value = m.name;
      const cap = m.capability?.general || m.capability?.coding || "--";
      const inPrice = (m.prices && m.prices.input !== undefined) ? m.prices.input : "--";
      opt.textContent = `${m.name} (${m.provider} - 算力:${cap}, $${inPrice}/Mtok)`;
      sel.appendChild(opt);
    });
  }
  sel.value = currentVal;
  if (!sel.value && sel.options.length > 0) {
    sel.selectedIndex = 0;
  }
}

function onVerifyEnabledChanged() {
  const chk = document.getElementById("verifyEnabledSwitch") ? document.getElementById("verifyEnabledSwitch").checked : false;
  updateVerifyBadge(chk);
}

function updateVerifyBadge(enabled) {
  const badge = document.getElementById("verifyStatusBadge");
  const label = document.getElementById("verifyEnabledLabel");
  if (badge) {
    if (enabled) {
      badge.textContent = "已启用";
      badge.style.background = "#ecfdf5";
      badge.style.color = "#059669";
      badge.style.borderColor = "#a7f3d0";
    } else {
      badge.textContent = "未启用";
      badge.style.background = "#f1f5f9";
      badge.style.color = "#64748b";
      badge.style.borderColor = "#cbd5e1";
    }
  }
  if (label) {
    label.textContent = enabled ? "质检验收已开启" : "启用质检验收";
  }
}

// Render Policy Form
function renderPolicyForm() {
  if (!currentConfig || !currentConfig.policy) return;
  const pol = currentConfig.policy;
  const clf = pol.classifier || {};
  const vfy = pol.verify || {};

  const curRate = (pol.currency && pol.currency.usd_cny_rate) ? pol.currency.usd_cny_rate : 7.20;
  const usdRateInput = document.getElementById("usdCnyRateInput");
  if (usdRateInput) usdRateInput.value = curRate;

  document.getElementById("policyNameSelect").value = pol.name || "F_expected";
  document.getElementById("stakesInput").value = pol.stakes_usd || 2.0;
  document.getElementById("detectProbInput").value = pol.detect_prob || 0.6;
  document.getElementById("failureCostInput").value = pol.failure_cost_multiplier || 1.0;
  document.getElementById("remainingTurnsInput").value = pol.remaining_turns || 3.0;
  document.getElementById("requestTimeoutInput").value = pol.request_timeout_seconds || 60;

  document.getElementById("classifierBackendSelect").value = clf.backend || "local";
  document.getElementById("layaDeviceSelect").value = clf.device || "auto";
  document.getElementById("layaModelInput").value = clf.model || "convaiinnovations/laya";
  document.getElementById("layaSubfolderInput").value = clf.subfolder || "multilingual";
  document.getElementById("requestCharsInput").value = clf.request_chars || 6000;

  // Verify and Escalate Controls
  populateJudgeModelSelect(vfy.judge_model);
  const verifySwitch = document.getElementById("verifyEnabledSwitch");
  if (verifySwitch) {
    const isVfyOn = vfy.enabled !== false;
    verifySwitch.checked = isVfyOn;
    updateVerifyBadge(isVfyOn);
  }
  const threshInp = document.getElementById("verifyThresholdInput");
  if (threshInp) {
    threshInp.value = (vfy.thresholds && vfy.thresholds.default !== undefined) ? vfy.thresholds.default : 0.30;
  }
  const maxCapInp = document.getElementById("verifyMaxCapabilityInput");
  if (maxCapInp) {
    maxCapInp.value = vfy.max_capability !== undefined ? vfy.max_capability : 77.5;
  }
  const maxPriceInp = document.getElementById("verifyMaxPriceInput");
  if (maxPriceInp) {
    maxPriceInp.value = vfy.max_price_per_mtok !== undefined ? vfy.max_price_per_mtok : 2.5;
  }
}

// Save Policy Form
async function savePolicyConfig() {
  if (!currentConfig) return;
  currentConfig.policy.name = document.getElementById("policyNameSelect").value;
  currentConfig.policy.stakes_usd = parseFloat(document.getElementById("stakesInput").value);
  currentConfig.policy.detect_prob = parseFloat(document.getElementById("detectProbInput").value);
  currentConfig.policy.failure_cost_multiplier = parseFloat(document.getElementById("failureCostInput").value);
  currentConfig.policy.remaining_turns = parseFloat(document.getElementById("remainingTurnsInput").value);
  currentConfig.policy.request_timeout_seconds = parseFloat(document.getElementById("requestTimeoutInput").value) || 60;

  const usdRate = parseFloat(document.getElementById("usdCnyRateInput")?.value) || 7.20;
  currentConfig.policy.currency = {
    base: "CNY",
    usd_cny_rate: usdRate
  };

  currentConfig.policy.classifier = {
    backend: document.getElementById("classifierBackendSelect").value,
    device: document.getElementById("layaDeviceSelect").value,
    model: document.getElementById("layaModelInput").value,
    subfolder: document.getElementById("layaSubfolderInput").value,
    threads: 4,
    request_chars: parseInt(document.getElementById("requestCharsInput").value) || 6000,
    context_chars: 2000,
  };

  // Verify and Escalate settings
  const vfyEnabled = document.getElementById("verifyEnabledSwitch") ? document.getElementById("verifyEnabledSwitch").checked : true;
  const vfyJudge = document.getElementById("verifyJudgeModelSelect") ? document.getElementById("verifyJudgeModelSelect").value : "deepseek-v4-flash";
  const vfyThresh = parseFloat(document.getElementById("verifyThresholdInput")?.value) || 0.30;
  const vfyMaxCap = parseFloat(document.getElementById("verifyMaxCapabilityInput")?.value) || 77.5;
  const vfyMaxPrice = parseFloat(document.getElementById("verifyMaxPriceInput")?.value) || 2.5;

  currentConfig.policy.verify = {
    ...(currentConfig.policy.verify || {}),
    enabled: vfyEnabled,
    judge_model: vfyJudge,
    max_capability: vfyMaxCap,
    max_price_per_mtok: vfyMaxPrice,
    thresholds: {
      ...((currentConfig.policy.verify && currentConfig.policy.verify.thresholds) || {}),
      default: vfyThresh,
      coding: vfyThresh,
      math: vfyThresh,
    }
  };

  try {
    const res = await fetch("/api/config", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(currentConfig),
    });
    if (res.ok) {
      showToast("策略与质检配置保存成功并已实时生效！", "success");
      refreshStatus();
      renderModels();
    } else {
      showToast("保存配置失败", "danger");
    }
  } catch (err) {
    showToast(`错误: ${err.message}`, "danger");
  }
}

// Quick Prompt Set
function setQuickPrompt(text) {
  document.getElementById("testPromptInput").value = text;
}

let lastDiagnosticData = null;

function escapeHtml(str) {
  if (!str) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

// Single Test Route Run with Diagnostics & Trace Logs
async function runSingleTest() {
  const prompt = document.getElementById("testPromptInput").value.trim();
  if (!prompt) {
    showToast("请输入测试提示词", "warning");
    return;
  }
  const isEndToEnd = document.getElementById("chkEndToEndTest")?.checked || false;
  const isStream = isEndToEnd && (document.getElementById("chkStreamTest")?.checked || false);
  const mode = isEndToEnd ? "end_to_end" : "predict_only";

  const btn = document.getElementById("btnTestRoute");
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner" style="width:14px;height:14px;border-width:2px;"></span> ${isStream ? '正在流式打字生成...' : (isEndToEnd ? '正在端到端生成...' : '正在极速决策...')}`;

  try {
    if (isStream) {
      // --- 🌊 流式打字机交互模式 ---
      document.getElementById("singleTestResult").style.display = "block";
      const dispSection = document.getElementById("dispatchedModelsSection");
      const dispList = document.getElementById("dispatchedModelsList");
      const dispCountBadge = document.getElementById("badgeDispatchedCount");
      const consoleEl = document.getElementById("traceConsole");
      if (consoleEl) consoleEl.innerHTML = "";
      if (dispSection) dispSection.style.display = "block";
      if (dispList) dispList.innerHTML = "";

      const addTraceLine = (item) => {
        if (!consoleEl || !item) return;
        const row = document.createElement("div");
        row.className = "trace-line";
        const tagClass = item.level === "ERROR" ? "error" : (item.level === "WARN" ? "warn" : "info");
        row.innerHTML = `
          <span class="trace-ts">[${item.timestamp}]</span>
          <span class="trace-tag ${tagClass}">${item.level}</span>
          <span class="trace-msg">${escapeHtml(item.message)}</span>
        `;
        consoleEl.appendChild(row);
        consoleEl.scrollTop = consoleEl.scrollHeight;
      };

      const updateTimingAndBars = (layaMs, scoreMs, upMs, verMs, totMs) => {
        const tot = Math.max(totMs || (layaMs + scoreMs + upMs + verMs), 0.1);
        document.getElementById("chipLayaMs").textContent = `${layaMs} ms`;
        document.getElementById("chipScoringMs").textContent = `${scoreMs} ms`;
        const upWrap = document.getElementById("chipUpstreamWrap");
        const verWrap = document.getElementById("chipVerifyWrap");
        if (upMs > 0) {
          upWrap.style.display = "inline-flex";
          document.getElementById("chipUpstreamMs").textContent = `${upMs} ms`;
        } else {
          upWrap.style.display = "none";
        }
        if (verMs > 0) {
          verWrap.style.display = "inline-flex";
          document.getElementById("chipVerifyMs").textContent = `${verMs} ms`;
        } else {
          verWrap.style.display = "none";
        }
        document.getElementById("barLaya").style.width = `${Math.min(100, (layaMs / tot) * 100)}%`;
        document.getElementById("barScoring").style.width = `${Math.min(100, (scoreMs / tot) * 100)}%`;
        document.getElementById("barUpstream").style.width = `${Math.min(100, (upMs / tot) * 100)}%`;
        document.getElementById("barVerify").style.width = `${Math.min(100, (verMs / tot) * 100)}%`;
        document.getElementById("resTotalLatency").textContent = `${Math.round(tot)} ms`;
      };

      const res = await fetch("/api/router/test-single", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: prompt,
          mode: mode,
          max_tokens: 1024,
          stream: true,
        }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        showToast(`测试请求响应失败: ${errData.detail || res.statusText}`, "danger");
        return;
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder("utf-8");
      let buffer = "";
      let curAttempt = 0;
      let reasoningText = "";
      let contentText = "";
      let curModelCard = null;
      let reasoningBlock = null;
      let contentPre = null;
      let cursorSpan = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop(); // 保留不完整的末尾

        for (const rawLine of lines) {
          const line = rawLine.trim();
          if (!line.startsWith("data:")) continue;
          const jsonStr = line.slice(5).trim();
          if (!jsonStr || jsonStr === "[DONE]") continue;

          let ev;
          try {
            ev = JSON.parse(jsonStr);
          } catch (e) {
            continue;
          }

          if (ev.log) addTraceLine(ev.log);
          if (ev.logs) ev.logs.forEach(l => addTraceLine(l));

          if (ev.type === "route_result") {
            document.getElementById("resCategory").textContent = ev.classification?.category || "general";
            document.getElementById("resDifficulty").textContent = ev.classification?.difficulty !== undefined ? ev.classification.difficulty.toFixed(3) : "0.500";
            document.getElementById("resStakes").textContent = ev.classification?.stakes !== undefined ? ev.classification.stakes.toFixed(3) : "0.200";
            document.getElementById("resChosenModel").textContent = `${ev.decision?.chosen_model || '--'} (提供商: ${ev.decision?.provider || '--'} | ${ev.decision?.reason || '期望成本最小化'})`;

            const tBadge = document.getElementById("resTimeoutBadge");
            const effectiveTimeout = ev.decision?.timeout_seconds || (currentConfig?.policy?.request_timeout_seconds || 60);
            if (tBadge) {
              tBadge.textContent = `⏱️ 超时时限: ${effectiveTimeout}s`;
              tBadge.style.display = "inline-flex";
            }
            document.getElementById("resModeBadge").textContent = "模式: 🌊 端到端实时打字生成 (Stream)";

            const layaMs = ev.timing?.classifier_ms || 0;
            const scoreMs = ev.timing?.route_scoring_ms || 0;
            updateTimingAndBars(layaMs, scoreMs, 0, 0, layaMs + scoreMs);
          } else if (ev.type === "start_model") {
            curAttempt = ev.attempt || 1;
            if (dispCountBadge) {
              dispCountBadge.textContent = `${curAttempt} 个模型已调度`;
            }
            reasoningText = "";
            contentText = "";

            curModelCard = document.createElement("div");
            curModelCard.className = "dispatched-model-card status-streaming";
            curModelCard.id = `dispCard_${curAttempt}`;

            const roleTag = ev.is_primary 
              ? `<span class="dispatched-role-tag dispatched-role-primary">👑 路由首选模型</span>`
              : `<span class="dispatched-role-tag dispatched-role-fallback">🔄 自动降级备选 (#${curAttempt})</span>`;

            curModelCard.innerHTML = `
              <div class="dispatched-card-header">
                <div class="dispatched-model-title">
                  ${roleTag}
                  <span class="dispatched-model-name">${escapeHtml(ev.model_name || '--')}</span>
                  <span class="badge badge-subtle" style="font-size:11px;">提供商: ${escapeHtml(ev.provider || '--')}</span>
                  <span style="font-size:11.5px; color:var(--text-muted); margin-left:4px;">⏱️ 时限: ${ev.timeout_seconds || 60}s</span>
                </div>
                <div style="display:flex; align-items:center; gap:8px;" id="dispHeaderActions_${curAttempt}">
                  <span class="badge badge-info" id="dispStatusBadge_${curAttempt}" style="font-size:11.5px; padding:3px 8px;">
                    <span class="spinner" style="width:11px;height:11px;border-width:2px;display:inline-block;vertical-align:middle;margin-right:4px;"></span>正在实时生成...
                  </span>
                </div>
              </div>
              <div class="dispatched-card-body" id="dispBody_${curAttempt}">
                <div class="reasoning-block" id="reasoningBlock_${curAttempt}" style="display:none;">
                  <div class="reasoning-block-title">🧠 思考推导过程 (Reasoning)</div>
                  <div class="reasoning-content" id="reasoningText_${curAttempt}"></div>
                </div>
                <pre class="dispatched-output-content" id="outputContent_${curAttempt}"></pre>
              </div>
            `;
            dispList.appendChild(curModelCard);

            reasoningBlock = document.getElementById(`reasoningBlock_${curAttempt}`);
            contentPre = document.getElementById(`outputContent_${curAttempt}`);
            cursorSpan = document.createElement("span");
            cursorSpan.className = "streaming-cursor";
            contentPre.appendChild(cursorSpan);
          } else if (ev.type === "delta") {
            if (ev.reasoning) {
              reasoningText += ev.reasoning;
              if (reasoningBlock) {
                reasoningBlock.style.display = "block";
                const rTextEl = document.getElementById(`reasoningText_${curAttempt}`);
                if (rTextEl) rTextEl.textContent = reasoningText;
              }
            }
            if (ev.content) {
              contentText += ev.content;
              if (contentPre) {
                contentPre.textContent = contentText;
                contentPre.appendChild(cursorSpan);
                contentPre.scrollTop = contentPre.scrollHeight;
              }
            }
          } else if (ev.type === "attempt_error") {
            const item = ev.item || {};
            const card = document.getElementById(`dispCard_${curAttempt}`) || curModelCard;
            if (card) {
              card.className = "dispatched-model-card status-error";
              const badge = document.getElementById(`dispStatusBadge_${curAttempt}`);
              if (badge) {
                badge.className = "badge badge-danger";
                badge.textContent = `✗ 调度失败 (${item.latency_ms || 0}ms)`;
              }
              const body = document.getElementById(`dispBody_${curAttempt}`);
              if (body) {
                body.innerHTML = `
                  <div class="dispatched-error-content">
                    <strong>上游调用异常:</strong>
                    <span>${escapeHtml(item.output || item.error || '请求失败')}</span>
                    ${ev.next_model ? `<span style="color:#b91c1c; font-size:11.5px; margin-top:2px;">⚠️ 已自动触发降级容灾，转向调度下一个候选模型 [${escapeHtml(ev.next_model)}]</span>` : ''}
                  </div>
                `;
              }
            }
          } else if (ev.type === "model_success") {
            const item = ev.item || {};
            const card = document.getElementById(`dispCard_${curAttempt}`) || curModelCard;
            if (card) {
              card.className = "dispatched-model-card status-success";
              const badge = document.getElementById(`dispStatusBadge_${curAttempt}`);
              if (badge) {
                badge.className = "badge badge-success";
                badge.textContent = `✓ 调度成功 (${item.latency_ms || 0}ms)`;
              }
              const headerActions = document.getElementById(`dispHeaderActions_${curAttempt}`);
              if (headerActions) {
                const copyBtn = document.createElement("button");
                copyBtn.type = "button";
                copyBtn.className = "btn btn-outline btn-sm";
                copyBtn.style.padding = "2px 8px";
                copyBtn.style.fontSize = "11.5px";
                copyBtn.innerHTML = "📋 复制回答";
                copyBtn.onclick = () => copyModelOutput(curAttempt, copyBtn);
                headerActions.appendChild(copyBtn);
              }
              if (cursorSpan && cursorSpan.parentNode) {
                cursorSpan.parentNode.removeChild(cursorSpan);
              }
              if (contentPre) {
                contentPre.textContent = contentText || "（生成完成，上游返回空白文本）";
              }
            }
          } else if (ev.type === "verify_start") {
            const verWrap = document.getElementById("chipVerifyWrap");
            if (verWrap) {
              verWrap.style.display = "inline-flex";
              document.getElementById("chipVerifyMs").innerHTML = `<span style="animation: pulse 1.5s infinite;">⏳ 评判中...</span>`;
            }
          } else if (ev.type === "verify_done") {
            const verWrap = document.getElementById("chipVerifyWrap");
            const verMs = ev.verification_ms || 0;
            if (verWrap) {
              verWrap.style.display = "inline-flex";
              const vLabel = ev.verdict?.escalate ? "⚠️ 触发升级" : "✓ 合格";
              document.getElementById("chipVerifyMs").textContent = `${verMs} ms (${vLabel})`;
            }
            // 在首选卡片右上角增加质检结果徽章
            const pCard = document.getElementById("dispCard_1");
            if (pCard) {
              const hActions = document.getElementById("dispHeaderActions_1");
              if (hActions && !document.getElementById("dispVerifyBadge_1")) {
                const vBadge = document.createElement("span");
                vBadge.id = "dispVerifyBadge_1";
                vBadge.className = "badge";
                if (ev.verdict?.escalate) {
                  vBadge.style.cssText = "background:#fef3c7; color:#b45309; border:1px solid #fde68a; font-size:11px; padding:2px 8px;";
                  vBadge.title = ev.verdict?.reason || "质检验收未通过";
                  vBadge.textContent = `⚠️ 质检 ${Number(ev.verdict?.p_adequate || 0).toFixed(2)} (未通过)`;
                } else {
                  vBadge.style.cssText = "background:#ecfdf5; color:#059669; border:1px solid #a7f3d0; font-size:11px; padding:2px 8px;";
                  vBadge.title = "质检合格直接交付";
                  vBadge.textContent = `✓ 质检合格 (${Number(ev.verdict?.p_adequate || 0).toFixed(2)})`;
                }
                hActions.prepend(vBadge);
              }
            }
          } else if (ev.type === "escalate_start") {
            const pCard = document.getElementById("dispCard_1");
            if (pCard && !document.getElementById("dispEscalateAlert_1")) {
              const alertBox = document.createElement("div");
              alertBox.id = "dispEscalateAlert_1";
              alertBox.style.cssText = "background:#fffbeb; border:1px solid #fde68a; color:#92400e; padding:8px 12px; border-radius:6px; font-size:12px; margin-top:8px;";
              alertBox.innerHTML = `⚠️ <strong>质检未达标触发两跳升级</strong>：评分偏低 (${escapeHtml(ev.reason || '')})，已自动调度旗舰高阶模型 <strong>${escapeHtml(ev.target_model || '')}</strong> 重新生成高确定性解答。`;
              const b = document.getElementById("dispBody_1");
              if (b) b.prepend(alertBox);
            }
          } else if (ev.type === "done") {
            lastDiagnosticData = ev;
            const timing = ev.timing || {};
            const layaMs = timing.classifier_ms || 0;
            const scoreMs = timing.route_scoring_ms || 0;
            const upMs = timing.upstream_request_ms || 0;
            const verMs = timing.verification_ms || 0;
            const totMs = timing.total_latency_ms || 0;
            updateTimingAndBars(layaMs, scoreMs, upMs, verMs, totMs);

            if (cursorSpan && cursorSpan.parentNode) {
              cursorSpan.parentNode.removeChild(cursorSpan);
            }
            showToast(`流式生成完毕 (全程总耗时: ${totMs}ms)`, "success");
          }
        }
      }
      return;
    }

    // --- ⚡ 纯路由预测或非流式模式 ---
    const res = await fetch("/api/router/test-single", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: prompt,
        mode: mode,
        max_tokens: 1024,
      }),
    });

    if (res.ok) {
      const data = await res.json();
      lastDiagnosticData = data;

      document.getElementById("singleTestResult").style.display = "block";

      // 1. Core Metrics & Timeout Badge
      document.getElementById("resCategory").textContent = data.classification?.category || "general";
      document.getElementById("resDifficulty").textContent = data.classification?.difficulty !== undefined ? data.classification.difficulty.toFixed(3) : "0.500";
      document.getElementById("resStakes").textContent = data.classification?.stakes !== undefined ? data.classification.stakes.toFixed(3) : "0.200";
      document.getElementById("resTotalLatency").textContent = `${data.timing?.total_latency_ms || 0} ms`;
      document.getElementById("resChosenModel").textContent = `${data.decision?.chosen_model || '--'} (提供商: ${data.decision?.provider || '--'} | ${data.decision?.reason || '期望成本最小化'})`;

      const tBadge = document.getElementById("resTimeoutBadge");
      const effectiveTimeout = data.decision?.timeout_seconds || data.upstream_response?.timeout_seconds || (currentConfig?.policy?.request_timeout_seconds || 60);
      if (tBadge) {
        tBadge.textContent = `⏱️ 超时时限: ${effectiveTimeout}s`;
        tBadge.style.display = "inline-flex";
      }

      // 2. Timing Breakdown
      const timing = data.timing || {};
      const tot = Math.max(timing.total_latency_ms || 1, 0.1);
      const layaMs = timing.classifier_ms || 0;
      const scoreMs = timing.route_scoring_ms || 0;
      const upMs = timing.upstream_request_ms || 0;
      const verMs = timing.verification_ms || 0;

      document.getElementById("resModeBadge").textContent = isEndToEnd ? "模式: 🌐 端到端全链路" : "模式: ⚡ 纯路由预测 (亚秒级)";
      document.getElementById("chipLayaMs").textContent = `${layaMs} ms`;
      document.getElementById("chipScoringMs").textContent = `${scoreMs} ms`;

      const upWrap = document.getElementById("chipUpstreamWrap");
      const verWrap = document.getElementById("chipVerifyWrap");

      if (isEndToEnd) {
        upWrap.style.display = "inline-flex";
        document.getElementById("chipUpstreamMs").textContent = `${upMs} ms`;
        if (verMs > 0) {
          verWrap.style.display = "inline-flex";
          document.getElementById("chipVerifyMs").textContent = `${verMs} ms`;
        } else {
          verWrap.style.display = "none";
        }
      } else {
        upWrap.style.display = "none";
        verWrap.style.display = "none";
      }

      // Progress bar widths
      document.getElementById("barLaya").style.width = `${Math.min(100, (layaMs / tot) * 100)}%`;
      document.getElementById("barScoring").style.width = `${Math.min(100, (scoreMs / tot) * 100)}%`;
      document.getElementById("barUpstream").style.width = `${Math.min(100, (upMs / tot) * 100)}%`;
      document.getElementById("barVerify").style.width = `${Math.min(100, (verMs / tot) * 100)}%`;

      // 3. Render Dispatched Models & Generation Outputs
      const dispSection = document.getElementById("dispatchedModelsSection");
      const dispList = document.getElementById("dispatchedModelsList");
      const dispCountBadge = document.getElementById("badgeDispatchedCount");

      const dispatched = data.dispatched_models || (data.upstream_response ? [data.upstream_response] : []);
      if (isEndToEnd && dispatched.length > 0 && dispSection && dispList) {
        dispSection.style.display = "block";
        if (dispCountBadge) {
          dispCountBadge.textContent = `${dispatched.length} 个模型已调度`;
        }
        dispList.innerHTML = "";

        dispatched.forEach((item, idx) => {
          const card = document.createElement("div");
          const isSuccess = item.status === "success";
          card.className = `dispatched-model-card ${isSuccess ? 'status-success' : 'status-error'}`;

          const isEscalation = item.is_escalation || false;
          let roleTag = "";
          if (isEscalation) {
            roleTag = `<span class="dispatched-role-tag" style="background:#f5f3ff; color:#7c3aed; border:1px solid #ddd6fe;">🎯 质检二次升级交付</span>`;
          } else if (item.is_primary) {
            roleTag = `<span class="dispatched-role-tag dispatched-role-primary">👑 路由首选模型</span>`;
          } else {
            roleTag = `<span class="dispatched-role-tag dispatched-role-fallback">🔄 自动降级备选 (#${item.attempt || idx + 1})</span>`;
          }

          let vfyBadge = "";
          if (item.verification && item.verification.verified) {
            const v = item.verification;
            if (v.escalate) {
              vfyBadge = `<span class="badge" style="background:#fef3c7; color:#b45309; border:1px solid #fde68a; font-size:11px; padding:2px 8px;" title="${escapeHtml(v.reason || '未达标')}">⚠️ 质检 ${Number(v.score || 0).toFixed(2)} (未通过)</span>`;
            } else {
              vfyBadge = `<span class="badge" style="background:#ecfdf5; color:#059669; border:1px solid #a7f3d0; font-size:11px; padding:2px 8px;" title="满意度达标">✓ 质检合格 (${Number(v.score || 0).toFixed(2)})</span>`;
            }
          }

          const statusBadge = isSuccess
            ? `<span class="badge badge-success" style="font-size:11.5px; padding:3px 8px;">✓ 调度成功 (${item.latency_ms}ms)</span>`
            : `<span class="badge badge-danger" style="font-size:11.5px; padding:3px 8px;">✗ 调度失败 (${item.latency_ms}ms)</span>`;

          const tokensInfo = item.usage ? `Prompt: ${item.usage.prompt_tokens || 0} / Comp: ${item.usage.completion_tokens || 0}` : '';

          card.innerHTML = `
            <div class="dispatched-card-header">
              <div class="dispatched-model-title">
                ${roleTag}
                <span class="dispatched-model-name">${escapeHtml(item.model_name || '--')}</span>
                <span class="badge badge-subtle" style="font-size:11px;">提供商: ${escapeHtml(item.provider || '--')}</span>
                ${tokensInfo ? `<span style="font-size:11.5px; color:var(--text-muted); margin-left:4px;">Tokens [${tokensInfo}]</span>` : ''}
              </div>
              <div style="display:flex; align-items:center; gap:8px;">
                ${vfyBadge}
                ${statusBadge}
                ${isSuccess ? `<button type="button" class="btn btn-outline btn-sm" onclick="copyModelOutput(${idx}, this)" style="padding: 2px 8px; font-size:11.5px;">📋 复制回答</button>` : ''}
              </div>
            </div>
            <div class="dispatched-card-body">
              ${isSuccess 
                ? `<pre class="dispatched-output-content" id="outputContent_${idx}">${escapeHtml(item.output || item.full_reply || '（生成完成，无文本返回）')}</pre>`
                : `<div class="dispatched-error-content">
                    <strong>上游调用异常:</strong>
                    <span>${escapeHtml(item.output || item.error || '请求失败')}</span>
                    <span style="color:#b91c1c; font-size:11.5px; margin-top:2px;">⚠️ 系统已自动记录并尝试降级调度下一个可用模型</span>
                   </div>`
              }
            </div>
          `;
          dispList.appendChild(card);
        });
      } else if (dispSection) {
        dispSection.style.display = "none";
      }

      // 4. Execution Trace Console Logs
      const consoleEl = document.getElementById("traceConsole");
      consoleEl.innerHTML = "";
      const logs = data.logs || [];
      if (logs.length === 0) {
        consoleEl.innerHTML = `<div style="color: #94a3b8;">暂无流水日志</div>`;
      } else {
        logs.forEach(item => {
          const row = document.createElement("div");
          row.className = "trace-line";
          const tagClass = item.level === "ERROR" ? "error" : (item.level === "WARN" ? "warn" : "info");
          row.innerHTML = `
            <span class="trace-ts">[${item.timestamp}]</span>
            <span class="trace-tag ${tagClass}">${item.level}</span>
            <span class="trace-msg">${escapeHtml(item.message)}</span>
          `;
          consoleEl.appendChild(row);
        });
        consoleEl.scrollTop = consoleEl.scrollHeight;
      }

      showToast(`路由决策完成 (总耗时: ${tot}ms)`, "success");
    } else {
      const errData = await res.json().catch(() => ({}));
      showToast(`测试请求响应失败: ${errData.detail || res.statusText}`, "danger");
    }
  } catch (err) {
    showToast(`测试出错: ${err.message}`, "danger");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<svg width="16" height="16" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M13 10V3L4 14h7v7l9-11h-7z"></path></svg><span>立即运行路由与模型生成</span>`;
  }
}

// Copy Model Output Text
function copyModelOutput(idx, btnEl) {
  const el = document.getElementById(`outputContent_${idx}`);
  if (!el) return;
  const text = el.textContent;
  navigator.clipboard.writeText(text).then(() => {
    const originalText = btnEl.innerHTML;
    btnEl.innerHTML = "✓ 已复制!";
    btnEl.classList.add("btn-primary");
    btnEl.classList.remove("btn-outline");
    setTimeout(() => {
      btnEl.innerHTML = originalText;
      btnEl.classList.remove("btn-primary");
      btnEl.classList.add("btn-outline");
    }, 2000);
  }).catch(err => {
    showToast("复制失败: " + err, "danger");
  });
}

// Copy Diagnostic JSON
function copyDiagnosticJson(event) {
  if (event) {
    event.stopPropagation();
    event.preventDefault();
  }
  if (!lastDiagnosticData) {
    showToast("当前暂无排障诊断数据", "warning");
    return;
  }
  const jsonStr = JSON.stringify(lastDiagnosticData, null, 2);
  navigator.clipboard.writeText(jsonStr).then(() => {
    showToast("排障诊断 JSON 已成功复制到剪贴板！", "success");
  }).catch(err => {
    showToast(`复制失败: ${err.message}`, "danger");
  });
}

// Benchmark Mode Selection
function setBenchmarkMode(mode) {
  currentBenchmarkMode = mode;
  document.getElementById("radioSimLabel").classList.toggle("active", mode === "simulation");
  document.getElementById("radioRealLabel").classList.toggle("active", mode === "real");
}

// Start Benchmark Test
async function startBenchmark() {
  const btn = document.getElementById("btnStartBenchmark");
  const spinner = document.getElementById("benchRunningSpinner");
  const progressText = document.getElementById("benchProgressText");

  btn.style.display = "none";
  spinner.style.display = "flex";
  progressText.textContent = currentBenchmarkMode === "real" 
    ? "正在向实际端点 (LM Studio / API) 逐项发送测试请求..." 
    : "正在通过 Laya 模型在本地加速硬件上执行极速模拟测试...";

  try {
    const res = await fetch("/api/benchmark/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ mode: currentBenchmarkMode }),
    });

    if (res.ok) {
      const data = await res.json();
      const s = data.summary;

      const sym = s.currency_symbol || "¥";
      document.getElementById("latestResultCard").style.display = "block";
      document.getElementById("latestSummaryMeta").textContent = `完成时间: ${s.timestamp} | 模式: ${data.mode} | 测试集: ${s.total_cases} 条用例`;
      document.getElementById("resSavingsPct").textContent = `${s.total_savings_pct}%`;
      document.getElementById("resSavingsUsd").textContent = `节省金额: ${sym}${s.total_savings_usd.toFixed(4)}`;
      document.getElementById("resRouterCost").textContent = `${sym}${s.cost_router_total.toFixed(4)}`;
      document.getElementById("resExpensiveCost").textContent = `全量昂贵对比: ${sym}${s.cost_expensive_total.toFixed(4)}`;
      document.getElementById("resLayaAvgLat").textContent = `${s.avg_classifier_latency_ms} ms`;
      document.getElementById("resAccuracy").textContent = `${s.alignment_rate}%`;
      document.getElementById("resTotalCases").textContent = `测试用例: ${s.total_cases} 条`;

      const btnView = document.getElementById("btnViewReport");
      btnView.href = data.report_url;
      const btnDl = document.getElementById("btnDownloadReport");
      btnDl.href = `${data.report_url}?download=1`;

      showToast("综合评估测试完成！HTML 报告已生成。", "success");
      loadReportsList();
    } else {
      let errMsg = `HTTP ${res.status}`;
      try {
        const err = await res.json();
        errMsg = err.detail || err.message || errMsg;
      } catch (_) {
        const text = await res.text();
        if (text) errMsg = text.slice(0, 200);
      }
      showToast(`测试失败: ${errMsg}`, "danger");
    }
  } catch (err) {
    showToast(`测试执行异常: ${err.message}`, "danger");
  } finally {
    btn.style.display = "inline-flex";
    spinner.style.display = "none";
  }
}

// Load Historical Reports List with Checkbox & Deletion Actions
async function loadReportsList() {
  const tbody = document.getElementById("reportsTableBody");
  if (!tbody) return;

  const selectAllChk = document.getElementById("selectAllReportsChk");
  if (selectAllChk) selectAllChk.checked = false;
  updateBatchDeleteReportsBtn();

  tbody.innerHTML = `<tr><td colspan="5" style="color:var(--text-muted);text-align:center;">加载中...</td></tr>`;

  try {
    const res = await fetch("/api/reports");
    if (!res.ok) return;
    const reports = await res.json();

    if (reports.length === 0) {
      tbody.innerHTML = `<tr><td colspan="5" style="color:var(--text-muted);text-align:center;">暂无历史报告，点击上方“一键开始运行测试”即可生成。</td></tr>`;
      return;
    }

    tbody.innerHTML = "";
    reports.forEach((r, idx) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td style="text-align: center;">
          <input type="checkbox" class="report-row-chk" value="${r.filename}" id="rep_chk_${idx}" onchange="onReportCheckboxChanged()" style="cursor: pointer;">
        </td>
        <td><strong style="color:var(--primary); font-family:var(--font-mono);">${r.filename}</strong></td>
        <td>${r.created_at}</td>
        <td>${r.size_kb} KB</td>
        <td>
          <a href="${r.url}" target="_blank" class="btn btn-secondary btn-sm" style="margin-right:6px;">👁️ 在线预览</a>
          <a href="${r.url}?download=1" class="btn btn-primary btn-sm" style="margin-right:6px;">⬇️ 下载</a>
          <button class="btn btn-secondary btn-sm" style="color:var(--danger);" onclick="deleteSingleReport('${r.filename}')">🗑️ 删除</button>
        </td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    console.error("Failed to load reports:", err);
    tbody.innerHTML = `<tr><td colspan="5" style="color:var(--danger);text-align:center;">加载报告列表失败: ${err.message}</td></tr>`;
  }
}

function onReportCheckboxChanged() {
  const allRows = document.querySelectorAll(".report-row-chk");
  const checkedRows = document.querySelectorAll(".report-row-chk:checked");
  const selectAllChk = document.getElementById("selectAllReportsChk");

  if (selectAllChk) {
    selectAllChk.checked = allRows.length > 0 && allRows.length === checkedRows.length;
  }
  updateBatchDeleteReportsBtn();
}

function toggleSelectAllReports(checked) {
  const allRows = document.querySelectorAll(".report-row-chk");
  allRows.forEach(chk => {
    chk.checked = checked;
  });
  updateBatchDeleteReportsBtn();
}

function updateBatchDeleteReportsBtn() {
  const checkedRows = document.querySelectorAll(".report-row-chk:checked");
  const btn = document.getElementById("btnBatchDeleteReports");
  const text = document.getElementById("batchDeleteReportsText");

  if (!btn || !text) return;

  if (checkedRows.length > 0) {
    btn.style.display = "inline-flex";
    text.textContent = `批量删除 (已选 ${checkedRows.length} 项)`;
  } else {
    btn.style.display = "none";
  }
}

async function deleteSingleReport(filename) {
  if (!confirm(`确定要彻底删除评估报告 "${filename}" 吗？此操作无法撤销。`)) {
    return;
  }

  try {
    const res = await fetch(`/api/reports/${encodeURIComponent(filename)}`, {
      method: "DELETE",
    });
    const data = await res.json();
    if (res.ok) {
      showToast(`报告 [${filename}] 已成功删除！`, "success");
      loadReportsList();
    } else {
      showToast(`删除失败: ${data.detail || data.error || '未知错误'}`, "danger");
    }
  } catch (err) {
    showToast(`删除出错: ${err.message}`, "danger");
  }
}

async function batchDeleteSelectedReports() {
  const checkedBoxes = document.querySelectorAll(".report-row-chk:checked");
  const filenames = Array.from(checkedBoxes).map(c => c.value);

  if (filenames.length === 0) {
    showToast("请先勾选需要删除的评估报告", "warning");
    return;
  }

  if (!confirm(`确定要批量彻底删除选中的 ${filenames.length} 个评估报告吗？此操作无法撤销。`)) {
    return;
  }

  try {
    const res = await fetch("/api/reports/batch-delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filenames }),
    });
    const data = await res.json();
    if (res.ok) {
      showToast(data.message || `成功删除 ${data.deleted_count} 个报告！`, "success");
      loadReportsList();
    } else {
      showToast(`批量删除失败: ${data.detail || '未知错误'}`, "danger");
    }
  } catch (err) {
    showToast(`批量删除出错: ${err.message}`, "danger");
  }
}

// ---------------------------------------------------------------------------
// Provider Modal with Auto Model Fetching & Multi-selection
// ---------------------------------------------------------------------------
function openAddProviderModal() {
  document.getElementById("providerModalTitle").textContent = "添加模型提供商 (Provider)";
  document.getElementById("provIdInput").value = "";
  document.getElementById("provUrlInput").value = "http://localhost:1234/v1";
  document.getElementById("provKeyInput").value = "";
  document.getElementById("provModelsContainer").style.display = "none";
  currentModalModels = [];
  document.getElementById("providerModal").style.display = "flex";
}

function closeModal(id) {
  document.getElementById(id).style.display = "none";
}

async function fetchModelsForCurrentModal() {
  const url = document.getElementById("provUrlInput").value.trim();
  const key = document.getElementById("provKeyInput").value.trim();
  if (!url) {
    showToast("请先填写 Base URL 端点地址", "warning");
    return;
  }

  const btn = document.getElementById("btnFetchProvModels");
  btn.disabled = true;
  btn.innerHTML = `<span class="spinner" style="width:12px;height:12px;border-width:2px;"></span> 正在遍历端点模型...`;

  try {
    const res = await fetch("/api/provider/models", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ base_url: url, api_key: key }),
    });
    let data;
    try {
      data = await res.json();
    } catch (_) {
      const errText = await res.text().catch(() => "");
      throw new Error(`服务返回非 JSON 响应 (HTTP ${res.status}): ${errText || '未知错误'}`);
    }

    if (data.status === "ok") {
      currentModalModels = data.models || [];
      renderModalModelsCheckList("provModelsCheckList", currentModalModels, true);
      document.getElementById("provModelsContainer").style.display = "block";
      updateProvSelectedCount();

      if (currentModalModels.length === 0) {
        showToast("端点已连接成功，但上游未检测到加载的模型（如本地 LM Studio，请在 LM Studio 中加载模型）", "info");
      } else {
        showToast(`成功探测并获取到 ${currentModalModels.length} 个可用模型！`, "success");
      }
    } else {
      showToast(`获取模型失败: ${data.error || '连接异常'}`, "danger");
    }
  } catch (e) {
    showToast(`请求异常: ${e.message}`, "danger");
  } finally {
    btn.disabled = false;
    btn.innerHTML = `<svg width="14" height="14" fill="none" stroke="currentColor" stroke-width="2" viewBox="0 0 24 24"><path d="M4 4v5h.582m15.356 2A8.001 8.001 0 004.582 9m0 0H9m11 11v-5h-.581m0 0a8.003 8.003 0 01-15.357-2m15.357 2H15"></path></svg><span>获取端点模型列表</span>`;
  }
}

function renderModalModelsCheckList(containerId, models, defaultChecked = true) {
  const list = document.getElementById(containerId);
  if (!list) return;
  list.innerHTML = "";

  if (models.length === 0) {
    list.innerHTML = `<div style="text-align:center; padding: 18px; color:var(--text-muted); font-size:13px;">暂未检索到可用模型。若使用本地 LM Studio，请确认已在开发者页面中启动模型。</div>`;
    return;
  }

  models.forEach((m, idx) => {
    const item = document.createElement("div");
    item.className = `model-check-item ${defaultChecked ? 'selected' : ''}`;
    item.dataset.modelId = m.id;
    item.innerHTML = `
      <div class="model-check-left">
        <input type="checkbox" id="${containerId}_chk_${idx}" value="${m.id}" ${defaultChecked ? 'checked' : ''} onchange="onModelItemCheckboxChanged(this)">
        <label for="${containerId}_chk_${idx}" style="cursor:pointer;">${m.id}</label>
      </div>
      <span class="badge badge-blue" style="font-size:11px;">${m.owned_by || 'available'}</span>
    `;
    list.appendChild(item);
  });
}

function onModelItemCheckboxChanged(chk) {
  const item = chk.closest(".model-check-item");
  if (item) item.classList.toggle("selected", chk.checked);
  updateProvSelectedCount();
  updateQuickSyncSelectedCount();
}

function toggleSelectAllProvModels(checked) {
  const chks = document.querySelectorAll("#provModelsCheckList input[type='checkbox']");
  chks.forEach(c => {
    c.checked = checked;
    c.closest(".model-check-item")?.classList.toggle("selected", checked);
  });
  updateProvSelectedCount();
}

function updateProvSelectedCount() {
  const chks = document.querySelectorAll("#provModelsCheckList input[type='checkbox']:checked");
  const el = document.getElementById("provSelectedCount");
  if (el) el.textContent = chks.length;
}

function filterProvModelsList() {
  const q = document.getElementById("provModelSearchInput").value.toLowerCase();
  const items = document.querySelectorAll("#provModelsCheckList .model-check-item");
  items.forEach(it => {
    const id = (it.dataset.modelId || "").toLowerCase();
    it.style.display = id.includes(q) ? "flex" : "none";
  });
}

async function testModalProvider() {
  const url = document.getElementById("provUrlInput").value.trim();
  const key = document.getElementById("provKeyInput").value.trim();
  if (!url) {
    showToast("请先填写 Base URL", "warning");
    return;
  }
  showToast("正在测试端点连通性...", "info");
  try {
    const res = await fetch("/api/provider/test", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ base_url: url, api_key: key }),
    });
    const data = await res.json();
    if (data.status === "ok") {
      showToast(`连接成功！延迟: ${data.latency_ms}ms`, "success");
    } else {
      showToast(`连接失败: ${data.error || '错误'}`, "danger");
    }
  } catch (e) {
    showToast(`连接测试出错: ${e.message}`, "danger");
  }
}

function saveProviderFromModal() {
  const id = document.getElementById("provIdInput").value.trim();
  const url = document.getElementById("provUrlInput").value.trim();
  const key = document.getElementById("provKeyInput").value.trim();

  if (!id || !url) {
    showToast("请填写提供商 ID 和 URL", "warning");
    return;
  }

  if (!currentConfig.providers) currentConfig.providers = {};
  currentConfig.providers[id] = {
    name: id,
    base_url: url,
    api_key: key,
    cache: "openai",
    api: "openai",
  };

  // Collect checked models
  const checkedBoxes = document.querySelectorAll("#provModelsCheckList input[type='checkbox']:checked");
  const selectedModelIds = Array.from(checkedBoxes).map(c => c.value);
  const isLocal = url.includes("localhost") || url.includes("127.0.0.1");

  // Save to discovered cache
  discoveredProviderModels[id] = selectedModelIds;

  // Batch import checked models to router catalog
  if (!currentConfig.models) currentConfig.models = [];
  let importedCount = 0;

  selectedModelIds.forEach(mId => {
    const exists = currentConfig.models.some(m => m.provider === id && m.upstream_id === mId);
    if (!exists) {
      const cleanName = mId.split("/").pop().replace(/[:.]/g, "-");
      currentConfig.models.push({
        name: cleanName,
        provider: id,
        upstream_id: mId,
        free: isLocal,
        prices: {
          input: isLocal ? 0.0 : (mId.includes("deepseek") ? 0.14 : (mId.includes("gpt-4") ? 2.5 : 0.5)),
          output: isLocal ? 0.0 : (mId.includes("deepseek") ? 0.28 : (mId.includes("gpt-4") ? 10.0 : 1.5)),
          cache_read: isLocal ? 0.0 : 0.05,
          cache_write: isLocal ? 0.0 : 0.1,
        },
        capability: {
          coding: isLocal ? 75 : 90,
          math: isLocal ? 70 : 88,
          general: isLocal ? 80 : 92,
          knowledge: isLocal ? 75 : 90,
          summarisation: isLocal ? 80 : 90,
          agentic: isLocal ? 70 : 85,
          tool_use: isLocal ? 72 : 88,
        },
        context_tokens: isLocal ? 32768 : 128000,
        tools: true,
        vision: false,
      });
      importedCount++;
    }
  });

  savePolicyConfig();
  renderProviders();
  renderModels();
  closeModal("providerModal");
  showToast(`提供商 [${id}] 保存成功！已同步导入 ${importedCount} 个模型到路由目录。`, "success");
}

// ---------------------------------------------------------------------------
// Quick Sync Modal for Existing Provider Cards
// ---------------------------------------------------------------------------
async function openQuickSyncModal(providerId) {
  quickSyncProviderId = providerId;
  const p = currentConfig.providers[providerId];
  if (!p) return;

  document.getElementById("providerModelsModalTitle").textContent = `选择并同步提供商模型：[${providerId}]`;
  document.getElementById("quickSyncProvMeta").textContent = `正在连接 ${p.base_url} 遍历可用模型...`;
  document.getElementById("quickSyncCheckList").innerHTML = `<div style="text-align:center; padding:20px;"><span class="spinner"></span> 正在拉取模型...</div>`;
  document.getElementById("providerModelsModal").style.display = "flex";

  await refreshQuickSyncModels();
}

async function refreshQuickSyncModels() {
  if (!quickSyncProviderId) return;
  const p = currentConfig.providers[quickSyncProviderId];
  if (!p) return;

  try {
    const res = await fetch("/api/provider/models", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ base_url: p.base_url, api_key: p.api_key }),
    });
    let data;
    try {
      data = await res.json();
    } catch (_) {
      const errText = await res.text().catch(() => "");
      throw new Error(`服务返回非 JSON 响应 (HTTP ${res.status}): ${errText || '未知错误'}`);
    }
    if (data.status === "ok") {
      quickSyncModels = data.models || [];
      const displayUrl = data.resolved_base_url && data.resolved_base_url !== p.base_url 
        ? `${p.base_url} (${data.resolved_base_url})` 
        : p.base_url;
      document.getElementById("quickSyncProvMeta").textContent = `端点: ${displayUrl} (共检索到 ${quickSyncModels.length} 个模型)`;

      // Identify already configured models
      const configuredUpstream = (currentConfig.models || [])
        .filter(m => m.provider === quickSyncProviderId)
        .map(m => m.upstream_id);

      const list = document.getElementById("quickSyncCheckList");
      list.innerHTML = "";

      if (quickSyncModels.length === 0) {
        list.innerHTML = `<div style="text-align:center; padding: 24px; color:var(--text-muted); font-size:13.5px;">端点响应成功，但当前未加载任何模型。若是 LM Studio，请在客户端加载本地模型后点击“重新拉取”。</div>`;
        updateQuickSyncSelectedCount();
        return;
      }

      quickSyncModels.forEach((m, idx) => {
        const isConfigured = configuredUpstream.includes(m.id);
        const item = document.createElement("div");
        item.className = `model-check-item ${isConfigured ? 'selected' : ''}`;
        item.dataset.modelId = m.id;
        item.innerHTML = `
          <div class="model-check-left">
            <input type="checkbox" id="qs_chk_${idx}" value="${m.id}" ${isConfigured ? 'checked' : ''} onchange="onModelItemCheckboxChanged(this)">
            <label for="qs_chk_${idx}" style="cursor:pointer;">${m.id}</label>
          </div>
          <div>
            ${isConfigured ? '<span class="badge badge-green" style="font-size:11px; margin-right:6px;">已在路由目录</span>' : ''}
            <span class="badge badge-blue" style="font-size:11px;">${m.owned_by || 'model'}</span>
          </div>
        `;
        list.appendChild(item);
      });

      updateQuickSyncSelectedCount();
      // Cache discovered
      discoveredProviderModels[quickSyncProviderId] = quickSyncModels.map(m => m.id);
    } else {
      document.getElementById("quickSyncCheckList").innerHTML = `<div style="color:var(--danger); padding:16px;">获取失败: ${data.error}</div>`;
    }
  } catch (err) {
    document.getElementById("quickSyncCheckList").innerHTML = `<div style="color:var(--danger); padding:16px;">请求异常: ${err.message}</div>`;
  }
}

function toggleSelectAllQuickSync(checked) {
  const chks = document.querySelectorAll("#quickSyncCheckList input[type='checkbox']");
  chks.forEach(c => {
    c.checked = checked;
    c.closest(".model-check-item")?.classList.toggle("selected", checked);
  });
  updateQuickSyncSelectedCount();
}

function updateQuickSyncSelectedCount() {
  const chks = document.querySelectorAll("#quickSyncCheckList input[type='checkbox']:checked");
  const el = document.getElementById("quickSyncSelectedCount");
  if (el) el.textContent = chks.length;
}

function filterQuickSyncList() {
  const q = document.getElementById("quickSyncSearchInput").value.toLowerCase();
  const items = document.querySelectorAll("#quickSyncCheckList .model-check-item");
  items.forEach(it => {
    const id = (it.dataset.modelId || "").toLowerCase();
    it.style.display = id.includes(q) ? "flex" : "none";
  });
}

function saveQuickSyncSelection() {
  if (!quickSyncProviderId) return;
  const p = currentConfig.providers[quickSyncProviderId];
  const isLocal = p.base_url.includes("localhost") || p.base_url.includes("127.0.0.1");

  const checkedBoxes = document.querySelectorAll("#quickSyncCheckList input[type='checkbox']:checked");
  const selectedIds = Array.from(checkedBoxes).map(c => c.value);

  if (!currentConfig.models) currentConfig.models = [];
  let addedCount = 0;

  selectedIds.forEach(mId => {
    const exists = currentConfig.models.some(m => m.provider === quickSyncProviderId && m.upstream_id === mId);
    if (!exists) {
      const cleanName = mId.split("/").pop().replace(/[:.]/g, "-");
      currentConfig.models.push({
        name: cleanName,
        provider: quickSyncProviderId,
        upstream_id: mId,
        free: isLocal,
        prices: {
          input: isLocal ? 0.0 : (mId.includes("deepseek") ? 0.14 : (mId.includes("gpt-4") ? 2.5 : 0.5)),
          output: isLocal ? 0.0 : (mId.includes("deepseek") ? 0.28 : (mId.includes("gpt-4") ? 10.0 : 1.5)),
          cache_read: isLocal ? 0.0 : 0.05,
          cache_write: isLocal ? 0.0 : 0.1,
        },
        capability: {
          coding: isLocal ? 75 : 90,
          math: isLocal ? 70 : 88,
          general: isLocal ? 80 : 92,
          knowledge: isLocal ? 75 : 90,
          summarisation: isLocal ? 80 : 90,
          agentic: isLocal ? 70 : 85,
          tool_use: isLocal ? 72 : 88,
        },
        context_tokens: isLocal ? 32768 : 128000,
        tools: true,
        vision: false,
      });
      addedCount++;
    }
  });

  savePolicyConfig();
  renderProviders();
  renderModels();
  closeModal("providerModelsModal");
  showToast(`同步成功！新增了 ${addedCount} 个路由模型。`, "success");
}

// ---------------------------------------------------------------------------
// Cascading Add / Edit Model Modal
// ---------------------------------------------------------------------------
function openAddModelModal() {
  document.getElementById("modelEditIndex").value = "-1";
  document.getElementById("modelModalTitle").textContent = "添加路由候选模型";
  document.getElementById("saveModelBtn").textContent = "保存模型到路由目录";
  document.getElementById("modelLockNotice").style.display = "none";

  const provSelect = document.getElementById("modelProvSelect");
  const upstreamSelect = document.getElementById("modelUpstreamSelect");
  const chkFree = document.getElementById("modelFreeCheckbox");

  provSelect.disabled = false;
  upstreamSelect.disabled = false;
  chkFree.disabled = false;

  provSelect.innerHTML = "";

  if (!currentConfig.providers || Object.keys(currentConfig.providers).length === 0) {
    showToast("请先添加至少一个模型提供商", "warning");
    return;
  }

  for (const provId of Object.keys(currentConfig.providers)) {
    const opt = document.createElement("option");
    opt.value = provId;
    opt.textContent = `${provId} (${currentConfig.providers[provId].base_url})`;
    provSelect.appendChild(opt);
  }

  // Reset capabilities defaults
  document.getElementById("capCodingInput").value = "0.85";
  document.getElementById("capMathInput").value = "0.85";
  document.getElementById("capReasoningInput").value = "0.85";
  document.getElementById("capGeneralInput").value = "0.85";
  document.getElementById("modelTimeoutInput").value = "";

  switchModelModalTab("basics");
  onModelProviderChanged();
  updatePricingUsdDisplay();
  document.getElementById("modelModal").style.display = "flex";

  const initialUpstream = document.getElementById("modelUpstreamSelect")?.value;
  if (initialUpstream && initialUpstream !== "__custom__") {
    matchAndRenderArenaCard(initialUpstream);
  }
}

function openEditModelModal(modelIdx) {
  if (!currentConfig || !currentConfig.models || !currentConfig.models[modelIdx]) return;
  const m = currentConfig.models[modelIdx];

  document.getElementById("modelEditIndex").value = modelIdx;
  document.getElementById("modelModalTitle").textContent = `编辑路由模型属性: [${m.name}]`;
  document.getElementById("saveModelBtn").textContent = "保存修改";
  document.getElementById("modelLockNotice").style.display = "block";

  // Lock fixed immutable properties
  const provSelect = document.getElementById("modelProvSelect");
  const upstreamSelect = document.getElementById("modelUpstreamSelect");
  const chkFree = document.getElementById("modelFreeCheckbox");

  provSelect.disabled = true;
  upstreamSelect.disabled = true;
  chkFree.disabled = true;

  // Populate providers select
  provSelect.innerHTML = "";
  for (const provId of Object.keys(currentConfig.providers || {})) {
    const opt = document.createElement("option");
    opt.value = provId;
    opt.textContent = `${provId} (${currentConfig.providers[provId].base_url})`;
    provSelect.appendChild(opt);
  }
  provSelect.value = m.provider || "";

  // Populate upstream select
  upstreamSelect.innerHTML = "";
  const optCurrent = document.createElement("option");
  optCurrent.value = m.upstream_id || "default";
  optCurrent.textContent = m.upstream_id || "default";
  upstreamSelect.appendChild(optCurrent);
  upstreamSelect.value = m.upstream_id || "default";

  document.getElementById("customModelInputBox").style.display = "none";
  document.getElementById("modelNameInput").value = m.name || "";
  chkFree.checked = !!m.free;

  // Fill pricing
  const p = m.prices || { input: 0, output: 0, cache_read: 0 };
  document.getElementById("modelInputPrice").value = p.input !== undefined ? p.input : 0;
  document.getElementById("modelOutputPrice").value = p.output !== undefined ? p.output : 0;
  document.getElementById("modelCachePrice").value = p.cache_read !== undefined ? p.cache_read : 0;

  // Timeout input
  document.getElementById("modelTimeoutInput").value = (m.timeout_seconds !== undefined && m.timeout_seconds !== null) ? m.timeout_seconds : "";

  // Context tokens
  const ctxSelect = document.getElementById("modelContextSelect");
  const ctxVal = String(m.context_tokens || 131072);
  let matched = false;
  for (let i = 0; i < ctxSelect.options.length; i++) {
    if (ctxSelect.options[i].value === ctxVal) {
      ctxSelect.selectedIndex = i;
      matched = true;
      break;
    }
  }
  if (!matched) {
    const optCustom = document.createElement("option");
    optCustom.value = ctxVal;
    optCustom.textContent = `${Math.round(parseInt(ctxVal)/1024)}k (${ctxVal})`;
    ctxSelect.appendChild(optCustom);
    ctxSelect.value = ctxVal;
  }

  // Capabilities
  const cap = m.capability || {};
  const normalizeCap = (val) => {
    if (val === undefined || val === null) return "0.85";
    const num = parseFloat(val);
    return num > 1.0 ? (num / 100).toFixed(2) : num.toFixed(2);
  };
  document.getElementById("capCodingInput").value = normalizeCap(cap.coding);
  document.getElementById("capMathInput").value = normalizeCap(cap.math);
  document.getElementById("capReasoningInput").value = normalizeCap(cap.reasoning);
  document.getElementById("capGeneralInput").value = normalizeCap(cap.general);

  switchModelModalTab("basics");
  onModelFreeToggled();
  updatePricingUsdDisplay();
  document.getElementById("modelModal").style.display = "flex";

  // Trigger arena match for editing model
  matchAndRenderArenaCard(m.upstream_id || m.name);
}

function onModelProviderChanged() {
  const provId = document.getElementById("modelProvSelect").value;
  const p = currentConfig.providers[provId];
  const isLocal = p && (p.base_url.includes("localhost") || p.base_url.includes("127.0.0.1"));

  // Checkbox for free
  const chkFree = document.getElementById("modelFreeCheckbox");
  chkFree.checked = isLocal;
  onModelFreeToggled();

  // Populate Upstream Models
  const upstreamSelect = document.getElementById("modelUpstreamSelect");
  upstreamSelect.innerHTML = "";

  const known = discoveredProviderModels[provId] || [];
  if (known.length > 0) {
    known.forEach(mId => {
      const opt = document.createElement("option");
      opt.value = mId;
      opt.textContent = mId;
      upstreamSelect.appendChild(opt);
    });
  }

  // Add default option if local
  if (isLocal && !known.includes("default")) {
    const optDefault = document.createElement("option");
    optDefault.value = "default";
    optDefault.textContent = "default (LM Studio 当前活动模型)";
    upstreamSelect.appendChild(optDefault);
  }

  // Custom option
  const optCustom = document.createElement("option");
  optCustom.value = "__custom__";
  optCustom.textContent = "✏️ 手动输入其他自定义模型 ID...";
  upstreamSelect.appendChild(optCustom);

  onModelUpstreamChanged();
}

function onModelUpstreamChanged() {
  const upstreamSelect = document.getElementById("modelUpstreamSelect");
  const val = upstreamSelect.value;
  const customBox = document.getElementById("customModelInputBox");
  const nameInput = document.getElementById("modelNameInput");

  if (val === "__custom__") {
    customBox.style.display = "block";
    nameInput.value = "";
  } else {
    customBox.style.display = "none";
    const clean = val.split("/").pop().replace(/[:.]/g, "-");
    nameInput.value = clean;
    matchAndRenderArenaCard(val);
  }

  // Suggest pricing (in RMB/CNY as default base)
  const provId = document.getElementById("modelProvSelect").value;
  const p = currentConfig.providers[provId];
  const isLocal = p && (p.base_url.includes("localhost") || p.base_url.includes("127.0.0.1"));

  if (isLocal) {
    document.getElementById("modelInputPrice").value = 0;
    document.getElementById("modelOutputPrice").value = 0;
    document.getElementById("modelCachePrice").value = 0;
  } else if (val.includes("deepseek")) {
    document.getElementById("modelInputPrice").value = 1.0;
    document.getElementById("modelOutputPrice").value = 2.0;
    document.getElementById("modelCachePrice").value = 0.1;
  } else if (val.includes("gpt-4") || val.includes("claude-3-5")) {
    document.getElementById("modelInputPrice").value = 18.0;
    document.getElementById("modelOutputPrice").value = 72.0;
    document.getElementById("modelCachePrice").value = 9.0;
  } else {
    document.getElementById("modelInputPrice").value = 3.5;
    document.getElementById("modelOutputPrice").value = 10.5;
    document.getElementById("modelCachePrice").value = 0.7;
  }
  updatePricingUsdDisplay();
}

let customModelMatchTimer = null;
function onCustomModelIdInput() {
  const val = document.getElementById("modelCustomUpstreamInput").value.trim();
  const nameInput = document.getElementById("modelNameInput");
  if (val && (!nameInput.value || nameInput.value.length < 3)) {
    nameInput.value = val.split("/").pop().replace(/[:.]/g, "-");
  }
  clearTimeout(customModelMatchTimer);
  customModelMatchTimer = setTimeout(() => {
    if (val) matchAndRenderArenaCard(val);
  }, 350);
}

function onModelFreeToggled() {
  const isFree = document.getElementById("modelFreeCheckbox").checked;
  const inputEl = document.getElementById("modelInputPrice");
  const outEl = document.getElementById("modelOutputPrice");
  const cacheEl = document.getElementById("modelCachePrice");

  if (isFree) {
    inputEl.value = 0;
    outEl.value = 0;
    cacheEl.value = 0;
    inputEl.disabled = true;
    outEl.disabled = true;
    cacheEl.disabled = true;
  } else {
    inputEl.disabled = false;
    outEl.disabled = false;
    cacheEl.disabled = false;
  }
  updatePricingUsdDisplay();
}

function saveModelFromModal() {
  const editIdx = parseInt(document.getElementById("modelEditIndex").value);
  const provId = document.getElementById("modelProvSelect").value;
  let upstreamId = document.getElementById("modelUpstreamSelect").value;
  if (upstreamId === "__custom__") {
    upstreamId = document.getElementById("modelCustomUpstreamInput").value.trim();
    if (!upstreamId) {
      showToast("请输入自定义上游模型 ID", "warning");
      return;
    }
  }

  const name = document.getElementById("modelNameInput").value.trim() || upstreamId;
  const isFree = document.getElementById("modelFreeCheckbox").checked;
  const inputPrice = isFree ? 0 : (parseFloat(document.getElementById("modelInputPrice").value) || 0);
  const outPrice = isFree ? 0 : (parseFloat(document.getElementById("modelOutputPrice").value) || 0);
  const cachePrice = isFree ? 0 : (parseFloat(document.getElementById("modelCachePrice").value) || 0);
  const contextTokens = parseInt(document.getElementById("modelContextSelect").value) || 131072;

  const capCoding = parseFloat(document.getElementById("capCodingInput").value) || 0.85;
  const capMath = parseFloat(document.getElementById("capMathInput").value) || 0.85;
  const capReasoning = parseFloat(document.getElementById("capReasoningInput").value) || 0.85;
  const capGeneral = parseFloat(document.getElementById("capGeneralInput").value) || 0.85;
  const timeoutVal = document.getElementById("modelTimeoutInput").value.trim();
  const timeoutSec = timeoutVal ? (parseFloat(timeoutVal) || null) : null;

  if (!currentConfig.models) currentConfig.models = [];

  if (editIdx >= 0 && editIdx < currentConfig.models.length) {
    // Edit existing model: preserve immutable attributes (provider, upstream_id, free)
    const target = currentConfig.models[editIdx];
    target.name = name;
    target.context_tokens = contextTokens;
    target.prices = {
      input: inputPrice,
      output: outPrice,
      cache_read: cachePrice,
      cache_write: cachePrice * 2,
    };
    target.capability = {
      ...(target.capability || {}),
      coding: Math.round(capCoding * 100),
      math: Math.round(capMath * 100),
      reasoning: Math.round(capReasoning * 100),
      general: Math.round(capGeneral * 100),
    };
    if (timeoutSec !== null) {
      target.timeout_seconds = timeoutSec;
    } else {
      delete target.timeout_seconds;
    }
    savePolicyConfig();
    closeModal("modelModal");
    showToast(`模型 [${name}] 属性已成功更新！`, "success");
  } else {
    // Add new model
    currentConfig.models.push({
      name: name,
      provider: provId,
      upstream_id: upstreamId,
      free: isFree,
      prices: {
        input: inputPrice,
        output: outPrice,
        cache_read: cachePrice,
        cache_write: cachePrice * 2,
      },
      capability: {
        coding: Math.round(capCoding * 100),
        math: Math.round(capMath * 100),
        reasoning: Math.round(capReasoning * 100),
        general: Math.round(capGeneral * 100),
        knowledge: isFree ? 75 : 90,
        summarisation: isFree ? 80 : 90,
        agentic: isFree ? 70 : 85,
        tool_use: isFree ? 72 : 88,
      },
      context_tokens: contextTokens,
      tools: true,
      vision: false,
      ...(timeoutSec !== null ? { timeout_seconds: timeoutSec } : {}),
    });
    savePolicyConfig();
    closeModal("modelModal");
    showToast(`模型 [${name}] 已成功添加到路由目录！`, "success");
  }
}

// ---------------------------------------------------------------------------
// Model Live Capability & Availability Probe Inspector
// ---------------------------------------------------------------------------
async function probeModelCapability(modelIdx) {
  const m = currentConfig.models[modelIdx];
  if (!m) return;

  modelProbeCache[m.name] = { loading: true };
  renderModels();

  try {
    const res = await fetch("/api/model/probe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        provider_id: m.provider,
        upstream_id: m.upstream_id,
        prompt: "请用Python写一个快速排序函数，并用一句话解释其原理。",
      }),
    });
    const data = await res.json();
    modelProbeCache[m.name] = data;

    if (data.status === "ok") {
      showToast(`模型 [${m.name}] 探测成功！首字延迟: ${data.latency_ms}ms`, "success");
    } else {
      showToast(`模型 [${m.name}] 响应异常: ${data.error || '错误'}`, "danger");
    }
  } catch (err) {
    modelProbeCache[m.name] = {
      status: "error",
      latency_ms: 0,
      error: err.message,
    };
    showToast(`探测异常: ${err.message}`, "danger");
  } finally {
    renderModels();
  }
}

function showProbeDetail(modelIdx) {
  activeProbeModelIdx = modelIdx;
  const m = currentConfig.models[modelIdx];
  if (!m) return;

  const probe = modelProbeCache[m.name];
  document.getElementById("probeModalTitle").textContent = `模型可用性与能力探测：${m.name}`;
  const epInfo = probe && probe.resolved_base_url ? ` | 上游端点: ${probe.resolved_base_url}` : '';
  document.getElementById("probeModalSubtitle").textContent = `所属提供商: ${m.provider}${epInfo} | 上游标识: ${m.upstream_id} | 免费: ${m.free ? '是' : '计费'}`;

  const statusEl = document.getElementById("probeStatusText");
  const latEl = document.getElementById("probeLatencyNum");
  const tokEl = document.getElementById("probeTokensNum");
  const outBox = document.getElementById("probeOutputSnippet");

  if (!probe) {
    statusEl.innerHTML = `<span style="color:var(--text-muted)">未测试</span>`;
    latEl.textContent = `-- ms`;
    tokEl.textContent = `--`;
    outBox.textContent = `尚未对此模型发起探测。点击下方“⚡ 再次测试”即可执行实时能力探测。`;
  } else if (probe.status === "ok") {
    statusEl.innerHTML = `<span style="color:var(--success)">✓ 正常可用</span>`;
    latEl.textContent = `${probe.latency_ms} ms`;
    tokEl.textContent = `${probe.usage?.total_tokens || probe.usage?.completion_tokens || '--'}`;
    outBox.textContent = probe.reply_snippet || "（响应成功，生成回复为空）";
  } else {
    statusEl.innerHTML = `<span style="color:var(--danger)">⚠ 调用异常</span>`;
    latEl.textContent = `${probe.latency_ms} ms`;
    tokEl.textContent = `0`;
    outBox.textContent = `【上游端点返回错误信息】\n${probe.error || '未知异常'}`;
  }

  document.getElementById("probeDetailModal").style.display = "flex";
}

async function reRunModalProbe() {
  if (activeProbeModelIdx === null) return;
  const m = currentConfig.models[activeProbeModelIdx];
  const customPrompt = document.getElementById("probePromptInput").value.trim();

  const outBox = document.getElementById("probeOutputSnippet");
  outBox.textContent = "正在向模型上游端点发送实时推理探测请求...";
  document.getElementById("probeStatusText").innerHTML = `<span class="spinner" style="width:14px;height:14px;border-width:2px;"></span> 探测中...`;

  try {
    const res = await fetch("/api/model/probe", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        provider_id: m.provider,
        upstream_id: m.upstream_id,
        prompt: customPrompt || "请用Python写一个快速排序函数，并用一句话解释其原理。",
      }),
    });
    const data = await res.json();
    modelProbeCache[m.name] = data;
    showProbeDetail(activeProbeModelIdx);
    renderModels();
  } catch (err) {
    outBox.textContent = `网络或调用异常: ${err.message}`;
  }
}

function deleteProvider(id) {
  if (confirm(`确定要移除提供商 [${id}] 吗？`)) {
    delete currentConfig.providers[id];
    savePolicyConfig();
    renderProviders();
  }
}

function deleteModel(idx) {
  if (confirm(`确定要删除该模型吗？`)) {
    currentConfig.models.splice(idx, 1);
    savePolicyConfig();
    renderModels();
  }
}

// ---------------------------------------------------------------------------
// Model Modal Tabs Handling
// ---------------------------------------------------------------------------
function switchModelModalTab(tabKey) {
  const tabs = ["basics", "pricing", "capabilities"];
  tabs.forEach(t => {
    const pane = document.getElementById(`tabModel${t.charAt(0).toUpperCase() + t.slice(1)}`);
    const btn = document.getElementById(`btnTabModel${t.charAt(0).toUpperCase() + t.slice(1)}`);
    if (pane) pane.style.display = (t === tabKey) ? "flex" : "none";
    if (btn) btn.classList.toggle("active", t === tabKey);
  });
}

// ---------------------------------------------------------------------------
// LMSYS Chatbot Arena Authoritative Capability Matching & Adoption
// ---------------------------------------------------------------------------
let currentArenaMatchData = null;
let cachedLeaderboardList = [];

async function matchAndRenderArenaCard(query) {
  if (!query || !query.trim()) return;
  const card = document.getElementById("arenaMatchCard");
  if (!card) return;

  try {
    const res = await fetch(`/api/leaderboard/match?query=${encodeURIComponent(query.trim())}`);
    if (!res.ok) return;
    const data = await res.json();
    if (!data.matched || !data.model) return;

    currentArenaMatchData = data;
    renderArenaCard(data.model, data.candidates);
  } catch (err) {
    console.warn("Failed to match arena model:", err);
  }
}

function renderArenaCard(model, candidates = []) {
  if (!model) return;
  const nameEl = document.getElementById("arenaMatchedModelName");
  const orgEl = document.getElementById("arenaMatchedOrg");
  const licEl = document.getElementById("arenaMatchedLicense");
  const eloEl = document.getElementById("arenaValElo");
  const codEl = document.getElementById("arenaValCoding");
  const matEl = document.getElementById("arenaValMath");
  const reaEl = document.getElementById("arenaValReasoning");
  const genEl = document.getElementById("arenaValGeneral");

  const nCodEl = document.getElementById("arenaNormCoding");
  const nMatEl = document.getElementById("arenaNormMath");
  const nReaEl = document.getElementById("arenaNormReasoning");
  const nGenEl = document.getElementById("arenaNormGeneral");

  nameEl.textContent = model.display_name || model.model_name;
  orgEl.textContent = model.organization || "Unknown";
  licEl.textContent = model.license || "Proprietary";
  const isOpen = model.license && !["proprietary", "closed"].includes(model.license.toLowerCase());
  licEl.className = `arena-license-tag ${isOpen ? "open" : ""}`;

  eloEl.textContent = Math.round(model.rating_overall || 1200);
  codEl.textContent = Math.round(model.rating_coding || model.rating_overall || 1200);
  matEl.textContent = Math.round(model.rating_math || model.rating_overall || 1200);
  reaEl.textContent = Math.round(model.rating_hard || model.rating_overall || 1200);
  genEl.textContent = Math.round(model.rating_overall || 1200);

  const prof = model.normalized_profile || {};
  nCodEl.textContent = (prof.coding || 0.85).toFixed(2);
  nMatEl.textContent = (prof.math || 0.85).toFixed(2);
  nReaEl.textContent = (prof.reasoning || 0.85).toFixed(2);
  nGenEl.textContent = (prof.general || 0.85).toFixed(2);

  // Populate manual switcher dropdown
  const sel = document.getElementById("arenaManualSelect");
  if (sel) {
    sel.innerHTML = "";
    // Option for currently matched
    const optSelf = document.createElement("option");
    optSelf.value = model.model_name;
    optSelf.textContent = `⭐ ${model.display_name || model.model_name} (Elo ${Math.round(model.rating_overall || 1200)}) - 推荐匹配`;
    optSelf.selected = true;
    sel.appendChild(optSelf);

    // Populate other candidates or full list
    const candidateList = (candidates && candidates.length > 0) ? candidates : cachedLeaderboardList;
    candidateList.forEach(c => {
      if (c.model_name !== model.model_name) {
        const opt = document.createElement("option");
        opt.value = c.model_name;
        opt.textContent = `${c.display_name || c.model_name} (Elo ${Math.round(c.rating_overall || 1200)})`;
        sel.appendChild(opt);
      }
    });
  }
}

function applyArenaScoresToInputs() {
  if (!currentArenaMatchData || !currentArenaMatchData.model) {
    showToast("暂无可采纳的权威评分数据", "warning");
    return;
  }
  const prof = currentArenaMatchData.model.normalized_profile || currentArenaMatchData.normalized_capabilities || {};
  document.getElementById("capCodingInput").value = (prof.coding || 0.85).toFixed(2);
  document.getElementById("capMathInput").value = (prof.math || 0.85).toFixed(2);
  document.getElementById("capReasoningInput").value = (prof.reasoning || 0.85).toFixed(2);
  document.getElementById("capGeneralInput").value = (prof.general || 0.85).toFixed(2);

  const mName = currentArenaMatchData.model.display_name || currentArenaMatchData.model.model_name;
  showToast(`已成功将权威模型 [${mName}] 的四维评分回填至表单！`, "success");
}

function onArenaManualSelectChanged() {
  const sel = document.getElementById("arenaManualSelect");
  const targetName = sel.value;
  let target = cachedLeaderboardList.find(m => m.model_name === targetName);
  if (!target && currentArenaMatchData && currentArenaMatchData.candidates) {
    target = currentArenaMatchData.candidates.find(m => m.model_name === targetName);
  }
  if (target) {
    currentArenaMatchData.model = target;
    currentArenaMatchData.normalized_capabilities = target.normalized_profile || {};
    renderArenaCard(target, currentArenaMatchData.candidates);
    showToast(`已换选权威参考模型: ${target.display_name || target.model_name}`, "info");
  }
}

// ---------------------------------------------------------------------------
// Leaderboard Page & Candidate Models Cross-Comparison Logic
// ---------------------------------------------------------------------------
let currentLbCategory = "overall";
let isCandidateComparisonOnly = false;
let lbSearchTimer = null;

// ---------------------------------------------------------------------------
// Leaderboard Multi-Subset Columns & Sorting State
// ---------------------------------------------------------------------------
const DEFAULT_VISIBLE_COLUMNS = ["text", "webdev", "math", "hard", "agent", "vision", "text_factuality"];
let lbVisibleColumns = [];
let lbSubsetsMeta = null;
let lbCurrentSortCol = "text";
let lbCurrentSortOrder = "desc";

const LB_CAT_TO_COL = {
  overall: "text",
  coding: "webdev",
  math: "math",
  hard: "hard",
};

const LB_COL_TO_CAT = {
  text: "overall",
  webdev: "coding",
  math: "math",
  hard: "hard",
};

function updateCategoryButtonsState(activeCat) {
  const btns = {
    overall: document.getElementById("btnLbCatOverall"),
    coding: document.getElementById("btnLbCatCoding"),
    math: document.getElementById("btnLbCatMath"),
    hard: document.getElementById("btnLbCatHard"),
  };
  Object.keys(btns).forEach(k => {
    if (btns[k]) btns[k].classList.toggle("active", k === activeCat);
  });
}

function initLeaderboardColumns() {
  const saved = localStorage.getItem("amra_lb_visible_columns");
  if (saved) {
    try {
      const arr = JSON.parse(saved);
      if (Array.isArray(arr) && arr.length > 0) {
        lbVisibleColumns = arr;
      } else {
        lbVisibleColumns = [...DEFAULT_VISIBLE_COLUMNS];
      }
    } catch (e) {
      lbVisibleColumns = [...DEFAULT_VISIBLE_COLUMNS];
    }
  } else {
    lbVisibleColumns = [...DEFAULT_VISIBLE_COLUMNS];
  }
  updateSelectedColCountBadge();
}

function updateSelectedColCountBadge() {
  const badge = document.getElementById("selectedColCountBadge");
  if (badge) {
    badge.textContent = lbVisibleColumns.length;
  }
}

function toggleColumnPickerDropdown(event) {
  if (event) event.stopPropagation();
  const dd = document.getElementById("lbColumnsDropdown");
  if (!dd) return;
  const isHidden = (dd.style.display === "none" || !dd.style.display);
  dd.style.display = isHidden ? "flex" : "none";
}

function getSubsetDisplayName(sub) {
  if (sub && sub.zh && sub.en) {
    return { zh: sub.zh, en: sub.en };
  }
  const label = (sub && sub.label) ? sub.label : (sub && sub.key ? sub.key : "");
  const match = label.match(/^(.*?)\s*\((.*?)\)$/);
  if (match) {
    return { zh: match[1].trim(), en: match[2].trim() };
  }
  return { zh: label, en: (sub && sub.key) ? sub.key : "" };
}

function renderColumnPicker(meta) {
  if (!meta || !meta.categories) return;
  lbSubsetsMeta = meta;
  const container = document.getElementById("colPickerGroupsContainer");
  if (!container) return;

  container.innerHTML = "";
  meta.categories.forEach(cat => {
    const groupDiv = document.createElement("div");
    groupDiv.className = "col-picker-group";

    const titleDiv = document.createElement("div");
    titleDiv.className = "col-group-title";
    titleDiv.innerHTML = `
      <div class="col-group-title-left">
        <span>${cat.name}</span>
        <span style="font-size:10px; color:var(--text-muted); font-weight:400;">(${cat.subsets.length})</span>
      </div>
      <div class="col-group-actions">
        <button type="button" class="col-group-btn" onclick="toggleCategoryColumns('${cat.id}', true)">全选</button>
        <span style="color: #cbd5e1; font-size: 10px;">|</span>
        <button type="button" class="col-group-btn clear" onclick="toggleCategoryColumns('${cat.id}', false)">清空</button>
      </div>
    `;
    groupDiv.appendChild(titleDiv);

    const itemsDiv = document.createElement("div");
    itemsDiv.className = "col-picker-items";

    cat.subsets.forEach(sub => {
      const names = getSubsetDisplayName(sub);
      const label = document.createElement("label");
      label.className = "col-picker-item";
      const isChecked = lbVisibleColumns.includes(sub.key);
      label.innerHTML = `
        <input type="checkbox" value="${sub.key}" ${isChecked ? "checked" : ""} onchange="onColumnCheckboxChange('${sub.key}', this.checked)">
        <div class="col-picker-text">
          <span class="col-picker-title" title="${sub.label || names.zh}">${sub.icon || "📊"} ${names.zh}</span>
          <span class="col-picker-sub" title="${names.en}">${names.en}</span>
        </div>
      `;
      itemsDiv.appendChild(label);
    });

    groupDiv.appendChild(itemsDiv);
    container.appendChild(groupDiv);
  });
}

function toggleCategoryColumns(catId, enableAll) {
  if (!lbSubsetsMeta || !lbSubsetsMeta.categories) return;
  const cat = lbSubsetsMeta.categories.find(c => c.id === catId);
  if (!cat || !cat.subsets) return;

  const catKeys = cat.subsets.map(s => s.key);

  if (enableAll) {
    catKeys.forEach(k => {
      if (!lbVisibleColumns.includes(k)) {
        lbVisibleColumns.push(k);
      }
    });
  } else {
    // If clearing, ensure we still keep at least 1 column globally
    const remaining = lbVisibleColumns.filter(k => !catKeys.includes(k));
    if (remaining.length === 0) {
      showToast("全局至少需保留一个显示列，不能全部清空", "warning");
      lbVisibleColumns = [catKeys[0] || "text"];
    } else {
      lbVisibleColumns = remaining;
    }
  }

  localStorage.setItem("amra_lb_visible_columns", JSON.stringify(lbVisibleColumns));
  updateSelectedColCountBadge();

  // Sync checkboxes in DOM
  catKeys.forEach(k => {
    const chk = document.querySelector(`.col-picker-item input[value="${k}"]`);
    if (chk) chk.checked = lbVisibleColumns.includes(k);
  });

  renderLeaderboardTableHeader();
  renderLeaderboardTable(cachedLeaderboardList);
}

function onColumnCheckboxChange(colKey, isChecked) {
  if (isChecked) {
    if (!lbVisibleColumns.includes(colKey)) {
      lbVisibleColumns.push(colKey);
    }
  } else {
    if (lbVisibleColumns.length <= 1) {
      showToast("至少需保留一个显示列", "warning");
      const chk = document.querySelector(`.col-picker-item input[value="${colKey}"]`);
      if (chk) chk.checked = true;
      return;
    }
    lbVisibleColumns = lbVisibleColumns.filter(k => k !== colKey);
  }
  localStorage.setItem("amra_lb_visible_columns", JSON.stringify(lbVisibleColumns));
  updateSelectedColCountBadge();
  renderLeaderboardTableHeader();
  renderLeaderboardTable(cachedLeaderboardList);
}

function resetDefaultColumns() {
  lbVisibleColumns = [...DEFAULT_VISIBLE_COLUMNS];
  localStorage.setItem("amra_lb_visible_columns", JSON.stringify(lbVisibleColumns));
  updateSelectedColCountBadge();
  if (lbSubsetsMeta) {
    renderColumnPicker(lbSubsetsMeta);
  }
  renderLeaderboardTableHeader();
  renderLeaderboardTable(cachedLeaderboardList);
  showToast("已恢复默认显示列 (7项核心维度)", "info");
}

function sortLeaderboardByColumn(colKey) {
  if (lbCurrentSortCol === colKey) {
    lbCurrentSortOrder = (lbCurrentSortOrder === "desc" ? "asc" : "desc");
  } else {
    lbCurrentSortCol = colKey;
    lbCurrentSortOrder = "desc";
  }

  lbCurrentPage = 1;

  // Sync category buttons state
  const matchedCat = (lbCurrentSortOrder === "desc") ? LB_COL_TO_CAT[colKey] : null;
  currentLbCategory = matchedCat || "";
  updateCategoryButtonsState(matchedCat);

  cachedLeaderboardList.sort((a, b) => {
    let valA = 0.0;
    let valB = 0.0;

    if (a.subsets && a.subsets[colKey]) {
      valA = parseFloat(a.subsets[colKey].elo) || 0.0;
    } else if (a[colKey] !== undefined) {
      valA = parseFloat(a[colKey]) || 0.0;
    }

    if (b.subsets && b.subsets[colKey]) {
      valB = parseFloat(b.subsets[colKey].elo) || 0.0;
    } else if (b[colKey] !== undefined) {
      valB = parseFloat(b[colKey]) || 0.0;
    }

    return lbCurrentSortOrder === "asc" ? (valA - valB) : (valB - valA);
  });

  renderLeaderboardTableHeader();
  renderLeaderboardTable(cachedLeaderboardList);
}

function renderLeaderboardTableHeader() {
  const thead = document.getElementById("leaderboardTableHeader");
  if (!thead) return;

  const reg = (lbSubsetsMeta && lbSubsetsMeta.registry) ? lbSubsetsMeta.registry : {};

  let colsHtml = `
    <tr>
      <th style="width: 55px; text-align: center;">排名</th>
      <th style="min-width: 165px;">模型名称 (Model)</th>
      <th style="min-width: 95px;">研发机构 (Org)</th>
  `;

  lbVisibleColumns.forEach(colKey => {
    const meta = reg[colKey] || { label: colKey, icon: "📊" };
    const names = getSubsetDisplayName(meta);
    const isSorted = (lbCurrentSortCol === colKey);
    const arrow = isSorted ? (lbCurrentSortOrder === "asc" ? "▲" : "▼") : "↕";
    colsHtml += `
      <th class="sortable-th ${isSorted ? "sorted" : ""}" style="text-align: right; min-width: 95px;" onclick="sortLeaderboardByColumn('${colKey}')" title="点击按此列${isSorted && lbCurrentSortOrder === 'desc' ? '升序' : '降序'}排序">
        <div class="th-two-line">
          <div class="th-main-title">
            <span>${meta.icon || ''}</span>
            <span>${names.zh}</span>
          </div>
          <div class="th-sub-title">
            <span>${names.en}</span>
            <span class="sort-indicator">${arrow}</span>
          </div>
        </div>
      </th>
    `;
  });

  colsHtml += `
      <th style="min-width: 90px;">开源许可</th>
      <th style="min-width: 75px;">路由状态</th>
      <th style="min-width: 80px;">操作</th>
    </tr>
  `;

  thead.innerHTML = colsHtml;
}

function getColumnColorStyle(colKey) {
  const colorMap = {
    text: "color: #0f172a; font-weight: 700;",
    webdev: "color: #0284c7; font-weight: 700;",
    math: "color: #8b5cf6; font-weight: 700;",
    hard: "color: #059669; font-weight: 700;",
    agent: "color: #d97706; font-weight: 700;",
    agent_tool_hallucination: "color: #b45309; font-weight: 700;",
    agent_bash_recovery_steps: "color: #ea580c; font-weight: 700;",
    agent_steerability: "color: #ca8a04; font-weight: 700;",
    agent_task_outcome_explicit: "color: #16a34a; font-weight: 700;",
    agent_praise_complaint: "color: #0d9488; font-weight: 700;",
    vision: "color: #0891b2; font-weight: 700;",
    document: "color: #4f46e5; font-weight: 700;",
    search: "color: #2563eb; font-weight: 700;",
    text_factuality: "color: #15803d; font-weight: 700;",
    search_factuality: "color: #047857; font-weight: 700;",
    text_style_control: "color: #7c3aed; font-weight: 700;",
    document_style_control: "color: #9333ea; font-weight: 700;",
    search_style_control: "color: #c026d3; font-weight: 700;",
    vision_style_control: "color: #db2777; font-weight: 700;",
    image_edit: "color: #e11d48; font-weight: 700;",
    image_to_video: "color: #be123c; font-weight: 700;",
    text_to_image: "color: #e11d48; font-weight: 700;",
    text_to_video: "color: #9f1239; font-weight: 700;",
    video_edit: "color: #881337; font-weight: 700;",
  };
  return colorMap[colKey] || "color: var(--text-main); font-weight: 700;";
}

// ---------------------------------------------------------------------------
// Leaderboard Pagination State & Controls
// ---------------------------------------------------------------------------
let lbCurrentPage = 1;
let lbPageSize = 20;

function initLeaderboardPagination() {
  const savedSize = localStorage.getItem("amra_lb_page_size");
  if (savedSize && ["20", "50", "100", "200"].includes(savedSize)) {
    lbPageSize = parseInt(savedSize, 10);
  } else {
    lbPageSize = 20;
  }
  const select = document.getElementById("lbPageSizeSelect");
  if (select) {
    select.value = String(lbPageSize);
  }
}

function onLeaderboardPageSizeChange(val) {
  lbPageSize = parseInt(val, 10) || 20;
  localStorage.setItem("amra_lb_page_size", String(lbPageSize));
  lbCurrentPage = 1;
  renderLeaderboardTable(cachedLeaderboardList);
}

function changeLeaderboardPage(page) {
  const totalPages = Math.ceil(cachedLeaderboardList.length / lbPageSize) || 1;
  const target = Math.max(1, Math.min(page, totalPages));
  if (target !== lbCurrentPage) {
    lbCurrentPage = target;
    renderLeaderboardTable(cachedLeaderboardList);
    const card = document.querySelector(".integrated-table-card");
    if (card && card.getBoundingClientRect().top < 0) {
      card.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }
}

function prevLeaderboardPage() {
  if (lbCurrentPage > 1) {
    changeLeaderboardPage(lbCurrentPage - 1);
  }
}

function nextLeaderboardPage() {
  const totalPages = Math.ceil(cachedLeaderboardList.length / lbPageSize) || 1;
  if (lbCurrentPage < totalPages) {
    changeLeaderboardPage(lbCurrentPage + 1);
  }
}

function lastLeaderboardPage() {
  const totalPages = Math.ceil(cachedLeaderboardList.length / lbPageSize) || 1;
  changeLeaderboardPage(totalPages);
}

function jumpLeaderboardPage() {
  const input = document.getElementById("lbJumpPageInput");
  if (!input) return;
  const val = parseInt(input.value, 10);
  const totalPages = Math.ceil(cachedLeaderboardList.length / lbPageSize) || 1;
  if (!val || isNaN(val)) return;
  changeLeaderboardPage(Math.max(1, Math.min(val, totalPages)));
  input.value = "";
}

function renderLeaderboardPagination(totalItems) {
  const totalPages = Math.max(1, Math.ceil(totalItems / lbPageSize));
  if (lbCurrentPage > totalPages) {
    lbCurrentPage = totalPages;
  }

  const totalCountEl = document.getElementById("lbTotalCount");
  const currentPageEl = document.getElementById("lbCurrentPage");
  const totalPagesEl = document.getElementById("lbTotalPages");
  const btnFirst = document.getElementById("btnLbFirstPage");
  const btnPrev = document.getElementById("btnLbPrevPage");
  const btnNext = document.getElementById("btnLbNextPage");
  const btnLast = document.getElementById("btnLbLastPage");
  const numbersContainer = document.getElementById("lbPaginationNumbers");

  if (totalCountEl) totalCountEl.textContent = totalItems;
  if (currentPageEl) currentPageEl.textContent = totalItems > 0 ? lbCurrentPage : 0;
  if (totalPagesEl) totalPagesEl.textContent = totalPages;

  if (btnFirst) btnFirst.disabled = (lbCurrentPage <= 1 || totalItems === 0);
  if (btnPrev) btnPrev.disabled = (lbCurrentPage <= 1 || totalItems === 0);
  if (btnNext) btnNext.disabled = (lbCurrentPage >= totalPages || totalItems === 0);
  if (btnLast) btnLast.disabled = (lbCurrentPage >= totalPages || totalItems === 0);

  if (!numbersContainer) return;
  numbersContainer.innerHTML = "";
  if (totalItems === 0) return;

  const pagesToShow = [];
  if (totalPages <= 7) {
    for (let i = 1; i <= totalPages; i++) pagesToShow.push(i);
  } else {
    if (lbCurrentPage <= 4) {
      for (let i = 1; i <= 5; i++) pagesToShow.push(i);
      pagesToShow.push("...");
      pagesToShow.push(totalPages);
    } else if (lbCurrentPage >= totalPages - 3) {
      pagesToShow.push(1);
      pagesToShow.push("...");
      for (let i = totalPages - 4; i <= totalPages; i++) pagesToShow.push(i);
    } else {
      pagesToShow.push(1);
      pagesToShow.push("...");
      pagesToShow.push(lbCurrentPage - 1);
      pagesToShow.push(lbCurrentPage);
      pagesToShow.push(lbCurrentPage + 1);
      pagesToShow.push("...");
      pagesToShow.push(totalPages);
    }
  }

  pagesToShow.forEach(p => {
    if (p === "...") {
      const span = document.createElement("span");
      span.className = "pagination-ellipsis";
      span.textContent = "…";
      numbersContainer.appendChild(span);
    } else {
      const btn = document.createElement("button");
      btn.className = `btn btn-sm btn-secondary pagination-btn ${p === lbCurrentPage ? 'active' : ''}`;
      btn.textContent = p;
      btn.onclick = () => changeLeaderboardPage(p);
      numbersContainer.appendChild(btn);
    }
  });
}

// Sub-feature tab switching inside Leaderboard page
function switchLeaderboardSubtab(subtabKey) {
  const btnTable = document.getElementById("btnSubnavLbTable");
  const btnCompare = document.getElementById("btnSubnavLbCompare");
  const paneTable = document.getElementById("subpaneLeaderboardTable");
  const paneCompare = document.getElementById("subpaneLeaderboardCompare");

  if (btnTable) btnTable.classList.toggle("active", subtabKey === "table");
  if (btnCompare) btnCompare.classList.toggle("active", subtabKey === "compare");

  if (paneTable) paneTable.style.display = subtabKey === "table" ? "block" : "none";
  if (paneCompare) paneCompare.style.display = subtabKey === "compare" ? "block" : "none";
}

function toggleTableCandidateFilter(checked) {
  isCandidateComparisonOnly = checked;
  lbCurrentPage = 1;
  loadLeaderboardData();
}

async function loadLeaderboardData() {
  const searchInput = document.getElementById("lbSearchInput");
  const query = searchInput ? searchInput.value.trim() : "";
  const isOnlyOpen = document.getElementById("chkLbOpenSource")?.checked || false;

  let url = `/api/leaderboard?search=${encodeURIComponent(query)}&open_source_only=${isOnlyOpen}&candidate_only=${isCandidateComparisonOnly}`;
  if (lbCurrentSortCol) {
    url += `&sort_col=${encodeURIComponent(lbCurrentSortCol)}&sort_order=${lbCurrentSortOrder}`;
  } else if (currentLbCategory) {
    url += `&category=${currentLbCategory}`;
  }

  try {
    const res = await fetch(url);
    if (!res.ok) return;
    const json = await res.json();
    cachedLeaderboardList = json.data || [];

    if (json.subsets_meta) {
      renderColumnPicker(json.subsets_meta);
      renderLeaderboardTableHeader();
    }
    updateCategoryButtonsState(currentLbCategory);

    // Update sync tag
    const syncTag = document.getElementById("lbSyncTimeTag");
    if (syncTag && json.last_updated) {
      syncTag.textContent = `上次同步: ${json.last_updated}`;
    }

    // Update subnav candidate counts
    const count = (json.candidate_comparison || []).length;
    const subnavBadge = document.getElementById("subnavCandidateCountBadge");
    if (subnavBadge) subnavBadge.textContent = count;

    // Render candidate comparison cards
    renderCandidateComparisonCards(json.candidate_comparison || []);

    // Render leaderboard table with dynamic columns & pagination
    renderLeaderboardTable(cachedLeaderboardList);
  } catch (err) {
    console.error("Failed to load leaderboard data:", err);
  }
}

function renderCandidateComparisonCards(comparisons) {
  const container = document.getElementById("candidateCardsGrid");
  const countBadge = document.getElementById("candidateCountBadge");
  if (!container) return;

  if (countBadge) {
    countBadge.textContent = `${comparisons.length} 个路由候选模型`;
  }

  if (comparisons.length === 0) {
    container.innerHTML = `<div style="grid-column: 1/-1; padding: 24px; text-align: center; color: var(--text-muted); background: white; border-radius: 12px; border: 1px dashed var(--border-color);">
      当前暂未配置任何路由模型，可在“提供商与模型管理”中一键添加。
    </div>`;
    return;
  }

  container.innerHTML = "";
  comparisons.forEach(item => {
    const m = item.configured_model;
    const lb = item.matched_leaderboard;
    const isFree = !!m.free;
    const prices = m.prices || {};
    const cap = m.capability || {};

    const hasLbProf = lb && lb.normalized_profile;
    const prof = hasLbProf ? lb.normalized_profile : null;

    // 权威实测分数优先，若未匹配到权威则回退到本地配置分
    const normCoding = prof && prof.coding !== undefined ? prof.coding.toFixed(2) : ((cap.coding || 85) / 100).toFixed(2);
    const normMath = prof && prof.math !== undefined ? prof.math.toFixed(2) : ((cap.math || 85) / 100).toFixed(2);
    const normRea = prof && prof.reasoning !== undefined ? prof.reasoning.toFixed(2) : ((cap.reasoning || 85) / 100).toFixed(2);
    const normGen = prof && prof.general !== undefined ? prof.general.toFixed(2) : ((cap.general || 85) / 100).toFixed(2);

    const avgCap = ((parseFloat(normCoding) + parseFloat(normMath) + parseFloat(normRea) + parseFloat(normGen)) / 4).toFixed(2);
    const blendPrice = isFree ? 0 : ((prices.input || 0) * 0.7 + (prices.output || 0) * 0.3);
    const valueIndex = isFree ? "极高 (零成本)" : (blendPrice > 0 ? (avgCap / blendPrice * 10).toFixed(1) : "高");

    const badgeHtml = hasLbProf 
      ? `<span style="background: #e0f2fe; color: #0284c7; font-size: 11px; padding: 2px 7px; border-radius: 9999px; font-weight: 600;">权威实测基准</span>`
      : `<span style="background: #f1f5f9; color: #64748b; font-size: 11px; padding: 2px 7px; border-radius: 9999px; font-weight: 600;">本地自定义</span>`;

    const card = document.createElement("div");
    card.className = "candidate-card";
    card.innerHTML = `
      <div class="candidate-card-header">
        <div>
          <div class="candidate-card-title">${m.name}</div>
          <div class="candidate-card-prov">提供商: <strong>${m.provider}</strong> | ${m.upstream_id || 'default'}</div>
        </div>
        <div>
          ${isFree ? '<span class="kpi-badge badge-green">免费零成本</span>' : `<span class="kpi-badge badge-blue">¥${prices.input || 0} / ¥${prices.output || 0}</span>`}
        </div>
      </div>

      <div style="background: #f8fafc; border-radius: 8px; padding: 8px 10px; font-size: 11.5px; border: 1px solid #f1f5f9; display: flex; justify-content: space-between; align-items: center;">
        <div style="display:flex; align-items:center; gap:6px;">
          <span>权威锚定: <strong>${lb ? (lb.display_name || lb.model_name) : '通用基准'}</strong></span>
          ${badgeHtml}
        </div>
        <span style="font-weight: 700; color: var(--primary); font-family: var(--font-mono);">Elo ${lb ? Math.round(lb.rating_overall) : '--'}</span>
      </div>

      <div class="candidate-card-metrics">
        <div class="cap-bar-row">
          <span class="cap-bar-lbl">代码能力</span>
          <div class="cap-bar-track"><div class="cap-bar-fill coding" style="width: ${normCoding * 100}%"></div></div>
          <span class="cap-bar-val">${normCoding}</span>
        </div>
        <div class="cap-bar-row">
          <span class="cap-bar-lbl">数理推导</span>
          <div class="cap-bar-track"><div class="cap-bar-fill math" style="width: ${normMath * 100}%"></div></div>
          <span class="cap-bar-val">${normMath}</span>
        </div>
        <div class="cap-bar-row">
          <span class="cap-bar-lbl">逻辑推理</span>
          <div class="cap-bar-track"><div class="cap-bar-fill reasoning" style="width: ${normRea * 100}%"></div></div>
          <span class="cap-bar-val">${normRea}</span>
        </div>
        <div class="cap-bar-row">
          <span class="cap-bar-lbl">综合问答</span>
          <div class="cap-bar-track"><div class="cap-bar-fill general" style="width: ${normGen * 100}%"></div></div>
          <span class="cap-bar-val">${normGen}</span>
        </div>
      </div>

      <div class="candidate-card-footer" style="padding-bottom: 8px;">
        <span>综合能力: <strong style="color:var(--primary);">${avgCap}</strong></span>
        <span>综合性价比: <strong style="color:#059669;">${valueIndex}</strong></span>
      </div>

      <button class="btn btn-secondary btn-sm" style="width: 100%; font-size: 11.5px; padding: 6px 10px; display: flex; align-items: center; justify-content: center; gap: 5px;" onclick="applyLeaderboardToModel('${m.name}', '${lb ? (lb.model_name || '') : ''}')">
        <span>⚡ 同步此权威评分至路由配置</span>
      </button>
    `;
    container.appendChild(card);
  });
}

function renderLeaderboardTable(models) {
  const tbody = document.getElementById("leaderboardTableBody");
  if (!tbody) return;

  renderLeaderboardPagination(models.length);

  const totalCols = 3 + lbVisibleColumns.length + 3;

  if (models.length === 0) {
    tbody.innerHTML = `<tr><td colspan="${totalCols}" style="text-align: center; color: var(--text-muted); padding: 30px;">未检索到符合条件的评测大模型</td></tr>`;
    return;
  }

  const startIndex = (lbCurrentPage - 1) * lbPageSize;
  const endIndex = Math.min(startIndex + lbPageSize, models.length);
  const pageModels = models.slice(startIndex, endIndex);

  const reg = (lbSubsetsMeta && lbSubsetsMeta.registry) ? lbSubsetsMeta.registry : {};

  tbody.innerHTML = "";
  pageModels.forEach((m, idx) => {
    const isCandidate = !!m.is_candidate;
    const tr = document.createElement("tr");
    if (isCandidate) {
      tr.style.background = "#f0fdf4";
    }

    const rankDisplay = startIndex + idx + 1;
    const rankBadgeClass = rankDisplay === 1 ? 'color: #eab308; font-weight:800;' : (rankDisplay <= 3 ? 'color: #0284c7; font-weight:700;' : 'color: var(--text-muted);');

    let rowScoresHtml = "";
    lbVisibleColumns.forEach(colKey => {
      const sub = (m.subsets && m.subsets[colKey]) ? m.subsets[colKey] : null;
      let eloVal = 1200.0;
      let rawScore = null;
      let obsCount = null;
      let subRank = null;

      if (sub) {
        eloVal = sub.elo || 1200.0;
        rawScore = sub.raw_score;
        obsCount = sub.obs_count;
        subRank = sub.rank;
      } else {
        if (colKey === "text") eloVal = m.rating_overall || 1200;
        else if (colKey === "webdev") eloVal = m.rating_coding || m.rating_overall || 1200;
        else if (colKey === "math") eloVal = m.rating_math || m.rating_overall || 1200;
        else if (colKey === "hard") eloVal = m.rating_hard || m.rating_overall || 1200;
        else eloVal = m.rating_overall || 1200;
      }

      const meta = reg[colKey] || { label: colKey };
      let tooltip = `子项: ${meta.label}\n等价 Elo: ${Math.round(eloVal)}`;
      if (rawScore !== null && rawScore !== undefined) {
        tooltip += `\n原始得分: ${rawScore > 0 ? '+' : ''}${rawScore}`;
      }
      if (obsCount) {
        tooltip += `\n样本量: ${obsCount.toLocaleString()} 条`;
      }
      if (subRank && subRank < 999) {
        tooltip += `\n专项排名: #${subRank}`;
      }

      rowScoresHtml += `
        <td style="text-align: right;">
          <span class="elo-score-cell" style="${getColumnColorStyle(colKey)}" data-tooltip="${tooltip}">
            ${Math.round(eloVal)}
          </span>
        </td>
      `;
    });

    tr.innerHTML = `
      <td style="text-align: center; ${rankBadgeClass} font-family: var(--font-mono);">
        #${rankDisplay}
      </td>
      <td>
        <div style="display: flex; align-items: center; gap: 8px;">
          <strong style="color: var(--text-main); font-size: 13.5px;">${m.display_name || m.model_name}</strong>
          ${isCandidate ? '<span class="candidate-badge-active">✨ 路由候选</span>' : ''}
        </div>
        <span style="font-size: 11px; color: var(--text-muted); font-family: var(--font-mono);">${m.model_name}</span>
      </td>
      <td><span style="font-size: 12.5px;">${m.organization || 'OpenAI'}</span></td>
      ${rowScoresHtml}
      <td>
        <span class="arena-license-tag ${m.license && !['proprietary','closed'].includes(m.license.toLowerCase()) ? 'open' : ''}">
          ${m.license || 'Proprietary'}
        </span>
      </td>
      <td>
        ${isCandidate ? '<span style="color:#059669; font-weight:600; font-size:12px;">✅ 参与决策</span>' : '<span style="color:var(--text-muted); font-size:12px;">未添加</span>'}
      </td>
      <td>
        <button class="btn btn-secondary btn-sm" onclick="quickAdoptLeaderboardModel('${m.model_name}')" title="根据此模型创建路由候选并预填权威画像">
          ⚡ 引入配置
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function setLeaderboardCategory(cat) {
  currentLbCategory = cat;
  const targetCol = LB_CAT_TO_COL[cat] || "text";

  // If target column is not in visible columns, add it automatically
  if (!lbVisibleColumns.includes(targetCol)) {
    lbVisibleColumns.push(targetCol);
    localStorage.setItem("amra_lb_visible_columns", JSON.stringify(lbVisibleColumns));
    updateSelectedColCountBadge();
    if (lbSubsetsMeta) {
      renderColumnPicker(lbSubsetsMeta);
    }
  }

  lbCurrentSortCol = targetCol;
  lbCurrentSortOrder = "desc";
  lbCurrentPage = 1;

  updateCategoryButtonsState(cat);

  // Optimistic client-side sort
  if (cachedLeaderboardList && cachedLeaderboardList.length > 0) {
    cachedLeaderboardList.sort((a, b) => {
      let valA = 0.0;
      let valB = 0.0;
      if (a.subsets && a.subsets[targetCol]) valA = parseFloat(a.subsets[targetCol].elo) || 0.0;
      else if (a[targetCol] !== undefined) valA = parseFloat(a[targetCol]) || 0.0;

      if (b.subsets && b.subsets[targetCol]) valB = parseFloat(b.subsets[targetCol].elo) || 0.0;
      else if (b[targetCol] !== undefined) valB = parseFloat(b[targetCol]) || 0.0;

      return valB - valA;
    });
    renderLeaderboardTableHeader();
    renderLeaderboardTable(cachedLeaderboardList);
  }

  loadLeaderboardData();
}

function toggleCandidateComparisonOnly() {
  isCandidateComparisonOnly = !isCandidateComparisonOnly;
  lbCurrentPage = 1;
  const btn = document.getElementById("btnToggleCandidateComparison");
  const txt = document.getElementById("btnCandidateComparisonText");

  if (isCandidateComparisonOnly) {
    btn.style.background = "#15803d";
    btn.style.color = "#ffffff";
    txt.textContent = "✓ 正在横向对比当前路由候选模型 (点击恢复全量)";
  } else {
    btn.style.background = "#f0fdf4";
    btn.style.color = "#166534";
    txt.textContent = "仅横向对比当前路由候选模型";
  }
  loadLeaderboardData();
}

function onLeaderboardSearchInput() {
  clearTimeout(lbSearchTimer);
  lbSearchTimer = setTimeout(() => {
    lbCurrentPage = 1;
    loadLeaderboardData();
  }, 250);
}

function onLeaderboardFilterChange() {
  lbCurrentPage = 1;
  loadLeaderboardData();
}

async function triggerLeaderboardSync() {
  const btnText = document.getElementById("btnSyncLbText");
  const origText = btnText.textContent;
  btnText.textContent = "正在拉取...";
  showToast("正在连接 Hugging Face 获取最新权威榜单...", "info");

  try {
    const res = await fetch("/api/leaderboard/sync", { method: "POST" });
    const data = await res.json();
    if (res.ok && data.status === "ok") {
      showToast(data.message || "榜单已更新至最新！", "success");
      loadLeaderboardData();
    } else {
      showToast(`同步失败: ${data.message || '未知错误'}`, "danger");
    }
  } catch (err) {
    showToast(`同步请求异常: ${err.message}`, "danger");
  } finally {
    btnText.textContent = origText;
  }
}

function quickAdoptLeaderboardModel(modelName) {
  openAddModelModal();
  switchTab("models");
  const nameInput = document.getElementById("modelNameInput");
  const customBox = document.getElementById("customModelInputBox");
  const customInput = document.getElementById("modelCustomUpstreamInput");
  const upstreamSelect = document.getElementById("modelUpstreamSelect");

  upstreamSelect.value = "__custom__";
  customBox.style.display = "block";
  customInput.value = modelName;
  nameInput.value = modelName;

  switchModelModalTab("capabilities");
  matchAndRenderArenaCard(modelName).then(() => {
    applyArenaScoresToInputs();
  });
}

async function applyLeaderboardToModel(modelName, targetLbModel) {
  try {
    showToast(`正在将权威基准同步至模型 [${modelName}]...`, "info");
    const res = await fetch("/api/leaderboard/apply_to_model", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ model_name: modelName, target_leaderboard_model: targetLbModel || null }),
    });
    const data = await res.json();
    if (res.ok && data.status === "ok") {
      showToast(data.message || "模型能力已同步权威基准！", "success");
      loadLeaderboardData();
      if (typeof loadConfig === "function") loadConfig();
    } else {
      showToast(`同步失败: ${data.detail || data.message || "未知错误"}`, "danger");
    }
  } catch (err) {
    showToast(`请求异常: ${err.message}`, "danger");
  }
}

async function applyAllLeaderboardRatings() {
  const btn = document.getElementById("applyAllLeaderboardBtn");
  const origText = btn ? btn.innerHTML : "";
  if (btn) btn.innerHTML = "<span>⏳ 同步中...</span>";

  try {
    showToast("正在一键将权威评测基准同步至所有候选模型...", "info");
    const res = await fetch("/api/leaderboard/apply_all_candidates", { method: "POST" });
    const data = await res.json();
    if (res.ok && data.status === "ok") {
      showToast(data.message || "所有模型已成功同步权威评测能力！", "success");
      loadLeaderboardData();
      if (typeof loadConfig === "function") loadConfig();
    } else {
      showToast(`同步失败: ${data.detail || data.message || "未知错误"}`, "danger");
    }
  } catch (err) {
    showToast(`请求异常: ${err.message}`, "danger");
  } finally {
    if (btn) btn.innerHTML = origText;
  }
}

