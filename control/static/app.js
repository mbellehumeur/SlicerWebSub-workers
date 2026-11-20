(() => {
  const COLOR_MODE_KEY = "websubWorkerControlColorMode";
  const productsEl = document.getElementById("products");
  const errorEl = document.getElementById("error");
  const hubDefaultEl = document.getElementById("hub-default");
  const refreshBtn = document.getElementById("refresh");
  const colorModeEl = document.getElementById("color-mode");
  const orchidContainersEl = document.getElementById("orchidContainers");
  const orchidPlatformSubEl = document.getElementById("orchidPlatformSub");

  /** @type {Map<string, EventSource>} */
  const streams = new Map();
  /** @type {Map<string, string>} */
  const hubChoices = new Map();

  function applyColorMode(mode) {
    const next = mode === "light" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", "3dslicer");
    document.documentElement.setAttribute("data-mode", next);
    if (colorModeEl) colorModeEl.value = next;
    try {
      localStorage.setItem(COLOR_MODE_KEY, next);
    } catch (_) {
      /* ignore */
    }
    setTimeout(() => {
      try {
        redrawAllMetricCharts();
      } catch (_) {
        /* metrics not ready yet */
      }
    }, 0);
  }

  function initColorMode() {
    let saved = null;
    try {
      saved = localStorage.getItem(COLOR_MODE_KEY);
    } catch (_) {
      /* ignore */
    }
    applyColorMode(saved === "light" || saved === "dark" ? saved : "dark");
    if (colorModeEl) {
      colorModeEl.addEventListener("change", () => {
        applyColorMode(colorModeEl.value);
      });
    }
  }

  function initSettingsMenu() {
    const toggle = document.getElementById("hubSettingsToggle");
    const panel = document.getElementById("hubSettingsPanel");
    if (!toggle || !panel) return;

    const closePanel = () => {
      panel.hidden = true;
      toggle.setAttribute("aria-expanded", "false");
    };
    const openPanel = () => {
      panel.hidden = false;
      toggle.setAttribute("aria-expanded", "true");
    };

    toggle.addEventListener("click", (e) => {
      e.stopPropagation();
      if (panel.hidden) openPanel();
      else closePanel();
    });
    document.addEventListener("click", (e) => {
      if (!panel.hidden && !panel.contains(e.target) && e.target !== toggle) {
        closePanel();
      }
    });
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") closePanel();
    });
  }

  function showError(msg) {
    if (!msg) {
      errorEl.hidden = true;
      errorEl.textContent = "";
      return;
    }
    errorEl.hidden = false;
    errorEl.textContent = msg;
  }

  async function api(path, options) {
    const res = await fetch(path, {
      headers: { Accept: "application/json", "Content-Type": "application/json" },
      ...options,
    });
    let data = null;
    try {
      data = await res.json();
    } catch (_) {
      data = null;
    }
    if (!res.ok) {
      const err = (data && data.error) || (data && data.status && data.status.error) || res.statusText;
      throw new Error(err || `HTTP ${res.status}`);
    }
    return data;
  }

  function escapeHtml(s) {
    return String(s)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  function statusBadge(product) {
    const st = product.status || {};
    if (!product.startable) {
      return '<span class="badge warn">unavailable</span>';
    }
    if (st.running) {
      return '<span class="badge running">running</span>';
    }
    return '<span class="badge stopped">stopped</span>';
  }

  function metaBadges(product) {
    const st = product.status || {};
    const badges = [];
    if (product.enabled === false) {
      badges.push('<span class="badge muted">disabled in catalog</span>');
    }
    if (st.running && st.hub) {
      badges.push(`<span class="badge muted">hub: ${escapeHtml(st.hub)}</span>`);
    }
    if (st.running && st.pid) {
      badges.push(`<span class="badge muted">pid ${st.pid}</span>`);
    }
    if (st.running && st.managed === false) {
      badges.push('<span class="badge muted">external</span>');
    }
    return badges.join("");
  }

  function ensureStream(productId, logEl) {
    if (streams.has(productId)) return;
    const es = new EventSource(`/api/products/${encodeURIComponent(productId)}/logs/stream`);
    es.addEventListener("log", (ev) => {
      try {
        const payload = JSON.parse(ev.data);
        if (payload.text) {
          logEl.textContent += payload.text;
          logEl.scrollTop = logEl.scrollHeight;
        }
      } catch (_) {
        /* ignore */
      }
    });
    es.onerror = () => {
      /* browser will retry */
    };
    streams.set(productId, es);
  }

  function closeStream(productId) {
    const es = streams.get(productId);
    if (es) {
      es.close();
      streams.delete(productId);
    }
  }

  function renderProduct(product) {
    const st = product.status || {};
    const hub =
      hubChoices.get(product.id) ||
      hubDefaultEl.value ||
      "cloud";
    hubChoices.set(product.id, hub);

    const row = document.createElement("article");
    row.className = "row";
    row.dataset.id = product.id;

    const subParts = [];
    if (product.entry_script) subParts.push(product.entry_script);
    if (st.error) subParts.push(st.error);
    const meta = metaBadges(product);

    row.innerHTML = `
      <div class="row-main">
        <div class="row-meta">
          <h2>${escapeHtml(product.title || product.id)}</h2>
          <p class="sub">${escapeHtml(subParts.join(" · ") || product.summary || "")}</p>
          ${meta ? `<div class="badges">${meta}</div>` : ""}
        </div>
        <div class="row-actions">
          ${statusBadge(product)}
          <select class="hub-select" data-role="hub" aria-label="Hub for ${escapeHtml(product.id)}" ${product.startable && !st.running ? "" : "disabled"}>
            <option value="local" ${hub === "local" ? "selected" : ""}>Local</option>
            <option value="cloud" ${hub === "cloud" ? "selected" : ""}>Cloud</option>
          </select>
          <button type="button" class="btn primary" data-role="start" ${product.startable && !st.running ? "" : "disabled"}>Start</button>
          <button type="button" class="btn" data-role="stop" ${st.running ? "" : "disabled"}>Stop</button>
          <button type="button" class="btn ghost small" data-role="toggle-log">Logs</button>
        </div>
      </div>
      <div class="log-panel" data-role="log-panel" hidden>
        <div class="log-toolbar">
          <span>worker.log</span>
          <div class="log-toolbar-actions">
            <button type="button" class="btn ghost small" data-role="copy-log">Copy</button>
            <button type="button" class="btn ghost small" data-role="clear-log">Clear log</button>
          </div>
        </div>
        <pre class="log-view" data-role="log-view"></pre>
      </div>
    `;

    const hubSelect = row.querySelector('[data-role="hub"]');
    hubSelect.addEventListener("change", () => {
      hubChoices.set(product.id, hubSelect.value);
    });

    row.querySelector('[data-role="start"]').addEventListener("click", async () => {
      showError("");
      try {
        await api(`/api/products/${encodeURIComponent(product.id)}/start`, {
          method: "POST",
          body: JSON.stringify({ hub: hubSelect.value || hubDefaultEl.value }),
        });
        await refresh();
      } catch (err) {
        showError(err.message || String(err));
        await refresh();
      }
    });

    row.querySelector('[data-role="stop"]').addEventListener("click", async () => {
      showError("");
      try {
        await api(`/api/products/${encodeURIComponent(product.id)}/stop`, {
          method: "POST",
          body: "{}",
        });
        await refresh();
      } catch (err) {
        showError(err.message || String(err));
        await refresh();
      }
    });

    const panel = row.querySelector('[data-role="log-panel"]');
    const logView = row.querySelector('[data-role="log-view"]');
    row.querySelector('[data-role="toggle-log"]').addEventListener("click", async () => {
      const open = panel.hidden;
      panel.hidden = !open;
      if (open) {
        try {
          const data = await api(
            `/api/products/${encodeURIComponent(product.id)}/logs?tail=200`
          );
          logView.textContent = data.text || "(no log yet)\n";
          logView.scrollTop = logView.scrollHeight;
        } catch (err) {
          logView.textContent = `(failed to load logs: ${err.message})\n`;
        }
        ensureStream(product.id, logView);
      } else {
        closeStream(product.id);
      }
    });

    row.querySelector('[data-role="clear-log"]').addEventListener("click", async () => {
      showError("");
      try {
        await api(`/api/products/${encodeURIComponent(product.id)}/logs/clear`, {
          method: "POST",
          body: "{}",
        });
        logView.textContent = "";
      } catch (err) {
        showError(err.message || String(err));
      }
    });

    const copyBtn = row.querySelector('[data-role="copy-log"]');
    copyBtn.addEventListener("click", async () => {
      showError("");
      const label = copyBtn.textContent;
      try {
        const data = await api(
          `/api/products/${encodeURIComponent(product.id)}/logs?tail=0`
        );
        await navigator.clipboard.writeText(data.text || "");
        copyBtn.textContent = "Copied";
      } catch (err) {
        showError(err.message || String(err));
        copyBtn.textContent = "Failed";
      }
      setTimeout(() => {
        copyBtn.textContent = label;
      }, 1200);
    });

    return row;
  }

  function orchidContainerBadge(container) {
    const state = String(container.state || "").toLowerCase();
    const health = String(container.health || "").toLowerCase();
    if (state === "running" || state === "up") {
      if (health === "unhealthy") {
        return '<span class="badge warn">unhealthy</span>';
      }
      if (health === "starting") {
        return '<span class="badge warn">starting</span>';
      }
      return '<span class="badge running">running</span>';
    }
    if (state === "exited" || state === "dead" || state === "stopped") {
      return '<span class="badge stopped">stopped</span>';
    }
    if (!state) {
      return '<span class="badge muted">unknown</span>';
    }
    return `<span class="badge warn">${escapeHtml(state)}</span>`;
  }

  function renderOrchidContainer(container) {
    const row = document.createElement("article");
    row.className = "row orchid-container-row";
    const service = container.service || container.name || "service";
    const subParts = [];
    if (container.name && container.name !== service) {
      subParts.push(container.name);
    }
    if (container.status) subParts.push(container.status);
    if (container.ports) subParts.push(container.ports);
    row.innerHTML = `
      <div class="row-main">
        <div class="row-meta">
          <h2>${escapeHtml(service)}</h2>
          <p class="sub">${escapeHtml(subParts.join(" · ") || "—")}</p>
        </div>
        <div class="row-actions">
          ${orchidContainerBadge(container)}
        </div>
      </div>
    `;
    return row;
  }

  async function refreshOrchid() {
    if (!orchidContainersEl || !orchidPlatformSubEl) return;
    try {
      const data = await api("/api/orchid");
      const apiUrl = data.api_url || "http://127.0.0.1:8000";
      const subBits = [];
      if (data.api_ok) {
        const aet = data.api && data.api.aet ? data.api.aet : "ORCHID";
        const dicomPort =
          data.api && data.api.dicom_port != null ? data.api.dicom_port : "11112";
        subBits.push(
          `API <a href="${escapeHtml(apiUrl)}" target="_blank" rel="noopener">${escapeHtml(apiUrl)}</a> ok`
        );
        subBits.push(`AE ${escapeHtml(String(aet))} · DICOM :${escapeHtml(String(dicomPort))}`);
      } else {
        const err = data.api_error || "unreachable";
        subBits.push(`API ${escapeHtml(apiUrl)} — ${escapeHtml(err)}`);
      }
      if (data.orchid_src) {
        subBits.push(`src ${escapeHtml(data.orchid_src)}`);
      }
      orchidPlatformSubEl.innerHTML = subBits.join(" · ");

      orchidContainersEl.innerHTML = "";
      if (data.error && !data.compose_ok) {
        orchidContainersEl.innerHTML = `<p class="empty">${escapeHtml(data.error)}</p>`;
        return;
      }
      const list = data.containers || [];
      if (!list.length) {
        orchidContainersEl.innerHTML =
          '<p class="empty">No Orchid containers running. Start with orchid/docker/up.sh or websub-orchid.service.</p>';
        return;
      }
      for (const container of list) {
        orchidContainersEl.appendChild(renderOrchidContainer(container));
      }
      if (data.error) {
        const note = document.createElement("p");
        note.className = "orchid-platform-note";
        note.textContent = data.error;
        orchidContainersEl.appendChild(note);
      }
    } catch (err) {
      orchidPlatformSubEl.textContent = err.message || String(err);
      orchidContainersEl.innerHTML =
        '<p class="empty">Could not load Orchid status.</p>';
    }
  }

  async function refresh() {
    showError("");
    try {
      const data = await api("/api/products");
      const openLogs = new Set(
        [...productsEl.querySelectorAll(".row")].filter((r) => {
          const panel = r.querySelector('[data-role="log-panel"]');
          return panel && !panel.hidden;
        }).map((r) => r.dataset.id)
      );

      // Drop streams so they rebind to newly rendered log views
      for (const id of [...streams.keys()]) {
        closeStream(id);
      }

      productsEl.innerHTML = "";
      const list = data.products || [];
      pollStats();
      if (!list.length) {
        productsEl.innerHTML = '<p class="empty">No products found.</p>';
      } else {
        for (const product of list) {
          const row = renderProduct(product);
          productsEl.appendChild(row);
          if (openLogs.has(product.id)) {
            const panel = row.querySelector('[data-role="log-panel"]');
            const logView = row.querySelector('[data-role="log-view"]');
            panel.hidden = false;
            try {
              const logData = await api(
                `/api/products/${encodeURIComponent(product.id)}/logs?tail=200`
              );
              logView.textContent = logData.text || "(no log yet)\n";
              logView.scrollTop = logView.scrollHeight;
            } catch (_) {
              /* keep empty */
            }
            ensureStream(product.id, logView);
          }
        }
      }
    } catch (err) {
      showError(err.message || String(err));
    }
    await refreshOrchid();
  }

  const METRICS_URL = "/api/metrics";
  const METRICS_POLL_MS = 2000;
  const METRICS_HISTORY_LEN = 60;
  const metricsHistory = {
    memory: [],
    cpu: [],
    network: [],
    gpu: [],
    disk: [],
  };
  let metricsPollTimer = null;

  function pushMetricSample(key, value) {
    const arr = metricsHistory[key];
    if (!arr) return;
    arr.push(value);
    if (arr.length > METRICS_HISTORY_LEN) arr.shift();
  }

  function drawSparkline(canvas, values) {
    if (!canvas || !values.length) return;
    const rect = canvas.getBoundingClientRect();
    const w = Math.max(1, Math.floor(rect.width));
    const h = Math.max(1, Math.floor(rect.height));
    const dpr = window.devicePixelRatio || 1;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    const style = getComputedStyle(document.documentElement);
    const lineColor = style.getPropertyValue("--accent").trim() || "#6eb5ff";
    const fillColor =
      style.getPropertyValue("--channel-ws-bg").trim() || "#1e3a52";
    let min = Math.min(...values);
    let max = Math.max(...values);
    if (min === max) {
      min -= 1;
      max += 1;
    }
    const pad = 4;
    const innerH = h - pad * 2;
    const innerW = w - pad * 2;
    const pts = values.map((v, i) => {
      const x =
        pad + (values.length === 1 ? 0 : (i / (values.length - 1)) * innerW);
      const y = pad + innerH - ((v - min) / (max - min)) * innerH;
      return [x, y];
    });
    ctx.beginPath();
    ctx.moveTo(pts[0][0], h - pad);
    pts.forEach((p) => ctx.lineTo(p[0], p[1]));
    ctx.lineTo(pts[pts.length - 1][0], h - pad);
    ctx.closePath();
    ctx.fillStyle = fillColor;
    ctx.globalAlpha = 0.45;
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.beginPath();
    pts.forEach((p, i) => {
      if (i === 0) ctx.moveTo(p[0], p[1]);
      else ctx.lineTo(p[0], p[1]);
    });
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 1.5;
    ctx.stroke();
  }

  function redrawAllMetricCharts() {
    drawSparkline(document.getElementById("metricMemoryChart"), metricsHistory.memory);
    drawSparkline(document.getElementById("metricCpuChart"), metricsHistory.cpu);
    drawSparkline(document.getElementById("metricNetworkChart"), metricsHistory.network);
    drawSparkline(document.getElementById("metricGpuChart"), metricsHistory.gpu);
    drawSparkline(document.getElementById("metricDiskChart"), metricsHistory.disk);
  }

  function formatBytesPerSec(n) {
    const v = Number(n) || 0;
    if (v < 1024) return `${v.toFixed(0)} B/s`;
    if (v < 1024 * 1024) return `${(v / 1024).toFixed(1)} KB/s`;
    if (v < 1024 * 1024 * 1024) return `${(v / (1024 * 1024)).toFixed(1)} MB/s`;
    return `${(v / (1024 * 1024 * 1024)).toFixed(2)} GB/s`;
  }

  function formatMb(n) {
    const v = Number(n);
    if (!Number.isFinite(v)) return "—";
    if (v >= 1024) return `${(v / 1024).toFixed(1)} GB`;
    return `${v.toFixed(0)} MB`;
  }

  function setText(id, text) {
    const el = document.getElementById(id);
    if (el) el.textContent = text;
  }

  function formatUptime(seconds) {
    const s = Math.max(0, Math.floor(Number(seconds) || 0));
    if (s < 60) return `${s}s`;
    const m = Math.floor(s / 60);
    if (m < 60) return `${m}m ${s % 60}s`;
    const h = Math.floor(m / 60);
    if (h < 24) return `${h}h ${m % 60}m`;
    const d = Math.floor(h / 24);
    return `${d}d ${h % 24}h`;
  }

  function updateRuntimeMeta(data) {
    const runtimeEl = document.getElementById("hubRuntimeMeta");
    if (!runtimeEl) return;
    const uptime = Number(data?.uptime_seconds);
    runtimeEl.textContent = Number.isFinite(uptime) ? formatUptime(uptime) : "";
    if (data?.started_at) {
      const d = new Date(data.started_at);
      if (!Number.isNaN(d.getTime())) {
        runtimeEl.title = `Started: ${d.toLocaleString()}`;
      }
    }
  }

  async function pollMetrics() {
    try {
      const data = await api(METRICS_URL);
      updateRuntimeMeta(data);
      const mem = data.memory || {};
      const memPct = Number(mem.percent);
      const memUsed = Number(mem.used_mb);
      const memTotal = Number(mem.total_mb);
      if (Number.isFinite(memPct)) {
        pushMetricSample("memory", memPct);
        setText(
          "metricMemoryValue",
          Number.isFinite(memUsed) ? formatMb(memUsed) : `${memPct.toFixed(1)}%`
        );
        setText(
          "metricMemorySub",
          Number.isFinite(memTotal)
            ? `${memPct.toFixed(1)}% of ${formatMb(memTotal)}`
            : `${memPct.toFixed(1)}%`
        );
      }

      const cpu = Number(data.cpu_percent);
      if (Number.isFinite(cpu)) {
        const cpuDisplay = Math.min(cpu, 100);
        pushMetricSample("cpu", cpuDisplay);
        setText("metricCpuValue", `${cpuDisplay.toFixed(1)}%`);
        setText("metricCpuSub", "");
      }

      const network = data.network || {};
      const chartValue = Number(network.chart_value);
      const rx = Number(network.rx_bytes_per_s);
      const tx = Number(network.tx_bytes_per_s);
      if (Number.isFinite(chartValue)) {
        pushMetricSample("network", chartValue);
        setText("metricNetworkValue", formatBytesPerSec(chartValue));
        setText(
          "metricNetworkSub",
          Number.isFinite(rx) && Number.isFinite(tx)
            ? `↓${formatBytesPerSec(rx)} ↑${formatBytesPerSec(tx)}`
            : ""
        );
        const card = document.getElementById("metricNetworkCard");
        if (card) {
          card.title = `Network throughput (RX+TX): ${formatBytesPerSec(chartValue)}`;
        }
      }

      const gpu = data.gpu || {};
      if (gpu.available) {
        const util = Number(gpu.utilization_percent);
        if (Number.isFinite(util)) {
          pushMetricSample("gpu", Math.min(util, 100));
          setText("metricGpuValue", `${util.toFixed(0)}%`);
        }
        const gmemUsed = Number(gpu.memory_used_mb);
        const gmemTotal = Number(gpu.memory_total_mb);
        const name = gpu.name || "GPU";
        const memPart =
          Number.isFinite(gmemUsed) && Number.isFinite(gmemTotal)
            ? `${formatMb(gmemUsed)}/${formatMb(gmemTotal)}`
            : "";
        setText(
          "metricGpuSub",
          memPart ? `${name} · ${memPart}` : name
        );
        const card = document.getElementById("metricGpuCard");
        if (card) {
          card.title = `${name}: util ${util}%` +
            (memPart ? `, VRAM ${memPart}` : "");
        }
      } else {
        setText("metricGpuValue", "n/a");
        setText("metricGpuSub", "No NVIDIA GPU");
      }

      const disk = data.disk || {};
      const primary = disk.primary || {};
      const diskPct = Number(disk.percent ?? primary.percent);
      if (Number.isFinite(diskPct)) {
        pushMetricSample("disk", Math.min(diskPct, 100));
        setText("metricDiskValue", `${diskPct.toFixed(0)}%`);
        const mount = primary.mount || primary.path || "/";
        const usedGb = Number(primary.used_gb);
        const totalGb = Number(primary.total_gb);
        setText(
          "metricDiskSub",
          Number.isFinite(usedGb) && Number.isFinite(totalGb)
            ? `${mount} · ${usedGb}G/${totalGb}G`
            : String(mount)
        );
        const card = document.getElementById("metricDiskCard");
        if (card) {
          const parts = (disk.partitions || [])
            .map(
              (p) =>
                `${p.mount || p.path}: ${p.percent}% (${p.used_gb}G/${p.total_gb}G)`
            )
            .join("\n");
          card.title = parts || `Disk ${diskPct}%`;
        }
      }

      redrawAllMetricCharts();
    } catch (err) {
      console.warn("Metrics poll failed:", err);
      updateRuntimeMeta(null);
      setText("metricMemoryValue", "—");
      setText("metricCpuValue", "—");
      setText("metricNetworkValue", "—");
      setText("metricGpuValue", "—");
      setText("metricDiskValue", "—");
    }
  }

  let statsPollTimer = null;

  function formatGb(bytes) {
    const n = Number(bytes) || 0;
    const gb = n / (1024 ** 3);
    if (gb >= 100) return gb.toFixed(0);
    if (gb >= 10) return gb.toFixed(1);
    if (gb >= 1) return gb.toFixed(2);
    if (n >= 1024 * 1024) return `${(n / (1024 * 1024)).toFixed(1)} MB`;
    if (n <= 0) return "0";
    return `${(n / (1024 * 1024)).toFixed(2)} MB`;
  }

  function formatComputeTime(seconds) {
    const s = Math.max(0, Math.floor(Number(seconds) || 0));
    if (s < 60) return `${s}s`;
    const m = Math.floor(s / 60);
    if (m < 60) return `${m}m ${s % 60}s`;
    const h = Math.floor(m / 60);
    if (h < 24) return `${h}h ${m % 60}m`;
    const d = Math.floor(h / 24);
    return `${d}d ${h % 24}h`;
  }

  function emptyStats() {
    return { jobs: 0, bytes_in: 0, bytes_out: 0, wall_seconds: 0 };
  }

  async function pollStats() {
    try {
      const data = await api("/api/stats");
      const totals = data.totals || emptyStats();
      setText("statJobs", String(totals.jobs || 0));
      setText("statGbIn", formatGb(totals.bytes_in));
      setText("statGbOut", formatGb(totals.bytes_out));
      setText("statCompute", formatComputeTime(totals.wall_seconds));
    } catch (err) {
      console.warn("Stats poll failed:", err);
      setText("statJobs", "—");
      setText("statGbIn", "—");
      setText("statGbOut", "—");
      setText("statCompute", "—");
    }
  }

  function startMetricsPolling() {
    if (metricsPollTimer) return;
    pollMetrics();
    metricsPollTimer = setInterval(pollMetrics, METRICS_POLL_MS);
  }

  function startStatsPolling() {
    if (statsPollTimer) return;
    pollStats();
    statsPollTimer = setInterval(pollStats, METRICS_POLL_MS);
  }

  hubDefaultEl.addEventListener("change", () => {
    for (const [id, val] of hubChoices) {
      // Only update rows that still match previous default pattern: leave explicit choices
      void id;
      void val;
    }
  });

  initColorMode();
  initSettingsMenu();
  startMetricsPolling();
  startStatsPolling();
  refreshBtn.addEventListener("click", () => refresh());
  refresh();
  setInterval(() => refresh(), 8000);
})();
