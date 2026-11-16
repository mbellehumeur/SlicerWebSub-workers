(() => {
  const productsEl = document.getElementById("products");
  const errorEl = document.getElementById("error");
  const hubDefaultEl = document.getElementById("hub-default");
  const refreshBtn = document.getElementById("refresh");

  /** @type {Map<string, EventSource>} */
  const streams = new Map();
  /** @type {Map<string, string>} */
  const hubChoices = new Map();

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

  function statusBadges(product) {
    const st = product.status || {};
    const badges = [];
    if (!product.startable) {
      badges.push('<span class="badge warn">unavailable</span>');
    } else if (st.running) {
      badges.push('<span class="badge running">running</span>');
    } else {
      badges.push('<span class="badge stopped">stopped</span>');
    }
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
    if (product.product) subParts.push(product.product);
    if (product.entry_script) subParts.push(product.entry_script);
    if (st.error) subParts.push(st.error);

    row.innerHTML = `
      <div class="row-main">
        <div class="row-meta">
          <h2>${escapeHtml(product.title || product.id)}</h2>
          <p class="sub">${escapeHtml(subParts.join(" · ") || product.summary || "")}</p>
          <div class="badges">${statusBadges(product)}</div>
        </div>
        <div class="row-actions">
          <select class="hub-select" data-role="hub" aria-label="Hub for ${escapeHtml(product.id)}" ${product.startable && !st.running ? "" : "disabled"}>
            <option value="local" ${hub === "local" ? "selected" : ""}>Local</option>
            <option value="cloud" ${hub === "cloud" ? "selected" : ""}>Cloud</option>
          </select>
          <button type="button" class="btn primary" data-role="start" ${product.startable && !st.running ? "" : "disabled"}>Start</button>
          <button type="button" class="btn danger" data-role="stop" ${st.running ? "" : "disabled"}>Stop</button>
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
      if (!list.length) {
        productsEl.innerHTML = '<p class="empty">No products found.</p>';
        return;
      }
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
    } catch (err) {
      showError(err.message || String(err));
    }
  }

  hubDefaultEl.addEventListener("change", () => {
    for (const [id, val] of hubChoices) {
      // Only update rows that still match previous default pattern: leave explicit choices
      void id;
      void val;
    }
  });

  refreshBtn.addEventListener("click", () => refresh());
  refresh();
  setInterval(() => refresh(), 8000);
})();
