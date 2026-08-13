"use strict";

const state = {
  sessionId: null,
  ws: null,
  map: null,
  amapReady: false,
  overlays: [],
  handoff: null,
  editedHandoff: null,
  handoffConfirmed: false,
};

const els = {
  messages: document.getElementById("messages"),
  cards: document.getElementById("cards"),
  input: document.getElementById("input"),
  send: document.getElementById("send"),
  agentMode: document.getElementById("agent-mode"),
  amapMode: document.getElementById("amap-mode"),
  map: document.getElementById("map"),
};

const CATEGORY_LABEL = {
  scenic: "风景",
  culture: "文化",
  museum: "博物馆",
  food: "美食",
  shopping: "购物",
};

init();

async function init() {
  let config = {};
  try {
    config = await (await fetch("/config")).json();
  } catch (e) {
    /* ignore */
  }
  els.agentMode.textContent = config.real_agent_enabled
    ? "真实 ReAct Agent"
    : "离线兜底模式";
  els.agentMode.className = config.real_agent_enabled ? "pill pill-ok" : "pill pill-warn";

  if (config.amap_js_key) {
    loadAmap(config.amap_js_key, config.amap_js_security_key);
  } else {
    els.map.textContent = "未配置高德 JS key（config.toml [amap].js_key），地图不可用";
  }

  connect();
  bindUI();
}

function bindUI() {
  // 中文/日文等 IME：组合输入期间回车用于上屏，不能当作发送
  let composing = false;
  els.input.addEventListener("compositionstart", () => {
    composing = true;
  });
  els.input.addEventListener("compositionend", () => {
    composing = false;
  });

  els.send.addEventListener("click", sendMessage);
  els.input.addEventListener("keydown", (e) => {
    if (e.key !== "Enter" || e.shiftKey) return;
    if (e.isComposing || composing || e.keyCode === 229) return;
    e.preventDefault();
    sendMessage();
  });
  document.querySelectorAll(".example").forEach((btn) => {
    btn.addEventListener("click", () => {
      els.input.value = btn.textContent;
      sendMessage();
    });
  });
}

function connect() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  state.ws = new WebSocket(`${proto}://${location.host}/chat`);
  state.ws.onmessage = (evt) => handleEvent(JSON.parse(evt.data));
  state.ws.onclose = () => addStatus("连接已断开，刷新页面重试。");
}

function sendMessage() {
  const text = els.input.value.trim();
  if (!text || !state.ws || state.ws.readyState !== WebSocket.OPEN) return;
  addMessage("user", text);
  els.input.value = "";
  els.send.disabled = true;
  state.ws.send(JSON.stringify({ message: text, session_id: state.sessionId }));
}

function handleEvent(event) {
  switch (event.type) {
    case "session":
      state.sessionId = event.session_id;
      break;
    case "status":
      addStatus(event.message);
      break;
    case "trace":
      renderTrace(event);
      break;
    case "a2ui":
      renderCard(event.card);
      break;
    case "map":
      renderMap(event.map);
      break;
    case "text":
      clearStatus();
      addMessage("assistant", event.message);
      break;
    case "error":
      clearStatus();
      addMessage("assistant", "⚠️ " + event.message);
      els.send.disabled = false;
      break;
    case "done":
      els.send.disabled = false;
      break;
  }
}

function escapeHtml(text) {
  return text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

function inlineMarkdown(text) {
  let html = escapeHtml(text);
  // 先处理加粗，避免单星号正则破坏 ** 语法
  html = html.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/(?<!\*)\*([^*]+)\*(?!\*)/g, "<em>$1</em>");
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  return html;
}

/** 轻量 Markdown 渲染（仅助手消息；不执行 HTML，防 XSS） */
function renderMarkdown(text) {
  const lines = text.split("\n");
  const chunks = [];
  let inList = false;

  const closeList = () => {
    if (inList) {
      chunks.push("</ul>");
      inList = false;
    }
  };

  for (const raw of lines) {
    const line = raw.trimEnd();
    const trimmed = line.trim();

    if (!trimmed) {
      closeList();
      continue;
    }

    if (/^#{1,3}\s+/.test(trimmed)) {
      closeList();
      const level = trimmed.match(/^#+/)[0].length;
      const content = trimmed.replace(/^#+\s+/, "");
      const tag = level <= 2 ? "h3" : "h4";
      chunks.push(`<${tag}>${inlineMarkdown(content)}</${tag}>`);
      continue;
    }

    if (trimmed.startsWith("> ")) {
      closeList();
      chunks.push(
        `<blockquote>${inlineMarkdown(trimmed.slice(2))}</blockquote>`
      );
      continue;
    }

    if (/^[-*•·]\s*/.test(trimmed) || /^✅\s*/.test(trimmed)) {
      if (!inList) {
        chunks.push('<ul class="md-list">');
        inList = true;
      }
      const item = trimmed
        .replace(/^[-*•·]\s*/, "")
        .replace(/^✅\s*/, "✅ ");
      chunks.push(`<li>${inlineMarkdown(item)}</li>`);
      continue;
    }

    closeList();
    chunks.push(`<p>${inlineMarkdown(trimmed)}</p>`);
  }

  closeList();
  return chunks.join("") || `<p>${inlineMarkdown(text)}</p>`;
}

function addMessage(role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  if (role === "assistant") {
    const body = document.createElement("div");
    body.className = "md-body";
    body.innerHTML = renderMarkdown(text);
    div.appendChild(body);
  } else {
    div.textContent = text;
  }
  els.messages.appendChild(div);
  scrollMessages();
}

let statusEl = null;
function addStatus(text) {
  clearStatus();
  statusEl = document.createElement("div");
  statusEl.className = "msg status";
  statusEl.textContent = "⏳ " + text;
  els.messages.appendChild(statusEl);
  scrollMessages();
}
function clearStatus() {
  if (statusEl) {
    statusEl.remove();
    statusEl = null;
  }
}

function renderTrace(event) {
  els.cards.innerHTML = ""; // 新一轮回复开始，清空旧卡片
  const wrap = document.createElement("div");
  wrap.className = "trace";
  const mode = document.createElement("span");
  const modeInfo = traceMode(event);
  mode.className = "tag " + modeInfo.className;
  mode.textContent = modeInfo.text;
  wrap.appendChild(mode);
  (event.tool_trace || []).forEach((name) => {
    const tag = document.createElement("span");
    tag.className = "tag";
    tag.textContent = name;
    wrap.appendChild(tag);
  });
  els.messages.appendChild(wrap);
  scrollMessages();
}

function traceMode(event) {
  const tools = event.tool_trace || [];
  if (event.used_real_agent) {
    return { className: "real", text: "ReAct 自主调用" };
  }
  if (event.clarification) {
    return { className: "fallback", text: "信息补全/追问" };
  }
  if (tools.includes("plan_and_critique")) {
    return { className: "fallback", text: "离线规划兜底" };
  }
  return { className: "fallback", text: "本地规则处理" };
}

function scrollMessages() {
  els.messages.scrollTop = els.messages.scrollHeight;
}

// ---- cards ----
function renderCard(card) {
  const el = document.createElement("div");
  el.className = "card";
  if (card.type === "weather") {
    el.classList.add("weather-card");
    el.innerHTML = `<span style="font-size:22px">${card.rainy ? "🌧️" : "⛅"}</span>
      <div><strong>${card.city || ""} ${card.condition || ""}</strong>
      <div class="meta">约 ${card.temperature_c}°C · 来源 ${card.source}${
      card.rainy ? " · 注意雨天优先室内" : ""
    }</div></div>`;
  } else if (card.type === "summary") {
    el.innerHTML = `<h3>${card.summary || ""}</h3>
      <div class="meta">${card.city || ""} · 共 ${card.day_count} 天</div>`;
  } else if (card.type === "critic") {
    const badge = card.passed
      ? '<span class="badge badge-ok">critic 通过</span>'
      : '<span class="badge badge-warn">含可解释告警</span>';
    let html = `<h3>约束检查 ${badge}</h3>
      <div class="meta">违规项 ${card.original_issue_count} → ${card.final_issue_count} · 闭环 ${card.iterations} 轮</div>`;
    (card.revision_notes || []).forEach((n) => {
      html += `<div class="revision">↻ ${n}</div>`;
    });
    (card.issues || []).forEach((i) => {
      html += `<div class="meta">· [${i.severity}] ${i.message}</div>`;
    });
    el.innerHTML = html;
  } else if (card.type === "day") {
    let html = `<h3>第 ${card.day_index} 天 · ${card.theme || ""}</h3>`;
    (card.stops || []).forEach((s) => {
      const cat = CATEGORY_LABEL[s.category] || s.category || "";
      const route = s.route_from_previous
        ? `<div class="route">↳ 通勤约 ${s.route_from_previous.duration_min} 分钟 / ${s.route_from_previous.distance_km} 公里</div>`
        : "";
      html += `<div class="stop">${route}
        <div><span class="time">${s.start_time}</span> <span class="name">${s.name}</span>
        <span class="meta">· ${cat}${s.indoor ? " · 室内" : ""} · ${s.duration_min}分钟</span></div>
        <div class="note">${s.note || ""}</div></div>`;
    });
    el.innerHTML = html;
  } else if (card.type === "restaurants") {
    let html = `<h3>餐厅候选</h3>
      <div class="meta">${escapeHtml(card.city || "")}${
        card.cuisine ? " · " + escapeHtml(card.cuisine) : ""
      }${card.area ? " · " + escapeHtml(card.area) : ""}</div>`;
    (card.items || []).forEach((item) => {
      const tags = (item.tags || []).filter(Boolean).slice(0, 3).join(" / ");
      html += `<div class="compact-item">
        <div><strong>${escapeHtml(item.name || "")}</strong>
        <span class="meta"> · ${escapeHtml(item.price_level || "")}${
        item.rating ? " · " + escapeHtml(String(item.rating)) + "分" : ""
      }</span></div>
        <div class="meta">${escapeHtml(tags || item.source || "")}</div>
      </div>`;
    });
    el.innerHTML = html;
  } else if (card.type === "hotels") {
    let html = `<h3>住宿候选</h3>
      <div class="meta">${escapeHtml(card.city || "")} · ${escapeHtml(card.area || "")} · ${escapeHtml(card.budget_level || "")}</div>`;
    (card.items || []).forEach((item) => {
      html += `<div class="compact-item">
        <div><strong>${escapeHtml(item.name || "")}</strong>
        <span class="meta">${
          item.rating ? " · " + escapeHtml(String(item.rating)) + "分" : ""
        }</span></div>
        <div class="meta">约 ${escapeHtml(String(item.price_per_night || ""))} 元/晚 · ${escapeHtml(item.source || "")}</div>
      </div>`;
    });
    el.innerHTML = html;
  } else if (card.type === "budget") {
    let html = `<h3>预算估算</h3>
      <div class="budget-total">${escapeHtml(String(card.total_low || ""))} - ${escapeHtml(String(card.total_high || ""))} 元</div>
      <div class="meta">${escapeHtml(card.city || "")} · ${escapeHtml(String(card.days || ""))} 天 · ${escapeHtml(String(card.companions || ""))} 人</div>
      <div class="budget-grid">`;
    Object.entries(card.breakdown || {}).forEach(([label, value]) => {
      html += `<div><span>${escapeHtml(label)}</span><strong>${escapeHtml(String(value || 0))}</strong></div>`;
    });
    html += "</div>";
    el.innerHTML = html;
  } else if (card.type === "handoff") {
    state.handoff = cloneRoadbook(card.roadbook || null);
    state.editedHandoff = cloneRoadbook(card.roadbook || null);
    state.handoffConfirmed = false;
    el.classList.add("handoff-card");
    el.innerHTML = `<h3>${escapeHtml(card.title || "人工确认后导出")}</h3>
      <div class="meta">${escapeHtml(card.summary || "")}</div>
      ${renderHandoffEditor(state.editedHandoff)}
      <div class="handoff-actions">
        <button class="handoff-confirm" type="button">确认修改</button>
        <button class="handoff-regenerate" type="button">按修改重新生成</button>
        <button class="handoff-export" type="button" disabled>导出高德路书草稿</button>
      </div>
      <div class="handoff-status meta">请先确认路线；不会自动下单、不会自动支付，导出动作由用户确认触发。</div>`;
    bindHandoffEditor(el);
  }
  els.cards.appendChild(el);
}

function cloneRoadbook(roadbook) {
  if (!roadbook) return null;
  return JSON.parse(JSON.stringify(roadbook));
}

function renderHandoffEditor(roadbook) {
  if (!roadbook || !Array.isArray(roadbook.days)) {
    return '<div class="meta">暂无可导出的路书草稿。</div>';
  }
  let html = '<div class="roadbook-editor">';
  roadbook.days.forEach((day, dayIndex) => {
    html += `<div class="roadbook-day" data-day-index="${dayIndex}">
      <div class="roadbook-day-title">第 ${escapeHtml(String(day.day || dayIndex + 1))} 天</div>`;
    (day.stops || []).forEach((stop, stopIndex) => {
      html += `<div class="roadbook-stop" data-stop-index="${stopIndex}">
        <input class="roadbook-time" type="time" value="${escapeHtml(normalizeTime(stop.start_time))}" aria-label="开始时间" />
        <input class="roadbook-name" type="text" value="${escapeHtml(stop.name || "")}" aria-label="POI 名称" />
      </div>`;
    });
    html += "</div>";
  });
  html += "</div>";
  return html;
}

function normalizeTime(value) {
  const text = String(value || "");
  const match = text.match(/^(\d{1,2}):(\d{2})/);
  if (!match) return "";
  return `${match[1].padStart(2, "0")}:${match[2]}`;
}

function bindHandoffEditor(cardEl) {
  const confirm = cardEl.querySelector(".handoff-confirm");
  const regenerate = cardEl.querySelector(".handoff-regenerate");
  const exportBtn = cardEl.querySelector(".handoff-export");
  const status = cardEl.querySelector(".handoff-status");
  const inputs = cardEl.querySelectorAll(".roadbook-editor input");

  inputs.forEach((input) => {
    input.addEventListener("input", () => {
      state.handoffConfirmed = false;
      exportBtn.disabled = true;
      status.textContent = "有未确认修改；请确认后再导出。";
    });
  });

  confirm.addEventListener("click", () => {
    state.editedHandoff = collectEditedHandoff(cardEl);
    state.handoffConfirmed = true;
    exportBtn.disabled = false;
    status.textContent = "已确认修改，可以导出高德路书草稿。";
  });

  regenerate.addEventListener("click", () => {
    state.editedHandoff = collectEditedHandoff(cardEl);
    requestRegeneration(state.editedHandoff);
  });

  exportBtn.addEventListener("click", exportHandoff);
}

function collectEditedHandoff(cardEl) {
  const roadbook = cloneRoadbook(state.editedHandoff || state.handoff);
  if (!roadbook) return null;
  cardEl.querySelectorAll(".roadbook-day").forEach((dayEl) => {
    const dayIndex = Number(dayEl.dataset.dayIndex);
    const day = roadbook.days && roadbook.days[dayIndex];
    if (!day) return;
    dayEl.querySelectorAll(".roadbook-stop").forEach((stopEl) => {
      const stopIndex = Number(stopEl.dataset.stopIndex);
      const stop = day.stops && day.stops[stopIndex];
      if (!stop) return;
      stop.start_time = stopEl.querySelector(".roadbook-time").value || stop.start_time;
      stop.name = stopEl.querySelector(".roadbook-name").value.trim() || stop.name;
      stop.user_edited = true;
    });
  });
  roadbook.user_confirmed = state.handoffConfirmed;
  roadbook.edited_at = new Date().toISOString();
  return roadbook;
}

function exportHandoff() {
  if (!state.editedHandoff || !state.handoffConfirmed) return;
  state.editedHandoff.user_confirmed = true;
  const blob = new Blob([JSON.stringify(state.editedHandoff, null, 2)], {
    type: "application/json;charset=utf-8",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "amap-roadbook-draft.json";
  document.body.appendChild(link);
  link.click();
  link.remove();
  URL.revokeObjectURL(url);
}

function requestRegeneration(roadbook) {
  if (!roadbook || !state.ws || state.ws.readyState !== WebSocket.OPEN) return;
  const compact = (roadbook.days || [])
    .map((day) => {
      const stops = (day.stops || [])
        .map((stop) => `${stop.start_time || ""} ${stop.name || ""}`.trim())
        .filter(Boolean)
        .join("，");
      return `第${day.day}天：${stops}`;
    })
    .join("；");
  const message = `请基于我刚才人工修改后的路线重新生成并检查可行性：${compact}`;
  addMessage("user", message);
  els.send.disabled = true;
  state.ws.send(JSON.stringify({ message, session_id: state.sessionId }));
}


// ---- map ----
function loadAmap(key, securityKey) {
  if (securityKey) {
    window._AMapSecurityConfig = { securityJsCode: securityKey };
  }
  const script = document.createElement("script");
  script.src = `https://webapi.amap.com/maps?v=2.0&key=${key}`;
  script.onload = () => {
    state.amapReady = true;
    els.amapMode.textContent = "高德地图已加载";
    els.amapMode.className = "pill pill-ok";
    els.map.textContent = "";
    state.map = new AMap.Map(els.map, { zoom: 11, center: [116.397, 39.908] });
  };
  script.onerror = () => {
    els.map.textContent = "高德地图脚本加载失败（请检查 JS key / 域名白名单）";
  };
  document.head.appendChild(script);
}

const DAY_COLORS = ["#2563eb", "#16a34a", "#d97706", "#9333ea", "#dc2626"];

function renderMap(payload) {
  if (!state.amapReady || !state.map) return;
  state.overlays.forEach((o) => state.map.remove(o));
  state.overlays = [];

  (payload.markers || []).forEach((m) => {
    const marker = new AMap.Marker({
      position: [m.lng, m.lat],
      title: `${m.start_time || ""} ${m.name}`,
      label: { content: m.name, direction: "top" },
    });
    state.overlays.push(marker);
    state.map.add(marker);
  });

  (payload.routes || []).forEach((r, idx) => {
    const polyline = new AMap.Polyline({
      path: r.path,
      strokeColor: DAY_COLORS[(r.day - 1) % DAY_COLORS.length] || DAY_COLORS[idx % DAY_COLORS.length],
      strokeWeight: 5,
      strokeOpacity: 0.8,
    });
    state.overlays.push(polyline);
    state.map.add(polyline);
  });

  if (payload.center) state.map.setCenter(payload.center);
  if (state.overlays.length) state.map.setFitView(state.overlays);
}
