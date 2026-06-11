"use strict";

const state = {
  sessionId: null,
  ws: null,
  map: null,
  amapReady: false,
  overlays: [],
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
  els.send.addEventListener("click", sendMessage);
  els.input.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
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

function addMessage(role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
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
  mode.className = "tag " + (event.used_real_agent ? "real" : "fallback");
  mode.textContent = event.used_real_agent ? "ReAct 自主调用" : "离线兜底";
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
  }
  els.cards.appendChild(el);
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
