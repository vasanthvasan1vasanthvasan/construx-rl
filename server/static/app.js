const state = {
  sessionId: null,
  observation: null,
  running: false,
};

const els = {
  difficulty: document.querySelector("#difficulty"),
  resetBtn: document.querySelector("#resetBtn"),
  autoBtn: document.querySelector("#autoBtn"),
  commandInput: document.querySelector("#commandInput"),
  commandBtn: document.querySelector("#commandBtn"),
  commandHint: document.querySelector("#commandHint"),
  tabButtons: document.querySelectorAll(".tab-button"),
  tabPanels: document.querySelectorAll(".tab-panel"),
  dayStat: document.querySelector("#dayStat"),
  budgetStat: document.querySelector("#budgetStat"),
  scoreStat: document.querySelector("#scoreStat"),
  statusStat: document.querySelector("#statusStat"),
  taskCount: document.querySelector("#taskCount"),
  progressFill: document.querySelector("#progressFill"),
  taskList: document.querySelector("#taskList"),
  weatherList: document.querySelector("#weatherList"),
  permitList: document.querySelector("#permitList"),
  siteLog: document.querySelector("#siteLog"),
};

function activateTab(tabId) {
  els.tabButtons.forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tabId);
  });
  els.tabPanels.forEach((panel) => {
    panel.classList.toggle("active", panel.id === tabId);
  });
  if (window.location.hash !== `#${tabId}`) {
    history.replaceState(null, "", `#${tabId}`);
  }
}

function money(value) {
  return `$${Number(value || 0).toLocaleString()}`;
}

function score(value) {
  return Number(value || 0).toFixed(3);
}

async function postJson(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!response.ok) {
    throw new Error(await response.text());
  }
  return response.json();
}

async function resetEpisode() {
  els.statusStat.textContent = "Loading";
  const payload = {
    difficulty: els.difficulty.value,
    seed: 0,
    session_id: state.sessionId,
  };
  const result = await postJson("/reset", payload);
  state.sessionId = result.session_id;
  state.observation = result.observation;
  render();
}

async function step(action) {
  if (!state.sessionId) {
    await resetEpisode();
  }
  const result = await postJson("/step", {
    session_id: state.sessionId,
    action,
  });
  state.observation = result.observation;
  render();
  return result;
}

function firstOpenAlert(obs) {
  return obs.osha_alerts.find((alert) => !alert.incident_report_filed);
}

function firstBlockedMaterialNeed(obs) {
  for (const task of Object.values(obs.tasks)) {
    if (task.status === "done" || task.status === "in_progress") continue;
    for (const [material, qty] of Object.entries(task.required_materials || {})) {
      const pending = obs.pending_orders
        .filter((order) => order.material === material)
        .reduce((total, order) => total + order.quantity, 0);
      if ((obs.inventory[material] || 0) + pending < qty) {
        return { material, qty };
      }
    }
  }
  return null;
}

function firstAvailableTask(obs, crewType) {
  return Object.values(obs.tasks).find((task) => task.status === "available" && task.required_crew === crewType);
}

function heuristicAction(obs) {
  const alert = firstOpenAlert(obs);
  if (alert) {
    return {
      action_type: "file_incident_report",
      violation_code: alert.violation_code,
      crew_id: alert.crew_id,
      corrective_action: "Correct the cited hazard and retrain the crew before work resumes.",
    };
  }

  for (const [permitType, permit] of Object.entries(obs.permits)) {
    if (permit.status === "not_requested") {
      return { action_type: "request_permit", permit_type: permitType };
    }
  }

  const materialNeed = firstBlockedMaterialNeed(obs);
  if (materialNeed) {
    return {
      action_type: "order_material",
      material: materialNeed.material,
      quantity: materialNeed.qty,
      quality: "standard",
    };
  }

  for (const task of Object.values(obs.tasks)) {
    const needsInspection = task.status === "available"
      && task.osha_rules.length
      && !["office", "site"].includes(task.zone)
      && !Object.prototype.hasOwnProperty.call(obs.inspected_zones, task.zone);
    if (needsInspection) {
      return { action_type: "request_inspection", zone: task.zone };
    }
  }

  for (const task of Object.values(obs.tasks)) {
    if (task.status === "available" && task.required_crew === "subcontractor") {
      const quotes = Object.values(obs.subcontractor_quotes).filter((quote) => quote.task_id === task.task_id && quote.status === "open");
      if (!quotes.length) {
        return {
          action_type: "request_quote",
          subcontractor_id: task.task_id === "welding_stair_rails" ? "weldco" : "rapid_roof",
          task_id: task.task_id,
        };
      }
      const quote = quotes[0];
      if (quote.available_day <= obs.day) {
        return { action_type: "accept_quote", quote_id: quote.quote_id };
      }
      const crew = Object.values(obs.crews).find((item) => ["available", "held"].includes(item.status));
      if (crew) {
        return { action_type: "hold_crew", crew_id: crew.crew_id, reason: "Waiting for subcontractor availability." };
      }
    }
  }

  const today = obs.weather_forecast[0];
  if (today && (today.rain_probability >= 0.35 || today.wind_mph > 22)) {
    const checkedToday = obs.site_log.some((line) => line.includes(`Day ${obs.day}: weather checked`));
    if (!checkedToday) {
      return { action_type: "check_weather" };
    }
    const crew = Object.values(obs.crews).find((item) => ["available", "held"].includes(item.status));
    if (crew) {
      return { action_type: "hold_crew", crew_id: crew.crew_id, reason: "Waiting for safer weather." };
    }
  }

  for (const crew of Object.values(obs.crews)) {
    if (crew.status === "assigned" && crew.assigned_task) {
      return { action_type: "assign_crew", crew_id: crew.crew_id, task_id: crew.assigned_task };
    }
  }

  for (const crew of Object.values(obs.crews)) {
    const task = firstAvailableTask(obs, crew.crew_type);
    if (task) {
      return { action_type: "assign_crew", crew_id: crew.crew_id, task_id: task.task_id };
    }
  }

  const crew = Object.values(obs.crews).find((item) => item.status === "available");
  if (crew) {
    return {
      action_type: "hold_crew",
      crew_id: crew.crew_id,
      reason: "Waiting for dependency, permit, material, weather, or subcontractor availability.",
    };
  }

  return { action_type: "check_inventory" };
}

function parseTypedCommand(text, obs) {
  const raw = text.trim();
  if (!raw) {
    throw new Error("Type a command first.");
  }

  if (raw.startsWith("{")) {
    return JSON.parse(raw);
  }

  const lower = raw.toLowerCase();
  const tokens = lower.split(/[\s,]+/).filter(Boolean);

  if (lower.includes("weather")) {
    return { action_type: "check_weather" };
  }
  if (lower.includes("inventory") || lower.includes("materials")) {
    return { action_type: "check_inventory" };
  }

  if (lower.includes("permit")) {
    const permitType = Object.keys(obs.permits).find((name) => lower.includes(name)) || Object.keys(obs.permits)[0] || "building";
    if (lower.includes("status") || lower.includes("check")) {
      return { action_type: "check_permit_status", permit_type: permitType };
    }
    return { action_type: "request_permit", permit_type: permitType };
  }

  if (lower.includes("inspect") || lower.includes("inspection")) {
    const zone = tokens.find((token) => token.startsWith("zone_"))
      || Object.values(obs.tasks).find((task) => lower.includes(task.zone))?.zone
      || Object.values(obs.tasks).find((task) => task.status === "available" && task.zone !== "site" && task.zone !== "office")?.zone;
    if (!zone) throw new Error("Include a zone, for example: inspect zone_a.");
    return { action_type: "request_inspection", zone };
  }

  if (lower.includes("order")) {
    const materials = ["concrete", "steel", "lumber", "roofing", "wire", "pipe", "insulation", "drywall", "paint"];
    const material = materials.find((item) => lower.includes(item));
    const quantity = Number(tokens.find((token) => /^\d+$/.test(token))) || 1;
    if (!material) throw new Error("Include a material, for example: order 10 concrete.");
    return { action_type: "order_material", material, quantity, quality: "standard" };
  }

  if (lower.includes("assign")) {
    const crew = Object.values(obs.crews).find((item) => lower.includes(item.crew_id) || lower.includes(item.crew_type));
    const task = Object.values(obs.tasks).find((item) => lower.includes(item.task_id));
    if (!crew || !task) {
      throw new Error("Use: assign structural to site_survey.");
    }
    return { action_type: "assign_crew", crew_id: crew.crew_id, task_id: task.task_id };
  }

  if (lower.includes("hold") || lower.includes("wait")) {
    const crew = Object.values(obs.crews).find((item) => lower.includes(item.crew_id) || lower.includes(item.crew_type))
      || Object.values(obs.crews)[0];
    if (!crew) throw new Error("No crew available to hold.");
    return {
      action_type: "hold_crew",
      crew_id: crew.crew_id,
      reason: raw,
    };
  }

  if (lower.includes("quote")) {
    const task = Object.values(obs.tasks).find((item) => lower.includes(item.task_id) || item.required_crew === "subcontractor");
    if (!task) throw new Error("No subcontractor task is ready for a quote.");
    return {
      action_type: "request_quote",
      subcontractor_id: task.task_id === "welding_stair_rails" ? "weldco" : "rapid_roof",
      task_id: task.task_id,
    };
  }

  throw new Error("I could not map that command. Try request permit, order material, inspect zone, assign crew, hold crew, weather, or JSON.");
}

async function applyTypedCommand() {
  try {
    if (!state.observation) {
      await resetEpisode();
    }
    const action = parseTypedCommand(els.commandInput.value, state.observation);
    els.commandHint.textContent = `Applying ${action.action_type}...`;
    const result = await step(action);
    els.commandHint.textContent = result.info?.error ? result.info.error : `Applied ${action.action_type}`;
  } catch (error) {
    els.commandHint.textContent = error.message;
  }
}

async function runDemoPolicy() {
  if (state.running) return;
  state.running = true;
  els.autoBtn.textContent = "Running...";
  els.autoBtn.disabled = true;
  try {
    if (!state.observation || state.observation.done) {
      await resetEpisode();
    }
    for (let i = 0; i < 70; i += 1) {
      if (!state.observation || state.observation.done) break;
      const action = heuristicAction(state.observation);
      await step(action);
      await new Promise((resolve) => setTimeout(resolve, 140));
    }
  } finally {
    state.running = false;
    els.autoBtn.textContent = "Run Demo Policy";
    els.autoBtn.disabled = false;
  }
}

function renderTasks(obs) {
  const tasks = Object.values(obs.tasks);
  const done = tasks.filter((task) => task.status === "done").length;
  els.taskCount.textContent = `${done}/${tasks.length}`;
  els.progressFill.style.width = `${Math.round((done / Math.max(1, tasks.length)) * 100)}%`;
  els.taskList.innerHTML = tasks
    .map((task) => `<div class="task-pill ${task.status}"><strong>${task.task_id}</strong><br>${task.status}</div>`)
    .join("");
}

function renderWeather(obs) {
  els.weatherList.innerHTML = obs.weather_forecast
    .map((day) => `
      <div class="weather-item">
        <span>Day ${day.day}</span>
        <strong>${Math.round(day.rain_probability * 100)}% rain / ${day.wind_mph} mph</strong>
      </div>
    `)
    .join("");
}

function renderPermits(obs) {
  els.permitList.innerHTML = Object.entries(obs.permits)
    .map(([name, permit]) => `
      <div class="permit-item">
        <span>${name}</span>
        <strong>${permit.status}</strong>
      </div>
    `)
    .join("");
}

function renderLog(obs) {
  const logs = obs.site_log.slice(-9);
  els.siteLog.innerHTML = logs.map((line) => `<li>${line}</li>`).join("");
}

function render() {
  const obs = state.observation;
  if (!obs) return;
  els.dayStat.textContent = `${obs.day}/${obs.max_days}`;
  els.budgetStat.textContent = money(obs.remaining_budget);
  els.scoreStat.textContent = score(obs.current_score);
  els.statusStat.textContent = obs.done ? "Done" : "Active";
  renderTasks(obs);
  renderWeather(obs);
  renderPermits(obs);
  renderLog(obs);
}

els.resetBtn.addEventListener("click", () => {
  resetEpisode().catch((error) => {
    els.statusStat.textContent = "Error";
    console.error(error);
  });
});

els.autoBtn.addEventListener("click", () => {
  runDemoPolicy().catch((error) => {
    els.statusStat.textContent = "Error";
    console.error(error);
  });
});

els.commandBtn.addEventListener("click", () => {
  applyTypedCommand();
});

els.commandInput.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    applyTypedCommand();
  }
});

els.tabButtons.forEach((button) => {
  button.addEventListener("click", () => {
    activateTab(button.dataset.tab);
  });
});

const initialTab = window.location.hash?.replace("#", "");
if (initialTab && document.getElementById(initialTab)?.classList.contains("tab-panel")) {
  activateTab(initialTab);
}

resetEpisode().catch((error) => {
  els.statusStat.textContent = "Error";
  console.error(error);
});
