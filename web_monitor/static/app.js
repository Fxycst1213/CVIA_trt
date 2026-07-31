const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
let config = null;
let savedConfig = null;
let toastTimer = null;
let statusTimer = null;
let previewTimer = null;
let poseTimer = null;
let isSaving = false;
let poseRequestPending = false;
let frameAvailability = {primary:false, secondary:false};
let previewLoadPending = false;
let previewLoadStartedAt = 0;
let previewRequestSequence = 0;
let previewAbortController = null;
let previewPairEtag = null;
let previewBitmaps = {primary:null, secondary:null};
let previewNaturalSize = {primary:{width:0,height:0}, secondary:{width:0,height:0}};
let latestPoseRow = null;
let latestPoseFileTime = 0;
let latestProjectionPoints = [];
let runtimeProjectionAvailable = false;
let projectionRequestPending = false;
let projectionEmptyMessage = "等待位姿";
let calibrationHistory = [];
let previousInferenceRunning = null;
let sessionModalDismissed = false;
let sessionDownloadPending = false;
let latestSessionStatus = null;
let mocapTargetApplying = false;
let discoveredBodiesSignature = "";

// 网页刷新周期：位姿页打开时曲线 500 ms；其他页面跟随 1000 ms 状态周期。
const POSE_REFRESH_MS = 200;
const STATUS_REFRESH_MS = 1000;
const PREVIEW_REQUEST_TIMEOUT_MS = 2000;

const poseAxes = [
  {key:"X", field:"x", group:"position"}, {key:"Y", field:"y", group:"position"}, {key:"Z", field:"z", group:"position"},
  {key:"Rx", field:"rx", group:"euler_deg"}, {key:"Ry", field:"ry", group:"euler_deg"}, {key:"Rz", field:"rz", group:"euler_deg"}
];
const projectionColors = ["#ef7b72", "#65d7e4", "#65d7e4", "#65d7e4", "#65d7e4", "#65d7e4", "#65d7e4"];

const cameraAcquisitionFields = [
  ["camera_id", "设备 ID", "number"], ["fps", "帧率 / FPS", "number"],
  ["resolution", "采集分辨率", "select"]
];
const cameraImageFields = [
  ["auto_exposure_mode", "自动曝光模式", "number"],
  ["exposure", "曝光值", "number"], ["white_balance_temperature", "白平衡色温", "number"],
  ["brightness", "亮度", "number"], ["contrast", "对比度", "number"], ["sharpness", "清晰度", "number"]
];

function getPath(object, path) { return path.split(".").reduce((value, key) => value[key], object); }
function setPath(object, path, value) { const keys = path.split("."); const last = keys.pop(); const parent = keys.reduce((item, key) => item[key], object); parent[last] = value; }
function clone(value) { return JSON.parse(JSON.stringify(value)); }
function showToast(message, error = false) { const toast = $("#toast"); toast.textContent = message; toast.classList.toggle("is-error", error); toast.classList.add("is-visible"); clearTimeout(toastTimer); toastTimer = setTimeout(() => toast.classList.remove("is-visible"), 3200); }

function parseMocapSelector(selector) {
  const text = String(selector || "").trim();
  if (text.startsWith("id:")) return {mode:"id", value:text.slice(3)};
  if (text.startsWith("name:")) return {mode:"name", value:text.slice(5)};
  return /^\d+$/.test(text) ? {mode:"id", value:text} : {mode:"name", value:text};
}

function selectedMocapMode() {
  return $('input[name="mocap-target-mode"]:checked')?.value || "name";
}

function updateMocapTargetEditor(updateConfig = true) {
  const mode = selectedMocapMode();
  const input = $("#mocapTargetValue");
  const value = input.value.trim();
  const isId = mode === "id";
  input.type = isId ? "number" : "text";
  if (isId) {
    input.min = "0"; input.step = "1"; input.removeAttribute("maxlength");
    input.placeholder = "0";
  } else {
    input.removeAttribute("min"); input.removeAttribute("step"); input.maxLength = 200;
    input.placeholder = "Tracker4";
  }
  $("#mocapTargetLabel").textContent = isId ? "刚体 SDK ID" : "刚体名称";
  $("#mocapTargetHint").textContent = isId
    ? "ID 以 SDK 的 DESC 输出为准，不等于名称末尾数字"
    : "名称按现场 SDK 的 DESC 输出填写，可包含空格";

  const valid = isId ? /^\d+$/.test(value) : Boolean(value);
  const selector = valid ? `${mode}:${value}` : "";
  $("#mocapSelectorPreview").textContent = selector || `${mode}:--`;
  $("#applyMocapTarget").disabled = !valid || !config?.mocap?.enabled || mocapTargetApplying;
  if (updateConfig && valid) {
    config.mocap.tracker = selector;
    updateDirty();
  }
  return {mode, value, valid, selector};
}

function bindMocapTargetEditor() {
  const target = parseMocapSelector(config?.mocap?.tracker);
  $$('input[name="mocap-target-mode"]').forEach(control => {
    control.checked = control.value === target.mode;
  });
  $("#mocapTargetValue").value = target.value;
  updateMocapTargetEditor(false);
}

function discoveredBodyForSelector(selector) {
  const target = parseMocapSelector(selector);
  return [...$("#mocapDiscoveredBodies").options].find(option =>
    option.dataset.id && (
      (target.mode === "id" && option.dataset.id === target.value) ||
      (target.mode === "name" && option.dataset.name === target.value)
    )
  );
}

function changeMocapTargetMode(mode) {
  const current = discoveredBodyForSelector(config?.mocap?.tracker);
  $$('input[name="mocap-target-mode"]').forEach(control => {
    control.checked = control.value === mode;
  });
  updateMocapTargetEditor(false);
  $("#mocapTargetValue").value = current
    ? (mode === "id" ? current.dataset.id : current.dataset.name)
    : "";
  updateMocapTargetEditor();
}

function selectDiscoveredMocapBody() {
  const option = $("#mocapDiscoveredBodies").selectedOptions[0];
  if (!option?.dataset.id) return;
  $("#mocapTargetValue").value = selectedMocapMode() === "id"
    ? option.dataset.id
    : option.dataset.name;
  updateMocapTargetEditor();
}

function updateDiscoveredMocapBodies(bodies) {
  if (!Array.isArray(bodies) || !bodies.length) return;
  const signature = JSON.stringify(bodies);
  if (signature === discoveredBodiesSignature) return;
  discoveredBodiesSignature = signature;
  const select = $("#mocapDiscoveredBodies");
  select.innerHTML = "";
  const placeholder = document.createElement("option");
  placeholder.value = "";
  placeholder.textContent = "选择 SDK 已发现的刚体";
  select.appendChild(placeholder);
  for (const body of bodies) {
    const option = document.createElement("option");
    option.value = String(body.id);
    option.dataset.id = String(body.id);
    option.dataset.name = String(body.name || "");
    option.textContent = `${body.name || "未命名刚体"} · SDK ID ${body.id}`;
    select.appendChild(option);
  }
  const current = discoveredBodyForSelector(config?.mocap?.tracker);
  select.value = current ? current.value : "";
}

async function applyMocapTarget() {
  const target = updateMocapTargetEditor();
  if (!target.valid || mocapTargetApplying) {
    if (!target.valid) showToast(target.mode === "id" ? "请输入非负整数刚体 ID" : "请输入刚体名称", true);
    return;
  }
  mocapTargetApplying = true;
  const button = $("#applyMocapTarget");
  button.disabled = true; button.textContent = "正在切换…";
  try {
    const response = await fetch("/api/mocap/target", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body:JSON.stringify({mode:target.mode, value:target.value}),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "目标刚体切换失败");
    config.mocap.tracker = result.tracker;
    savedConfig.mocap.tracker = result.tracker;
    updateDirty();
    updateMocapNetworkStatus(result.mocap);
    showToast(result.message);
    await refreshPose();
  } catch (error) {
    showToast(error.message, true);
  } finally {
    mocapTargetApplying = false;
    button.textContent = "应用目标并连接";
    updateMocapTargetEditor(false);
  }
}

function cameraFieldHtml(key, fields) {
  return fields.map(([name, text, type]) => {
    if (type === "select") return `<label class="field"><span>${text}</span><select data-path="${key}.${name}"><option value="HD1080">HD1080 · 1920×1080</option><option value="HD720">HD720 · 1280×720</option></select></label>`;
    return `<label class="field"><span>${text}</span><input type="number" step="any" data-path="${key}.${name}"></label>`;
  }).join("");
}

function cameraEditor(key, label, marker, open, configurable) {
  const acquisition = cameraFieldHtml(key, cameraAcquisitionFields);
  const imageControls = configurable
    ? `<div class="rule"></div><label class="switch-field switch-field--standalone"><input type="checkbox" data-path="${key}.apply_image_controls"><span class="switch"></span><span><b>应用红外相机成像配置</b><small>控制曝光、白平衡、亮度、对比度和清晰度</small></span></label><div class="field-row" data-image-controls="${key}">${cameraFieldHtml(key, cameraImageFields)}</div><label class="switch-field switch-field--standalone" data-image-controls="${key}"><input type="checkbox" data-path="${key}.apply_exposure"><span class="switch"></span><span><b>应用手动曝光</b><small>关闭时仅保留自动曝光模式</small></span></label><label class="switch-field switch-field--standalone" data-image-controls="${key}"><input type="checkbox" data-path="${key}.auto_white_balance"><span class="switch"></span><span><b>自动白平衡</b><small>由相机持续调整</small></span></label><label class="switch-field switch-field--standalone" data-image-controls="${key}"><input type="checkbox" data-path="${key}.apply_white_balance_temperature"><span class="switch"></span><span><b>应用手动色温</b><small>自动白平衡关闭时使用</small></span></label>`
    : `<div class="rule"></div><p class="muted">可见光相机仅设置采集格式、分辨率和帧率；曝光、白平衡及画质参数保持设备默认值。</p>`;
  return `<details class="camera-card" ${open ? "open" : ""}><summary><span><b>${marker}</b>${label}</span><small data-camera-summary="${key}"></small></summary><div class="camera-body"><div class="field-row">${acquisition}</div>${imageControls}</div></details>`;
}

function createMatrix(container, path, count) {
  const root = $(container); root.innerHTML = "";
  for (let index = 0; index < count; index++) {
    const input = document.createElement("input"); input.type = "number"; input.step = "any"; input.dataset.path = `${path}.${index}`; input.setAttribute("aria-label", `${path} 第 ${index + 1} 项`); root.appendChild(input);
  }
}

function createControls() {
  $("#cameraEditors").innerHTML = cameraEditor("detect_camera", "红外检测相机", "IR", true, true) + cameraEditor("photo_camera", "可见光相机", "RGB", false, false);
  createMatrix("#cameraMatrix", "calibration.camera_matrix", 9);
  createMatrix("#distortionMatrix", "calibration.distortion", 5);
  createMatrix("#extrinsicMatrix", "calibration.extrinsic", 16);
}

function bindConfig() {
  $$('[data-path]').forEach(control => {
    const path = control.dataset.path;
    const value = getPath(config, path);
    if (control.type === "radio") control.checked = control.value === value;
    else if (control.type === "checkbox") control.checked = Boolean(value);
    else control.value = value;
  });
  bindMocapTargetEditor(); updateDependentUI(); updateDirty();
}

function updateDependentUI() {
  const isFolder = config.input.mode === "folder";
  $("#folderOptions").style.opacity = isFolder ? "1" : ".46";
  $$('[data-path^="input.folder"], [data-path="input.interval_ms"], [data-path="input.loop"]').forEach(control => control.disabled = !isFolder);
  $("#sourceChip").textContent = `读取模式 · ${isFolder ? "文件夹" : "双相机"}`;
  const interval = Math.max(1, Number(config.input.interval_ms));
  $("#folderRate").textContent = `${interval} ms ≈ ${(1000 / interval).toFixed(1)} Hz；60 Hz 建议 17 ms`;
  const tcpEnabled = config.network.tcp_enabled;
  $("#tcpOptions").style.opacity = tcpEnabled ? "1" : ".46";
  $$('[data-path="network.tcp_ip"], [data-path="network.tcp_port"], [data-path="network.socket_mode"]').forEach(control => control.disabled = !tcpEnabled);
  const applyIrControls = Boolean(config.detect_camera.apply_image_controls);
  $$('[data-image-controls="detect_camera"]').forEach(group => {
    group.style.opacity = applyIrControls ? "1" : ".46";
    $$('[data-path]', group).forEach(control => control.disabled = !applyIrControls);
  });
  const mocapEnabled = Boolean(config.mocap?.enabled);
  $("#mocapOptions").style.opacity = mocapEnabled ? "1" : ".46";
  $$('[data-path^="mocap."]', $("#mocapOptions")).forEach(control => control.disabled = !mocapEnabled);
  $$('input[name="mocap-target-mode"], #mocapTargetValue, #mocapDiscoveredBodies').forEach(control => control.disabled = !mocapEnabled);
  updateMocapTargetEditor(false);
  $("#socketModeReadout").textContent = tcpEnabled ? ({0:"单图 + 位姿",1:"仅位姿",2:"双图 + 位姿"})[Number(config.network.socket_mode)] : "TCP 关闭 · 本地落盘";
  $("#resolutionReadout").textContent = `${config.runtime.frame_width} × ${config.runtime.frame_height}`;
  $$('[data-camera-summary]').forEach(item => { const camera = config[item.dataset.cameraSummary]; item.textContent = `ID ${camera.camera_id} · ${camera.fps} FPS`; });
}

function updateDirty() {
  const dirty = JSON.stringify(config) !== JSON.stringify(savedConfig);
  $("#saveButton").disabled = !dirty;
  $("#dirtyDot").classList.toggle("is-dirty", dirty);
  $("#saveTitle").textContent = dirty ? "有未保存的修改" : "配置已同步";
  $("#saveHint").textContent = dirty ? "保存后重启控制台生效" : "修改参数后在这里保存";
  $("#saveReadout").textContent = dirty ? "待保存" : "已同步";
}

async function loadConfig(announce = false) {
  try {
    const response = await fetch("/api/config", {cache:"no-store"});
    if (!response.ok) throw new Error("无法读取配置");
    config = await response.json(); savedConfig = clone(config); bindConfig();
    if (announce) showToast("已重新载入保存的配置");
  } catch (error) { showToast(error.message, true); }
}

async function saveConfig() {
  const button = $("#saveButton"); button.disabled = true; button.textContent = "保存中…";
  isSaving = true;
  clearTimeout(previewTimer);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 8000);
  try {
    const response = await fetch("/api/config", {method:"PUT", headers:{"Content-Type":"application/json"}, body:JSON.stringify(config), signal:controller.signal});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "保存失败");
    savedConfig = clone(config); updateDirty(); showToast(result.message); button.textContent = "已保存"; loadCalibrationHistory();
  } catch (error) { showToast(error.name === "AbortError" ? "保存超时：已暂停预览，请检查局域网连接后重试" : error.message, true); button.disabled = false; button.textContent = "保存配置"; }
  finally { clearTimeout(timeout); isSaving = false; schedulePreviewRefresh(); }
  setTimeout(() => { button.textContent = "保存配置"; updateDirty(); }, 1200);
}

function readControl(control) {
  if (control.type === "radio" && !control.checked) return;
  let value = control.type === "checkbox" ? control.checked : control.value;
  if (control.type === "number" || control.tagName === "SELECT" && /^\d+$/.test(value)) value = Number(value);
  setPath(config, control.dataset.path, value); updateDependentUI(); updateDirty();
}

function setFrame(name, state) {
  const canvas = $(`#${name}Image`), empty = $(`#${name}Empty`), meta = $(`#${name}Meta`);
  const live = state.available && state.age_ms < 3000;
  frameAvailability[name] = state.available;
  if (state.available) { empty.classList.add("is-hidden"); meta.textContent = `${Math.max(0, state.age_ms)} ms · ${Math.round(state.bytes / 1024)} KB`; }
  else {
    previewAbortController?.abort();
    previewAbortController = null;
    previewRequestSequence++;
    previewLoadPending = false;
    previewLoadStartedAt = 0;
    previewPairEtag = null;
    previewBitmaps[name]?.close?.();
    previewBitmaps[name] = null;
    previewNaturalSize[name] = {width:0, height:0};
    const context = canvas.getContext("2d");
    context.clearRect(0, 0, canvas.width, canvas.height);
    empty.classList.remove("is-hidden");
    meta.textContent = "等待帧";
  }
  return live;
}

function formatBytes(bytes) {
  if (!Number.isFinite(Number(bytes))) return "--";
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}

function suggestedSessionFilename() {
  const date = new Date();
  const part = value => String(value).padStart(2, "0");
  return `cvia_pnp_${date.getFullYear()}${part(date.getMonth() + 1)}${part(date.getDate())}_${part(date.getHours())}${part(date.getMinutes())}${part(date.getSeconds())}.csv`;
}

function openSessionModal(session) {
  if (sessionModalDismissed || !session?.available || !session?.current_run) return;
  $("#sessionFilename").textContent = suggestedSessionFilename();
  $("#sessionFileMeta").textContent = `${formatBytes(session.bytes)} · 临时会话已完成刷新`;
  $("#sessionModal").hidden = false;
}

function updateInferenceStatus(status) {
  const inference = status.inference || {running:false};
  const session = status.pnp_session || {available:false};
  const stopButton = $("#stopInferenceButton");
  latestSessionStatus = session;
  const canSave = !inference.running && session.available && session.current_run;
  stopButton.disabled = !inference.running && !canSave;
  stopButton.dataset.action = inference.running ? "stop" : "save";
  stopButton.textContent = inference.running ? "停止推理并保存" : canSave ? "保存本次 PnP" : "推理已停止";
  if (previousInferenceRunning === true && !inference.running) openSessionModal(session);
  else if (previousInferenceRunning === null && !inference.running) openSessionModal(session);
  previousInferenceRunning = inference.running;
}

async function stopInferenceOrSave() {
  const button = $("#stopInferenceButton");
  if (button.dataset.action === "save") {
    sessionModalDismissed = false;
    openSessionModal(latestSessionStatus);
    return;
  }
  button.disabled = true; button.textContent = "正在停止并刷新…";
  try {
    const response = await fetch("/api/inference/stop", {method:"POST", cache:"no-store"});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "停止推理失败");
    showToast(result.message);
    await refreshStatus();
  } catch (error) {
    showToast(error.message, true); button.disabled = false; button.textContent = "停止推理并保存";
  }
}

async function savePnpSession() {
  if (sessionDownloadPending) return;
  sessionDownloadPending = true;
  const button = $("#saveSessionButton");
  const filename = $("#sessionFilename").textContent || suggestedSessionFilename();
  button.disabled = true; button.textContent = "正在保存…";
  let fileHandle = null;
  try {
    if (window.isSecureContext && "showSaveFilePicker" in window) {
      fileHandle = await window.showSaveFilePicker({
        suggestedName: filename,
        types: [{description:"PnP CSV", accept:{"text/csv":[".csv"]}}],
      });
    }
    const response = await fetch("/api/pnp/download", {cache:"no-store"});
    if (!response.ok) {
      const result = await response.json();
      throw new Error(result.error || "PnP CSV 下载失败");
    }
    const blob = await response.blob();
    if (fileHandle) {
      const writable = await fileHandle.createWritable();
      await writable.write(blob); await writable.close();
      showToast("本次 PnP CSV 已保存到选择的位置");
    } else {
      const url = URL.createObjectURL(blob), link = document.createElement("a");
      link.href = url; link.download = filename; document.body.appendChild(link); link.click(); link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
      showToast("PnP CSV 已交给浏览器下载");
    }
    $("#sessionModal").hidden = true; sessionModalDismissed = true;
  } catch (error) {
    if (error.name !== "AbortError") showToast(error.message, true);
  } finally {
    sessionDownloadPending = false; button.disabled = false; button.textContent = "选择位置并保存";
  }
}

async function loadCalibrationHistory() {
  try {
    const response = await fetch("/api/calibration-history", {cache:"no-store"});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "标定历史读取失败");
    calibrationHistory = result.entries || [];
    const select = $("#calibrationHistorySelect");
    select.innerHTML = calibrationHistory.length ? calibrationHistory.map(entry => {
      const calibration = entry.calibration || {}, matrix = calibration.camera_matrix || [];
      const summary = matrix.length === 9 ? `fx ${Number(matrix[0]).toFixed(2)} · fy ${Number(matrix[4]).toFixed(2)}` : "标定参数";
      return `<option value="${entry.id}">${entry.saved_at} · ${summary}</option>`;
    }).join("") : '<option value="">暂无历史记录</option>';
    $("#loadCalibrationHistory").disabled = !calibrationHistory.length;
    $("#calibrationHistoryMeta").textContent = calibrationHistory.length ? `已缓存 ${calibrationHistory.length} 个去重版本，最新记录在最上方` : "保存标定参数后自动建立历史版本";
  } catch (error) {
    $("#calibrationHistoryMeta").textContent = "历史记录读取失败";
    showToast(error.message, true);
  }
}

function applyCalibrationHistory() {
  const id = $("#calibrationHistorySelect").value;
  const entry = calibrationHistory.find(item => item.id === id);
  if (!entry?.calibration) return;
  config.calibration = clone(entry.calibration);
  bindConfig();
  showToast(`已载入 ${entry.saved_at} 的标定参数；确认后请保存配置`);
}

async function refreshStatus() {
  try {
    const response = await fetch("/api/status", {cache:"no-store"}); if (!response.ok) throw new Error();
    const status = await response.json(); const liveA = setFrame("primary", status.frames.primary); const liveB = setFrame("secondary", status.frames.secondary);
    updateInferenceStatus(status);
    updateMocapNetworkStatus(status.mocap);
    $("#systemDot").classList.toggle("is-live", Boolean(status.inference?.running));
    $("#systemText").textContent = status.inference?.running ? (liveA || liveB ? "推理运行中 · 图像流在线" : "推理运行中 · 等待图像") : "推理已停止 · 等待保存";
  } catch { $("#systemDot").classList.remove("is-live"); $("#systemText").textContent = "网页服务连接中断"; }
}

function updateMocapNetworkStatus(mocap) {
  const target = $("#mocapNetworkState");
  if (!target) return;
  const labels = {
    disabled:"接收已关闭", starting:"正在启动", missing:"桥接程序缺失",
    connecting:"正在连接 SDK", ready:"SDK 已连接", waiting:"等待刚体",
    live:"数据在线", stale:"数据已暂停", invalid:"刚体解算无效",
    disconnected:"连接已断开", error:"启动失败", unavailable:"尚未初始化"
  };
  target.textContent = labels[mocap?.status] || labels[mocap?.connection] || "等待网页服务状态";
  target.classList.toggle("is-live", mocap?.status === "live");
  updateDiscoveredMocapBodies(mocap?.descriptions);
}

function scheduleStatusRefresh() {
  clearTimeout(statusTimer);
  statusTimer = setTimeout(async () => { await refreshStatus(); scheduleStatusRefresh(); }, STATUS_REFRESH_MS);
}

function requestPreviewPair(stamp) {
  if (!frameAvailability.primary || !frameAvailability.secondary) return false;
  const now = performance.now();
  if (previewLoadPending && now - previewLoadStartedAt < PREVIEW_REQUEST_TIMEOUT_MS) return false;
  previewAbortController?.abort();
  const controller = new AbortController();
  const sequence = ++previewRequestSequence;
  previewAbortController = controller;
  previewLoadPending = true;
  previewLoadStartedAt = now;
  loadPreviewPair(stamp, sequence, controller);
  return true;
}

function parsePreviewPair(payload) {
  const headerSize = 12;
  if (payload.byteLength < headerSize) throw new Error("双图帧包头不完整");
  const view = new DataView(payload);
  const magic = String.fromCharCode(
    view.getUint8(0), view.getUint8(1), view.getUint8(2), view.getUint8(3)
  );
  const primaryLength = view.getUint32(4, false);
  const secondaryLength = view.getUint32(8, false);
  if (magic !== "CVP1" || !primaryLength || !secondaryLength ||
      headerSize + primaryLength + secondaryLength !== payload.byteLength) {
    throw new Error("双图帧包格式无效");
  }
  return {
    primary:new Blob([payload.slice(headerSize, headerSize + primaryLength)], {type:"image/jpeg"}),
    secondary:new Blob([payload.slice(headerSize + primaryLength)], {type:"image/jpeg"})
  };
}

async function loadPreviewPair(stamp, sequence, controller) {
  let decoded = [];
  try {
    const headers = previewPairEtag ? {"If-None-Match":previewPairEtag} : {};
    const response = await fetch(`/preview/pair.bin?t=${stamp}`, {
      cache:"no-store",
      signal:controller.signal,
      headers
    });
    if (response.status === 304) return;
    if (!response.ok) throw new Error(`双图预览读取失败: ${response.status}`);
    const pair = parsePreviewPair(await response.arrayBuffer());
    const results = await Promise.allSettled([
      createImageBitmap(pair.primary),
      createImageBitmap(pair.secondary)
    ]);
    decoded = results
      .filter(result => result.status === "fulfilled")
      .map(result => result.value);
    if (results.some(result => result.status === "rejected")) {
      decoded.forEach(bitmap => bitmap.close?.());
      decoded = [];
      throw new Error("双图预览解码失败");
    }
    if (sequence !== previewRequestSequence) {
      decoded.forEach(bitmap => bitmap.close?.());
      decoded = [];
      return;
    }
    const [primaryBitmap, secondaryBitmap] = decoded;
    const previousPrimary = previewBitmaps.primary;
    const previousSecondary = previewBitmaps.secondary;
    previewBitmaps.primary = primaryBitmap;
    previewBitmaps.secondary = secondaryBitmap;
    previewNaturalSize.primary = {width:primaryBitmap.width, height:primaryBitmap.height};
    previewNaturalSize.secondary = {width:secondaryBitmap.width, height:secondaryBitmap.height};
    drawPreviewCanvas("primary");
    drawPreviewCanvas("secondary");
    previousPrimary?.close?.();
    previousSecondary?.close?.();
    decoded = [];
    previewPairEtag = response.headers.get("ETag") || previewPairEtag;
    renderReprojection();
  } catch (error) {
    decoded.forEach(bitmap => bitmap.close?.());
    if (error.name !== "AbortError") {
      // 必须等两张图都成功；瞬时网络或解码错误时整组保留上一帧。
    }
  } finally {
    if (sequence === previewRequestSequence) {
      previewLoadPending = false;
      previewLoadStartedAt = 0;
      previewAbortController = null;
    }
  }
}

function drawPreviewCanvas(name) {
  const canvas = $(`#${name}Image`);
  const bitmap = previewBitmaps[name];
  if (!canvas || !bitmap) return;
  const rect = canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(rect.width * ratio));
  const height = Math.max(1, Math.round(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const context = canvas.getContext("2d");
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.fillStyle = "#050a0c";
  context.fillRect(0, 0, rect.width, rect.height);
  const scale = Math.min(rect.width / bitmap.width, rect.height / bitmap.height);
  const drawWidth = bitmap.width * scale;
  const drawHeight = bitmap.height * scale;
  context.drawImage(
    bitmap,
    (rect.width - drawWidth) / 2,
    (rect.height - drawHeight) / 2,
    drawWidth,
    drawHeight
  );
}

function refreshPreviewImages() {
  if (isSaving || document.hidden) return;
  const stamp = Date.now();
  const pairRequested = requestPreviewPair(stamp);
  // 重投影只跟随成功发起的主图请求，避免在图像拥塞时继续制造 JSON 请求。
  if (pairRequested) refreshReprojection();
}

function schedulePreviewRefresh() {
  clearTimeout(previewTimer);
  const fps = Math.max(1, Math.min(30, Number(config?.runtime?.preview_fps || 10)));
  const interval = document.hidden ? STATUS_REFRESH_MS : Math.round(1000 / fps);
  previewTimer = setTimeout(() => { refreshPreviewImages(); schedulePreviewRefresh(); }, interval);
}

// Equivalent to cv2.projectPoints for one point with k1, k2, p1, p2, k3.
// CSV order is timestamp, x, y, z, rx, ry, rz; XYZ is the PnP translation in camera coordinates.
function projectPnpPoint(row) {
  if (!config || !row || row.length < 4) return null;
  const [x, y, z] = row.slice(1, 4).map(Number);
  if (![x, y, z].every(Number.isFinite) || z <= 0) return {valid:false, reason:"目标位于相机后方"};
  const k = config.calibration.camera_matrix.map(Number);
  const [k1, k2, p1, p2, k3] = config.calibration.distortion.map(Number);
  const xn = x / z, yn = y / z;
  const r2 = xn * xn + yn * yn, r4 = r2 * r2, r6 = r4 * r2;
  const radial = 1 + k1 * r2 + k2 * r4 + k3 * r6;
  const xd = xn * radial + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn);
  const yd = yn * radial + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn;
  const homogeneous = k[6] * xd + k[7] * yd + k[8];
  if (!Number.isFinite(homogeneous) || Math.abs(homogeneous) < 1e-12) return {valid:false, reason:"投影矩阵无效"};
  return {valid:true, u:(k[0] * xd + k[1] * yd + k[2]) / homogeneous, v:(k[3] * xd + k[4] * yd + k[5]) / homogeneous, x, y, z};
}

async function refreshReprojection() {
  if (projectionRequestPending) return;
  projectionRequestPending = true;
  try {
    const response = await fetch("/api/reprojection", {cache:"no-store"});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "重投影状态读取失败");
    runtimeProjectionAvailable = Boolean(result.available);
    if (!runtimeProjectionAvailable) latestProjectionPoints = [];
    if (runtimeProjectionAvailable) {
      latestPoseFileTime = result.mtime_ns ? result.mtime_ns / 1e6 : 0;
      if (result.valid && Array.isArray(result.pose) && result.pose.length === 6) {
        latestPoseRow = [Number(result.timestamp), ...result.pose.map(Number)];
        latestProjectionPoints = Array.isArray(result.points) ? result.points
          .filter(point => Array.isArray(point) && point.length === 2)
          .map(point => point.map(Number))
          .filter(point => point.every(Number.isFinite)) : [];
        projectionEmptyMessage = "等待位姿";
      } else {
        latestPoseRow = null;
        latestProjectionPoints = [];
        projectionEmptyMessage = "当前帧无有效 PnP";
      }
      renderReprojection();
    }
  } catch {
    // 新 C++ 版本尚未启动时继续使用 CSV 最新位姿作为兼容回退。
  } finally { projectionRequestPending = false; }
}

function renderReprojection() {
  const canvas = $("#reprojectionCanvas"), readout = $("#projectionReadout");
  if (!canvas || !readout) return;
  const rect = canvas.getBoundingClientRect(), ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(rect.width * ratio)), height = Math.max(1, Math.round(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  const ctx = canvas.getContext("2d"); ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, rect.width, rect.height);
  readout.classList.remove("is-outside", "is-stale");

  if (!latestPoseRow) { $("#projectionState").textContent = projectionEmptyMessage; $("#projectionPixel").textContent = "u -- · v --"; return; }
  const projectedOrigin = projectPnpPoint(latestPoseRow);
  if (!projectedOrigin?.valid) { $("#projectionState").textContent = projectedOrigin?.reason || "无法投影"; $("#projectionPixel").textContent = "u -- · v --"; readout.classList.add("is-outside"); return; }

  const sourceWidth = Math.max(1, Number(config.runtime.frame_width));
  const sourceHeight = Math.max(1, Number(config.runtime.frame_height));
  const imageWidth = previewNaturalSize.primary.width || sourceWidth;
  const imageHeight = previewNaturalSize.primary.height || sourceHeight;
  // 新版 C++ 直接发布与 solvePnP 同一组 _p3d/R/T 得到的 7 点；旧版回退为仅投影 P0。
  const sourcePoints = latestProjectionPoints.length === 7 ? latestProjectionPoints : [[projectedOrigin.u, projectedOrigin.v]];
  const displayPoints = sourcePoints.map(([sourceU, sourceV], index) => {
    const u = sourceU * imageWidth / sourceWidth, v = sourceV * imageHeight / sourceHeight;
    return {index, u, v, inside:u > 0 && u < imageWidth && v > 0 && v < imageHeight};
  });
  const origin = displayPoints[0], insideCount = displayPoints.filter(point => point.inside).length;
  const age = latestPoseFileTime ? Date.now() - latestPoseFileTime : 0;
  const stale = age > 2500;
  $("#projectionPixel").textContent = `P0 u ${origin.u.toFixed(1)} · v ${origin.v.toFixed(1)}`;
  $("#projectionState").textContent = stale ? "位姿已停止更新" : sourcePoints.length === 7 && insideCount === 7 ? "七点重投影有效" : `${insideCount}/${sourcePoints.length} 点在画面内`;
  readout.classList.toggle("is-outside", insideCount !== sourcePoints.length); readout.classList.toggle("is-stale", stale);
  if (!insideCount || stale || !previewBitmaps.primary) return;

  const scale = Math.min(rect.width / imageWidth, rect.height / imageHeight);
  const drawWidth = imageWidth * scale, drawHeight = imageHeight * scale;
  const offsetX = (rect.width - drawWidth) / 2, offsetY = (rect.height - drawHeight) / 2;
  for (const point of displayPoints) {
    if (!point.inside) continue;
    const px = offsetX + point.u * scale, py = offsetY + point.v * scale;
    const color = projectionColors[point.index] || "#e9f0f2";
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = point.index === 0 ? 1.5 : 1.2;
    ctx.beginPath(); ctx.arc(px, py, point.index === 0 ? 8 : 5, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.arc(px, py, point.index === 0 ? 3.2 : 2.4, 0, Math.PI * 2); ctx.fill();
    if (point.index === 0) {
      ctx.beginPath(); ctx.moveTo(px - 15, py); ctx.lineTo(px + 15, py); ctx.moveTo(px, py - 15); ctx.lineTo(px, py + 15); ctx.stroke();
    }
    const pointLabel = `P${point.index}`;
    const labelX = Math.min(rect.width - 18, Math.max(4, px + 7));
    const labelY = Math.min(rect.height - 7, Math.max(10, py - 7));
    ctx.font = "600 9px ui-monospace, monospace"; ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(5,10,12,.92)"; ctx.strokeText(pointLabel, labelX, labelY);
    ctx.fillStyle = color; ctx.fillText(pointLabel, labelX, labelY);
  }

  if (!origin.inside) return;
  const originPx = offsetX + origin.u * scale, originPy = offsetY + origin.v * scale;
  const label = `XYZ ${projectedOrigin.x.toFixed(1)}  ${projectedOrigin.y.toFixed(1)}  ${projectedOrigin.z.toFixed(1)}`;
  ctx.font = "10px ui-monospace, monospace"; const labelWidth = ctx.measureText(label).width;
  const labelX = Math.min(Math.max(8, originPx + 13), rect.width - labelWidth - 14), labelY = Math.max(17, originPy - 17);
  ctx.fillStyle = "rgba(5,10,12,.82)"; ctx.fillRect(labelX - 5, labelY - 11, labelWidth + 10, 16);
  ctx.fillStyle = "#f3aaa4"; ctx.fillText(label, labelX, labelY);
}

function formatPoseValue(value) {
  return Number.isFinite(Number(value)) ? Number(value).toFixed(3) : "--";
}

function setSourceState(dotSelector, live, warning = false) {
  const dot = $(dotSelector);
  dot.classList.toggle("is-live", live);
  dot.classList.toggle("is-warning", warning);
}

function renderPoseComparison(result) {
  const detection = result.detection || {};
  const mocap = result.mocap || {};
  const detectionPose = detection.pose || {};
  const mocapPose = mocap.pose || {};

  for (const axis of poseAxes) {
    $(`#detectValue${axis.key}`).textContent = formatPoseValue(detectionPose[axis.field]);
    const mocapGroup = mocapPose[axis.group] || {};
    $(`#mocapValue${axis.key}`).textContent = formatPoseValue(mocapGroup[axis.field]);
  }

  const detectionAge = Number(detection.age_ms);
  const detectionLive = Boolean(detection.available) && Number.isFinite(detectionAge) && detectionAge <= 2500;
  $("#detectionState").textContent = !detection.available ? "等待视觉位姿" : detectionLive ? "视觉检测在线" : "视觉位姿已暂停";
  $("#detectionMeta").textContent = detection.available
    ? `PnP ${detection.source === "reprojection" ? "当前帧" : "CSV 回退"} · ${Number.isFinite(detectionAge) ? `${detectionAge.toFixed(0)} ms 前更新` : "更新时间未知"}`
    : "PnP 尚未输出";
  setSourceState("#detectionDot", detectionLive, Boolean(detection.available) && !detectionLive);

  const mocapLabels = {
    disabled:"动捕接收已关闭", starting:"正在启动动捕", missing:"桥接程序缺失",
    connecting:"正在连接 SDK", ready:"SDK 已连接", waiting:"等待目标刚体",
    live:"NOKOV 动捕在线", stale:"动捕数据已暂停", invalid:"当前刚体解算无效",
    disconnected:"SDK 连接已断开", error:"动捕启动失败", unavailable:"动捕尚未初始化"
  };
  $("#mocapState").textContent = mocapLabels[mocap.status] || "等待动捕数据";
  const trackerName = mocapPose.tracker_name || mocap.selector || "RIGID BODY";
  $("#mocapTrackerHeading").textContent = String(trackerName).toUpperCase();
  $("#mocapMeta").textContent = mocapPose.frame !== undefined
    ? `帧 ${mocapPose.frame} · ${Number(mocap.age_ms).toFixed(0)} ms 前接收 · SDK ${mocap.sdk_version || "--"}`
    : (mocap.message || `${mocap.selector || "目标刚体"} · SDK ${mocap.sdk_version || "--"}`);
  setSourceState("#mocapDot", mocap.status === "live", ["stale", "invalid", "waiting", "ready"].includes(mocap.status));
  updateMocapNetworkStatus(mocap);

  const bothLive = detectionLive && mocap.status === "live";
  const eitherAvailable = Boolean(detection.available || mocapPose.frame !== undefined);
  $("#poseState").textContent = bothLive ? "双源在线" : eitherAvailable ? "单源 / 待同步" : "等待双源";
  $("#poseState").classList.toggle("is-live", bothLive);

  const delta = Number(result.receive_delta_ms);
  if (detectionLive && mocap.status === "live" && Number.isFinite(delta)) {
    const relation = delta >= 0 ? "视觉更新晚于动捕" : "视觉更新早于动捕";
    $("#comparisonDelta").textContent = `本机更新时差 · ${relation} ${Math.abs(delta).toFixed(1)} ms`;
  } else {
    $("#comparisonDelta").textContent = "双源同时在线后显示本机更新时差";
  }
}

async function refreshPose() {
  if (document.hidden) return;
  if (poseRequestPending) return;
  poseRequestPending = true;
  try {
    const response = await fetch("/api/pose-comparison", {cache:"no-store"});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "位姿读取失败");
    if (!runtimeProjectionAvailable && result.detection?.available) {
      const pose = result.detection.pose;
      latestPoseRow = [Number(result.detection.timestamp), pose.x, pose.y, pose.z, pose.rx, pose.ry, pose.rz];
      latestProjectionPoints = [];
      latestPoseFileTime = result.detection.updated_unix_ns ? result.detection.updated_unix_ns / 1e6 : 0;
      projectionEmptyMessage = "等待 CSV 位姿";
    }
    // C++ 已提供同帧七点投影时，图像预览循环负责绘制；这里只更新 CSV 回退投影。
    if (!runtimeProjectionAvailable) renderReprojection();
    renderPoseComparison(result);
  } catch (error) {
    $("#poseState").textContent = "读取失败"; $("#poseState").classList.remove("is-live");
  } finally {
    poseRequestPending = false;
  }
}

function schedulePoseRefresh() {
  clearTimeout(poseTimer);
  const active = $('.nav-tab[data-tab="pose"]')?.classList.contains("is-active");
  poseTimer = setTimeout(async () => {
    await refreshPose();
    schedulePoseRefresh();
  }, active ? POSE_REFRESH_MS : STATUS_REFRESH_MS);
}

document.addEventListener("DOMContentLoaded", async () => {
  createControls();
  $("#configForm").addEventListener("input", event => { if (event.target.dataset.path) readControl(event.target); });
  $("#configForm").addEventListener("change", event => { if (event.target.dataset.path) readControl(event.target); });
  $("#configForm").addEventListener("submit", event => { event.preventDefault(); saveConfig(); });
  $("#reloadButton").addEventListener("click", () => loadConfig(true));
  $("#resetCalibration").addEventListener("click", () => { config.calibration = clone(savedConfig.calibration); bindConfig(); showToast("标定值已恢复到当前保存状态"); });
  $("#loadCalibrationHistory").addEventListener("click", applyCalibrationHistory);
  $$('input[name="mocap-target-mode"]').forEach(control => control.addEventListener("change", () => changeMocapTargetMode(control.value)));
  $("#mocapTargetValue").addEventListener("input", updateMocapTargetEditor);
  $("#mocapDiscoveredBodies").addEventListener("change", selectDiscoveredMocapBody);
  $("#applyMocapTarget").addEventListener("click", applyMocapTarget);
  $("#stopInferenceButton").addEventListener("click", stopInferenceOrSave);
  $("#saveSessionButton").addEventListener("click", savePnpSession);
  $("#saveSessionLater").addEventListener("click", () => { $("#sessionModal").hidden = true; sessionModalDismissed = true; });
  $$(".nav-tab").forEach(tab => tab.addEventListener("click", () => { $$(".nav-tab").forEach(item => item.classList.toggle("is-active", item === tab)); $$(".form-section").forEach(panel => panel.classList.toggle("is-active", panel.dataset.panel === tab.dataset.tab)); if (tab.dataset.tab === "pose") requestAnimationFrame(refreshPose); }));
  window.addEventListener("beforeunload", event => { if (JSON.stringify(config) !== JSON.stringify(savedConfig)) { event.preventDefault(); event.returnValue = ""; } });
  document.addEventListener("visibilitychange", () => {
    schedulePreviewRefresh();
    schedulePoseRefresh();
    if (!document.hidden) {
      refreshStatus();
      refreshPreviewImages();
      refreshPose();
    }
  });
  await loadConfig(); await loadCalibrationHistory(); await refreshStatus(); refreshPreviewImages(); scheduleStatusRefresh(); schedulePreviewRefresh(); schedulePoseRefresh(); setInterval(() => $("#clock").textContent = new Date().toLocaleTimeString("zh-CN", {hour12:false}), 1000);
  window.addEventListener("resize", () => {
    drawPreviewCanvas("primary");
    drawPreviewCanvas("secondary");
    renderReprojection();
  });
});
