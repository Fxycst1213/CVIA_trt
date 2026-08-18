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
let latestProjectionOrigin = null;
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
let offlineSyncGenerating = false;
let offsetEstimating = false;

// 网页刷新周期：位姿页打开时曲线 500 ms；其他页面跟随 1000 ms 状态周期。
const POSE_REFRESH_MS = 200;
const STATUS_REFRESH_MS = 1000;
const PREVIEW_REQUEST_TIMEOUT_MS = 2000;
const COMPAT_STORAGE_KEY = "cvia.compat.mode";

const poseAxes = [
  {key:"X", field:"x", group:"position"}, {key:"Y", field:"y", group:"position"}, {key:"Z", field:"z", group:"position"},
  {key:"Rx", field:"rx", group:"euler_deg"}, {key:"Ry", field:"ry", group:"euler_deg"}, {key:"Rz", field:"rz", group:"euler_deg"}
];
const projectionColors = {origin:"#ef7b72", feature:"#65d7e4"};

function resolveCompatibilityMode(mode) {
  if (mode === "win7" || mode === "win11") return mode;
  return /Windows NT 6\.1/i.test(navigator.userAgent) ? "win7" : "win11";
}

function canvasPixelRatio() {
  return document.documentElement.classList.contains("compat-win7")
    ? 1
    : Math.min(window.devicePixelRatio || 1, 2);
}

function applyCompatibilityMode(mode, announce = false) {
  const preference = ["auto", "win7", "win11"].includes(mode) ? mode : "auto";
  const resolved = resolveCompatibilityMode(preference);
  const root = document.documentElement;
  root.classList.remove("compat-win7", "compat-win11");
  root.classList.add(`compat-${resolved}`);
  root.dataset.compatPreference = preference;
  root.dataset.compatResolved = resolved;
  try { localStorage.setItem(COMPAT_STORAGE_KEY, preference); } catch (error) {}
  const select = $("#compatMode"), state = $("#compatModeState");
  if (select) select.value = preference;
  if (state) state.textContent = preference === "auto"
    ? `自动 · ${resolved === "win7" ? "Win7 兼容" : "Win11"}`
    : resolved === "win7" ? "低负载兼容" : "完整效果";
  drawPreviewCanvas("primary");
  drawPreviewCanvas("secondary");
  renderReprojection();
  if (announce) showToast(resolved === "win7" ? "已切换 Windows 7 兼容模式" : "已切换 Windows 11 显示模式");
}

function initializeCompatibilityMode() {
  const root = document.documentElement;
  const preference = root.dataset.compatPreference || "auto";
  applyCompatibilityMode(preference);
  $("#compatMode")?.addEventListener("change", event => applyCompatibilityMode(event.target.value, true));
}

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
  const syncEnabled = Boolean(config.sync?.enabled);
  $$('[data-path^="sync."]').forEach(control => {
    if (control.dataset.path !== "sync.enabled") control.disabled = !syncEnabled;
  });
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

function clearWorldRotationInputs() {
  ["#worldRotationX", "#worldRotationY", "#worldRotationZ"].forEach(selector => {
    $(selector).value = "0";
  });
}

function formatRotationAngle(value) {
  const sign = value > 0 ? "+" : "";
  return `${sign}${Number(value.toPrecision(10))}°`;
}

function selectedWorldRotationOrder() {
  return $('input[name="world-rotation-order"]:checked')?.value || "XYZ";
}

function renderWorldRotationOrder() {
  const order = selectedWorldRotationOrder();
  const axisClass = {X:"axis-x", Y:"axis-y", Z:"axis-z"};
  $("#worldAxisOrder").innerHTML = order.split("").map((axis, index) =>
    `<span class="${axisClass[axis]}">${axis}</span>${index < 2 ? "<i>→</i>" : ""}`
  ).join("") + "<small>当前执行顺序</small>";
  $("#worldRotationFormula").textContent = order.split("").reverse().map(axis => `R${axis.toLowerCase()}`).join(" · ") + " · R";
}

function applyWorldRotationToExtrinsic() {
  const inputs = [$("#worldRotationX"), $("#worldRotationY"), $("#worldRotationZ")];
  const angles = inputs.map(input => Number(input.value));
  if (inputs.some((input, index) => input.value.trim() === "" || !Number.isFinite(angles[index]))) {
    showToast("世界轴旋转角度必须是有限数字", true);
    return;
  }
  if (angles.every(value => Math.abs(value) < 1e-12)) {
    showToast("至少输入一个非零旋转角度", true);
    return;
  }
  try {
    const order = selectedWorldRotationOrder();
    config.calibration.extrinsic = window.ExtrinsicRotation.applyWorldRotation(
      config.calibration.extrinsic,
      angles[0], angles[1], angles[2], order
    );
    bindConfig();
    clearWorldRotationInputs();
    $("#worldRotationLast").textContent = `顺序 ${order.split("").join("→")} · X ${formatRotationAngle(angles[0])} · Y ${formatRotationAngle(angles[1])} · Z ${formatRotationAngle(angles[2])}`;
    showToast(`已按世界轴 ${order.split("").join("→")} 更新 T_M_C；请检查矩阵后保存配置`);
  } catch (error) {
    showToast(error.message || "世界轴旋转应用失败", true);
  }
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
  const offlineButton = $("#generateOfflineSync");
  if (offlineButton && !offlineSyncGenerating) {
    offlineButton.disabled = Boolean(inference.running);
    offlineButton.textContent = inference.running ? "停止后生成" : "生成离线报告";
  }
  const estimateButton = $("#estimateOffset");
  if (estimateButton && !offsetEstimating) {
    estimateButton.disabled = Boolean(inference.running);
    estimateButton.textContent = inference.running ? "停止后估计" : "自动估计 offset";
  }
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
  const ratio = canvasPixelRatio();
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

async function refreshReprojection() {
  if (projectionRequestPending) return;
  projectionRequestPending = true;
  try {
    const response = await fetch("/api/reprojection", {cache:"no-store"});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "重投影状态读取失败");
    runtimeProjectionAvailable = Boolean(result.available);
    if (!runtimeProjectionAvailable) { latestProjectionOrigin = null; latestProjectionPoints = []; }
    if (runtimeProjectionAvailable) {
      latestPoseFileTime = result.mtime_ns ? result.mtime_ns / 1e6 : 0;
      if (result.valid && Array.isArray(result.pose) && result.pose.length === 6) {
        latestPoseRow = [Number(result.timestamp), ...result.pose.map(Number)];
        const origin = Array.isArray(result.origin) && result.origin.length === 2
          ? result.origin.map(Number) : null;
        latestProjectionOrigin = origin?.every(Number.isFinite) ? origin : null;
        latestProjectionPoints = Array.isArray(result.points) ? result.points
          .filter(point => Array.isArray(point) && point.length === 2)
          .map(point => point.map(Number))
          .filter(point => point.every(Number.isFinite)) : [];
        projectionEmptyMessage = "等待位姿";
      } else {
        latestPoseRow = null;
        latestProjectionOrigin = null;
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
  const rect = canvas.getBoundingClientRect(), ratio = canvasPixelRatio();
  const width = Math.max(1, Math.round(rect.width * ratio)), height = Math.max(1, Math.round(rect.height * ratio));
  if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
  const ctx = canvas.getContext("2d"); ctx.setTransform(ratio, 0, 0, ratio, 0, 0); ctx.clearRect(0, 0, rect.width, rect.height);
  readout.classList.remove("is-outside", "is-stale");

  if (!latestPoseRow) { $("#projectionState").textContent = projectionEmptyMessage; $("#projectionPixel").textContent = "u -- · v --"; return; }
  const sourceWidth = Math.max(1, Number(config.runtime.frame_width));
  const sourceHeight = Math.max(1, Number(config.runtime.frame_height));
  const imageWidth = previewNaturalSize.primary.width || sourceWidth;
  const imageHeight = previewNaturalSize.primary.height || sourceHeight;
  // C++ 分别发布刚体坐标系原点 (0,0,0) 和模型特征点。最终位姿属于原点，
  // 因此红点必须使用 origin；不能再把 _p3d[0] 当作刚体中心。
  if (!latestProjectionOrigin) {
    $("#projectionState").textContent = "等待刚体中心重投影";
    $("#projectionPixel").textContent = "CENTER u -- · v --";
    readout.classList.add("is-outside");
    return;
  }
  const toDisplayPoint = ([sourceU, sourceV], index = -1) => {
    const u = sourceU * imageWidth / sourceWidth, v = sourceV * imageHeight / sourceHeight;
    return {index, u, v, inside:u > 0 && u < imageWidth && v > 0 && v < imageHeight};
  };
  const origin = toDisplayPoint(latestProjectionOrigin);
  const displayPoints = latestProjectionPoints.map((point, index) => toDisplayPoint(point, index));
  const insideCount = displayPoints.filter(point => point.inside).length;
  const allInside = origin.inside && insideCount === displayPoints.length;
  const age = latestPoseFileTime ? Date.now() - latestPoseFileTime : 0;
  const stale = age > 2500;
  $("#projectionPixel").textContent = `CENTER u ${origin.u.toFixed(1)} · v ${origin.v.toFixed(1)}`;
  $("#projectionState").textContent = stale
    ? `位姿已停止更新 · 保留中心和 ${displayPoints.length} 个特征点`
    : !origin.inside
      ? `中心在画面外 · ${insideCount}/${displayPoints.length} 个特征点在画面内`
      : allInside
        ? `中心和 ${displayPoints.length} 个特征点重投影有效`
        : `中心有效 · ${insideCount}/${displayPoints.length} 个特征点在画面内`;
  readout.classList.toggle("is-outside", !allInside); readout.classList.toggle("is-stale", stale);
  if (!previewBitmaps.primary) return;

  const scale = Math.min(rect.width / imageWidth, rect.height / imageHeight);
  const drawWidth = imageWidth * scale, drawHeight = imageHeight * scale;
  const offsetX = (rect.width - drawWidth) / 2, offsetY = (rect.height - drawHeight) / 2;
  // 停止推理后预览图本身也冻结在同一帧；保留半透明重投影比直接清空更便于检查。
  ctx.globalAlpha = stale ? 0.48 : 1;
  for (const point of displayPoints) {
    if (!point.inside) continue;
    const px = offsetX + point.u * scale, py = offsetY + point.v * scale;
    const color = projectionColors.feature;
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 1.2;
    ctx.beginPath(); ctx.arc(px, py, 5, 0, Math.PI * 2); ctx.stroke();
    ctx.beginPath(); ctx.arc(px, py, 2.4, 0, Math.PI * 2); ctx.fill();
    const pointLabel = `P${point.index}`;
    const labelX = Math.min(rect.width - 18, Math.max(4, px + 7));
    const labelY = Math.min(rect.height - 7, Math.max(10, py - 7));
    ctx.font = "600 9px ui-monospace, monospace"; ctx.lineWidth = 3;
    ctx.strokeStyle = "rgba(5,10,12,.92)"; ctx.strokeText(pointLabel, labelX, labelY);
    ctx.fillStyle = color; ctx.fillText(pointLabel, labelX, labelY);
  }

  if (!origin.inside) { ctx.globalAlpha = 1; return; }
  const originPx = offsetX + origin.u * scale, originPy = offsetY + origin.v * scale;
  ctx.strokeStyle = projectionColors.origin; ctx.fillStyle = projectionColors.origin; ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.arc(originPx, originPy, 8, 0, Math.PI * 2); ctx.stroke();
  ctx.beginPath(); ctx.arc(originPx, originPy, 3.2, 0, Math.PI * 2); ctx.fill();
  ctx.beginPath();
  ctx.moveTo(originPx - 15, originPy); ctx.lineTo(originPx + 15, originPy);
  ctx.moveTo(originPx, originPy - 15); ctx.lineTo(originPx, originPy + 15); ctx.stroke();
  const originLabelX = Math.min(rect.width - 38, Math.max(4, originPx + 7));
  const originLabelY = Math.min(rect.height - 7, Math.max(10, originPy - 7));
  ctx.font = "600 9px ui-monospace, monospace"; ctx.lineWidth = 3;
  ctx.strokeStyle = "rgba(5,10,12,.92)"; ctx.strokeText("CENTER", originLabelX, originLabelY);
  ctx.fillStyle = projectionColors.origin; ctx.fillText("CENTER", originLabelX, originLabelY);
  const [poseX, poseY, poseZ] = latestPoseRow.slice(1, 4).map(Number);
  const label = `MOCAP XYZ ${poseX.toFixed(1)}  ${poseY.toFixed(1)}  ${poseZ.toFixed(1)}`;
  ctx.font = "10px ui-monospace, monospace"; const labelWidth = ctx.measureText(label).width;
  const labelX = Math.min(Math.max(8, originPx + 13), rect.width - labelWidth - 14), labelY = Math.max(17, originPy - 17);
  ctx.fillStyle = "rgba(5,10,12,.82)"; ctx.fillRect(labelX - 5, labelY - 11, labelWidth + 10, 16);
  ctx.fillStyle = "#f3aaa4"; ctx.fillText(label, labelX, labelY);
  ctx.globalAlpha = 1;
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
  const synchronization = result.synchronization || {};
  const detectionPose = detection.pose || {};
  const synchronized = ["matched", "interpolated"].includes(synchronization.status);
  const mocapPose = synchronized ? (synchronization.pose || {}) : {};

  for (const axis of poseAxes) {
    $(`#detectValue${axis.key}`).textContent = formatPoseValue(detectionPose[axis.field]);
    const mocapGroup = mocapPose[axis.group] || {};
    $(`#mocapValue${axis.key}`).textContent = formatPoseValue(mocapGroup[axis.field]);
  }

  const detectionAge = Number(detection.age_ms);
  const detectionLive = Boolean(detection.available) && Number.isFinite(detectionAge) && detectionAge <= 2500;
  $("#detectionState").textContent = !detection.available ? "等待视觉位姿" : detectionLive ? "视觉检测在线" : "视觉位姿已暂停";
  const timestampLabel = detection.timestamp_source === "v4l2_driver"
    ? `V4L2 ${detection.timestamp_point === "start_of_exposure" ? "SOE" : "EOF/未知"}`
    : detection.source === "reprojection" ? "软件接收时间" : "CSV";
  const pipelineLatency = detection.pipeline_latency_ms == null
    ? NaN : Number(detection.pipeline_latency_ms);
  $("#detectionMeta").textContent = detection.available
    ? `${timestampLabel} · ${Number.isFinite(pipelineLatency) ? `采集→结果 ${pipelineLatency.toFixed(1)} ms` : "延迟未知"} · ${Number.isFinite(detectionAge) ? `${detectionAge.toFixed(0)} ms 前更新` : "更新时间未知"}`
    : "PnP 尚未输出";
  setSourceState("#detectionDot", detectionLive, Boolean(detection.available) && !detectionLive);

  const mocapLabels = {
    disabled:"动捕接收已关闭", starting:"正在启动动捕", missing:"桥接程序缺失",
    connecting:"正在连接 SDK", ready:"SDK 已连接", waiting:"等待目标刚体",
    live:"NOKOV 动捕在线", stale:"动捕数据已暂停", invalid:"当前刚体解算无效",
    disconnected:"SDK 连接已断开", error:"动捕启动失败", unavailable:"动捕尚未初始化"
  };
  $("#mocapState").textContent = mocapLabels[mocap.status] || "等待动捕数据";
  const trackerName = synchronization.pose?.tracker_name || mocap.pose?.tracker_name || mocap.selector || "RIGID BODY";
  $("#mocapTrackerHeading").textContent = String(trackerName).toUpperCase();
  $("#mocapMeta").textContent = synchronized
    ? `配对帧 ${(synchronization.mocap_frames || [mocapPose.frame]).join(" → ")} · |Δt| ${Math.abs(Number(synchronization.sync_error_ms)).toFixed(2)} ms`
    : mocap.pose?.frame !== undefined
    ? `最新帧 ${mocap.pose.frame} 在线，但尚未匹配当前视觉帧`
    : (mocap.message || `${mocap.selector || "目标刚体"} · SDK ${mocap.sdk_version || "--"}`);
  setSourceState("#mocapDot", synchronized, mocap.status === "live" && !synchronized || ["stale", "invalid", "waiting", "ready"].includes(mocap.status));
  updateMocapNetworkStatus(mocap);

  const bothLive = detectionLive && mocap.status === "live";
  const eitherAvailable = Boolean(detection.available || mocap.pose?.frame !== undefined);
  $("#poseState").textContent = synchronized ? "实时已配对" : bothLive ? "双源在线 / 未配对" : eitherAvailable ? "单源 / 待同步" : "等待双源";
  $("#poseState").classList.toggle("is-live", synchronized);

  const rail = $("#syncRail");
  rail.dataset.state = synchronized ? "matched" : synchronization.status === "unmatched" ? "unmatched" : "waiting";
  const visualStamp = Number(synchronization.visual_timestamp_ms);
  $("#syncVisionStamp").textContent = Number.isFinite(visualStamp) && visualStamp > 0
    ? `${new Date(visualStamp).toLocaleTimeString("zh-CN", {hour12:false})}.${String(Math.trunc(visualStamp) % 1000).padStart(3, "0")}`
    : "等待视觉帧";
  const syncError = synchronization.sync_error_ms == null ? NaN : Number(synchronization.sync_error_ms);
  $("#syncDeltaValue").textContent = Number.isFinite(syncError)
    ? `${syncError >= 0 ? "+" : ""}${syncError.toFixed(2)} ms`
    : "-- ms";
  $("#syncMethod").textContent = synchronization.status === "interpolated"
    ? `SLERP · α ${Number(synchronization.alpha).toFixed(3)}`
    : synchronization.status === "matched" ? "NEAREST FRAME" : synchronization.message || "等待时间配对";
  const frames = synchronization.mocap_frames || [];
  $("#syncMocapFrames").textContent = frames.length ? `帧 ${frames.join(" → ")}` : "等待动捕帧";
  $("#syncBufferMeta").textContent = `历史缓存 ${Number(synchronization.history_samples || 0)} 帧`;

  const delta = result.receive_delta_ms == null ? NaN : Number(result.receive_delta_ms);
  if (detectionLive && mocap.status === "live" && Number.isFinite(delta)) {
    const relation = delta >= 0 ? "视觉更新晚于动捕" : "视觉更新早于动捕";
    $("#comparisonDelta").textContent = `本机更新时差 · ${relation} ${Math.abs(delta).toFixed(1)} ms`;
  } else {
    $("#comparisonDelta").textContent = "双源同时在线后显示本机更新时差";
  }
}

function renderOffsetEstimation(estimation) {
  const panel = $("#offsetEstimate");
  if (!estimation) {
    panel.dataset.confidence = "none";
    $("#offsetEstimateValue").textContent = "尚未估计";
    $("#offsetEstimateMeta").textContent = "停止推理后，建议做一段非周期的平移和转动";
    return;
  }
  panel.dataset.confidence = estimation.reliable ? estimation.confidence : "low";
  const sign = Number(estimation.offset_ms) >= 0 ? "+" : "";
  $("#offsetEstimateValue").textContent = `${sign}${Number(estimation.offset_ms).toFixed(1)} ms · ${estimation.applied ? "已应用" : "未应用"}`;
  const confidenceLabels = {high:"高置信", medium:"中置信", low:"低置信"};
  $("#offsetEstimateMeta").textContent = `${confidenceLabels[estimation.confidence] || "置信度未知"} · 相关 ${Number(estimation.correlation).toFixed(3)} · 峰差 ${Number(estimation.peak_margin).toFixed(3)} · ${estimation.message || ""}`;
}

function renderOfflineSync(result) {
  renderOffsetEstimation(result?.offset_estimation);
  const available = Boolean(result?.available && result.summary);
  $("#offlineSyncEmpty").hidden = available;
  $("#offlineSyncResult").hidden = !available;
  if (!available) return;
  const summary = result.summary;
  $("#offlineCoverage").textContent = `${Number(summary.coverage_percent || 0).toFixed(1)}%`;
  $("#offlineMatched").textContent = `${summary.matched_samples || 0} / ${summary.detection_samples || 0}`;
  $("#offlineP95").textContent = summary.p95_abs_error_ms == null ? "--" : `${Number(summary.p95_abs_error_ms).toFixed(2)} ms`;
  $("#offlineGeneratedAt").textContent = `${summary.generated_at || "报告已生成"} · 偏移 ${Number(summary.offset_ms || 0).toFixed(1)} ms`;
  const cacheKey = summary.generated_unix_ns || Date.now();
  const reportUrls = result.report_urls || {
    camera: "/api/sync/offline/report/camera.svg",
    mocap: "/api/sync/offline/report/mocap.svg",
    composite: result.report_url || "/api/sync/offline/report.svg",
  };
  for (const [mode, prefix] of [["camera", "camera"], ["mocap", "mocap"], ["composite", "composite"]]) {
    const reportUrl = `${reportUrls[mode]}?v=${cacheKey}`;
    $(`#${prefix}ReportImage`).src = reportUrl;
    $(`#${prefix}ReportLink`).href = reportUrl;
    $(`#open${prefix[0].toUpperCase()}${prefix.slice(1)}Report`).href = reportUrl;
  }
  $("#downloadSyncedCsv").href = result.csv_url || "/api/sync/offline/download.csv";
}

async function estimateOffset() {
  if (offsetEstimating) return;
  offsetEstimating = true;
  const button = $("#estimateOffset");
  button.disabled = true; button.textContent = "正在计算互相关…";
  try {
    const response = await fetch("/api/sync/offline/estimate-offset", {method:"POST", cache:"no-store"});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "自动估计时间补偿失败");
    renderOffsetEstimation(result.offset_estimation);
    if (result.offset_estimation?.applied) {
      await loadConfig();
      await refreshOfflineSync();
      showToast(`已自动应用 offset ${Number(result.config_offset_ms).toFixed(1)} ms；请重新生成离线报告`);
    } else {
      showToast(result.offset_estimation?.message || "估计结果未通过可靠性检查", true);
    }
  } catch (error) {
    showToast(error.message, true);
  } finally {
    offsetEstimating = false;
    button.disabled = Boolean(previousInferenceRunning);
    button.textContent = previousInferenceRunning ? "停止后估计" : "重新估计 offset";
  }
}

async function refreshOfflineSync() {
  try {
    const response = await fetch("/api/sync/offline", {cache:"no-store"});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "离线同步状态读取失败");
    renderOfflineSync(result);
  } catch (error) {
    const empty = $("#offlineSyncEmpty");
    empty.hidden = false;
    empty.querySelector("strong").textContent = "离线报告状态读取失败";
    empty.querySelector("small").textContent = error.message;
  }
}

async function generateOfflineSync() {
  if (offlineSyncGenerating) return;
  offlineSyncGenerating = true;
  const button = $("#generateOfflineSync");
  button.disabled = true; button.textContent = "正在配对并绘图…";
  try {
    const response = await fetch("/api/sync/offline/generate", {method:"POST", cache:"no-store"});
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "离线同步报告生成失败");
    renderOfflineSync(result);
    showToast("同步 CSV 与相机、动捕、合成三张六轴图已生成");
  } catch (error) {
    showToast(error.message, true);
  } finally {
    offlineSyncGenerating = false;
    button.disabled = Boolean(previousInferenceRunning);
    button.textContent = previousInferenceRunning ? "停止后生成" : "重新生成报告";
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
    // C++ 已提供同帧模型点投影时，图像预览循环负责绘制；这里只更新旧版 CSV 回退投影。
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
  initializeCompatibilityMode();
  createControls();
  $("#configForm").addEventListener("input", event => { if (event.target.dataset.path) readControl(event.target); });
  $("#configForm").addEventListener("change", event => { if (event.target.dataset.path) readControl(event.target); });
  $("#configForm").addEventListener("submit", event => { event.preventDefault(); saveConfig(); });
  $("#reloadButton").addEventListener("click", () => loadConfig(true));
  $("#resetCalibration").addEventListener("click", () => { config.calibration = clone(savedConfig.calibration); bindConfig(); showToast("标定值已恢复到当前保存状态"); });
  $("#applyWorldRotation").addEventListener("click", applyWorldRotationToExtrinsic);
  $("#clearWorldRotation").addEventListener("click", () => { clearWorldRotationInputs(); $("#worldRotationLast").textContent = "旋转角度已清零"; });
  $$('input[name="world-rotation-order"]').forEach(control => control.addEventListener("change", renderWorldRotationOrder));
  renderWorldRotationOrder();
  $("#loadCalibrationHistory").addEventListener("click", applyCalibrationHistory);
  $$('input[name="mocap-target-mode"]').forEach(control => control.addEventListener("change", () => changeMocapTargetMode(control.value)));
  $("#mocapTargetValue").addEventListener("input", updateMocapTargetEditor);
  $("#mocapDiscoveredBodies").addEventListener("change", selectDiscoveredMocapBody);
  $("#applyMocapTarget").addEventListener("click", applyMocapTarget);
  $("#stopInferenceButton").addEventListener("click", stopInferenceOrSave);
  $("#saveSessionButton").addEventListener("click", savePnpSession);
  $("#saveSessionLater").addEventListener("click", () => { $("#sessionModal").hidden = true; sessionModalDismissed = true; });
  $("#generateOfflineSync").addEventListener("click", generateOfflineSync);
  $("#estimateOffset").addEventListener("click", estimateOffset);
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
  await loadConfig(); await loadCalibrationHistory(); await refreshStatus(); await refreshOfflineSync(); refreshPreviewImages(); scheduleStatusRefresh(); schedulePreviewRefresh(); schedulePoseRefresh(); setInterval(() => $("#clock").textContent = new Date().toLocaleTimeString("zh-CN", {hour12:false}), 1000);
  window.addEventListener("resize", () => {
    drawPreviewCanvas("primary");
    drawPreviewCanvas("secondary");
    renderReprojection();
  });
});
