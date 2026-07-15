// =========================================================================
// 1. GLOBAL CONSTANTS & CONFIGURATION
// =========================================================================
const API_BASE = "http://127.0.0.1:8000"; // Ubah ke IP Server jika dideploy (misal: "http://192.168.1.10:8000")
const LEAD_COUNT = 3;
let charts = {};

// =========================================================================
// 2. ECG GRID PLUGIN (MILLIMETER CLINICAL GRID)
// =========================================================================
const ecgGridPlugin = {
  id: "ecgGrid",
  beforeDraw: (chart) => {
    const {
      ctx,
      chartArea: { left, top, right, bottom },
    } = chart;
    ctx.save();

    // Grid Kecil (Tipis) - Batas Kotak 1mm (0.04 Detik)
    ctx.strokeStyle = "rgba(255, 71, 71, 0.12)";
    ctx.lineWidth = 0.5;
    for (let x = left; x <= right; x += 6) {
      ctx.beginPath();
      ctx.moveTo(x, top);
      ctx.lineTo(x, bottom);
      ctx.stroke();
    }
    for (let y = top; y <= bottom; y += 6) {
      ctx.beginPath();
      ctx.moveTo(left, y);
      ctx.lineTo(right, y);
      ctx.stroke();
    }

    // Grid Besar (Tebal) - Batas Garis per 5mm (0.2 Detik)
    ctx.strokeStyle = "rgba(255, 71, 71, 0.4)";
    ctx.lineWidth = 1.0;
    for (let x = left; x <= right; x += 30) {
      ctx.beginPath();
      ctx.moveTo(x, top);
      ctx.lineTo(x, bottom);
      ctx.stroke();
    }
    for (let y = top; y <= bottom; y += 30) {
      ctx.beginPath();
      ctx.moveTo(left, y);
      ctx.lineTo(right, y);
      ctx.stroke();
    }
    ctx.restore();
  },
};
Chart.register(ecgGridPlugin);

// =========================================================================
// 3. CHART INITIALIZATION
// =========================================================================
function initChart(canvasId) {
  const canvasEl = document.getElementById(canvasId);
  if (!canvasEl) return null;

  const ctx = canvasEl.getContext("2d");
  return new Chart(ctx, {
    type: "line",
    data: {
      labels: [],
      datasets: [{ data: [], borderColor: "#1c1c1e", borderWidth: 1.5, pointRadius: 0, fill: false }],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { display: false },
        y: { border: { display: false }, grid: { display: false }, ticks: { color: "#1d1d1f", font: { size: 9, family: "monospace" } } },
      },
      plugins: { legend: { display: false } },
    },
  });
}

function initializeCharts() {
  for (let i = 0; i < LEAD_COUNT; i++) {
    charts[`raw_${i}`] = initChart(`raw_lead_${i}`);
    charts[`clean_${i}`] = initChart(`clean_lead_${i}`);
  }
}

// =========================================================================
// 4. UI HELPERS (FAIL-SAFE WRAPPERS)
// =========================================================================
function setText(id, value) {
  const el = document.getElementById(id);
  if (el) el.innerText = value;
}

function setHTML(id, htmlContent) {
  const el = document.getElementById(id);
  if (el) el.innerHTML = htmlContent;
}

function show(id, displayType = "block") {
  const el = document.getElementById(id);
  if (el) el.style.display = displayType;
}

function hide(id) {
  const el = document.getElementById(id);
  if (el) el.style.display = "none";
}

function setButtonStyle(id, background, color) {
  const btn = document.getElementById(id);
  if (btn) {
    btn.style.background = background;
    btn.style.color = color;
  }
}

// =========================================================================
// 5. TAB NAVIGATION
// =========================================================================
function switchTab(tabId) {
  if (tabId === 1) {
    show("view_tab_1", "flex");
    hide("view_tab_2");
    setButtonStyle("tab_btn_1", "#0071e3", "white");
    setButtonStyle("tab_btn_2", "#e5e5e7", "#1d1d1f");
  } else {
    hide("view_tab_1");
    show("view_tab_2", "flex");
    setButtonStyle("tab_btn_1", "#e5e5e7", "#1d1d1f");
    setButtonStyle("tab_btn_2", "#0071e3", "white");
    loadSimulatorFolders();
  }
}

// =========================================================================
// 6. DATASET OPTIONS API
// =========================================================================
// =========================================================================
// 6. DATASET OPTIONS API
// =========================================================================
async function loadRecordOptions() {
  try {
    const res = await fetch(`${API_BASE}/api/records`);
    const data = await res.json();

    const datasetEl = document.getElementById("dataset");
    const recordSelect = document.getElementById("record_id");
    if (!datasetEl || !recordSelect) return;

    const dataset = datasetEl.value;

    // -------------------------------------------------------------
    // SMART DEFAULT SELECTION (Sinkronisasi Parameter v5.0 Sebelum Inferensi)
    // -------------------------------------------------------------
    if (dataset === "ptbxl_500hz") {
      setTextBoxValue("median_kernel", "101");
      setTextBoxValue("highcut", "100");
      setSliderValue("w_level", "lbl_w_level", "4");
    } else if (dataset === "ptbxl_100hz") {
      setTextBoxValue("median_kernel", "51");
      setTextBoxValue("highcut", "45");
      setSliderValue("w_level", "lbl_w_level", "4");
    } else if (dataset === "chapman") {
      setTextBoxValue("median_kernel", "51");
      setTextBoxValue("highcut", "100"); // Dataset Chapman umumnya membutuhkan highcut lebar
      setSliderValue("w_level", "lbl_w_level", "4");
    } else if (dataset === "prosim_simulator") {
      setTextBoxValue("median_kernel", "51");
      setTextBoxValue("highcut", "45");
      setSliderValue("w_level", "lbl_w_level", "4");
    }
    // -------------------------------------------------------------

    recordSelect.innerHTML = "";

    if (!data[dataset] || data[dataset].length === 0) return;

    data[dataset].forEach((id) => {
      let opt = document.createElement("option");
      opt.value = id;
      opt.innerText = id;
      recordSelect.appendChild(opt);
    });

    recordSelect.selectedIndex = 0;
    await triggerProcessing();
  } catch (err) {
    console.error("Gagal memuat opsi rekaman dari backend:", err);
    setText("sample_class", "Error Koneksi");
  }
}

// Helper kecil tambahan untuk disisipkan pada Bagian 3 / Bagian 9 (Utility)
function setTextBoxValue(id, value) {
  const el = document.getElementById(id);
  if (el) el.value = value;
}

function setSliderValue(inputId, labelId, value) {
  const input = document.getElementById(inputId);
  const label = document.getElementById(labelId);
  if (input) input.value = value;
  if (label) label.innerText = value;
}

// =========================================================================
// 7. PROCESSING PIPELINE API (TAB 1 CORE)
// =========================================================================
async function triggerProcessing() {
  const getVal = (id) => document.getElementById(id)?.value || "";
  const r = getVal("record_id");
  if (!r) return;

  try {
    const d = getVal("dataset");
    const t_fs = getVal("target_fs");
    const wav = getVal("wavelet");
    const lvl = getVal("w_level");
    const med = getVal("median_kernel");
    const low = getVal("lowcut");
    const high = getVal("highcut");

    const url = `${API_BASE}/api/process?dataset=${d}&record_id=${r}&target_fs=${t_fs}&wavelet=${wav}&w_level=${lvl}&median_kernel=${med}&lowcut=${low}&highcut=${high}`;

    const res = await fetch(url);
    const result = await res.json();

    // Jalankan seluruh fungsi modular perenderan data
    renderDiagnosis(result);
    renderPerformance(result);
    renderHolter(result);
    renderCharts(result);
  } catch (err) {
    console.error("Gagal memproses pipeline DSP:", err);
    setText("sample_class", "Offline");
    setText("ai_class", "Offline");
  }
}

// =========================================================================
// 8. DATA RENDERING FUNCTIONS (TAB 1)
// =========================================================================
function renderDiagnosis(result) {
  const gtClass = result.target_class || "Unknown";
  setText("sample_class", gtClass);
  setText("sample_class_detail", gtClass);

  const kerasPred = result.keras_prediction || "Offline";
  const tflitePred = result.tflite_prediction || "Offline";

  const aiClassText = kerasPred && !kerasPred.includes("Offline") ? kerasPred : tflitePred && !tflitePred.includes("Offline") ? tflitePred : "Offline";
  const aiConfText = result.keras_confidence ? `${result.keras_confidence}%` : result.tflite_confidence ? `${result.tflite_confidence}%` : "--%";

  setText("ai_class", aiClassText);
  setText("ai_conf", `(${aiConfText})`);

  setText("keras_class", kerasPred);
  setText("keras_conf", result.keras_confidence ? `(${result.keras_confidence}%)` : "(--%)");

  setText("tflite_class", tflitePred);
  setText("tflite_conf", result.tflite_confidence ? `(${result.tflite_confidence}%)` : "(--%)");
}

function renderPerformance(result) {
  if (result.metrics) {
    setText("val_latency", `${result.metrics.latency_ms.toFixed(2)} ms`);
    setText("val_memory", `${result.metrics.peak_memory_mb.toFixed(4)} MB`);
  }
}

function renderHolter(result) {
  const holter = result.holter || (result.metrics ? result.metrics.holter : null);
  if (!holter) return;

  setHTML("h_hr", `${holter.hr || "--"}<span class="holter-unit"> BPM</span>`);
  setHTML("h_rr", `${holter.rr_avg_ms || "--"}<span class="holter-unit"> ms</span>`);
  setHTML("h_hrv", `${holter.rmssd_ms || "--"}<span class="holter-unit"> ms</span>`);

  const stText = holter.st_dev_mv !== undefined ? holter.st_dev_mv.toFixed(3) : "--";
  setHTML("h_st", `${stText}<span class="holter-unit"> mV</span>`);
  setHTML("h_qtc", `${holter.qtc_ms || "--"}<span class="holter-unit"> ms</span>`);

  renderEvents(holter.events);
}

function renderEvents(events) {
  const eventDiv = document.getElementById("h_events");
  if (!eventDiv) return;

  eventDiv.innerHTML = "";
  if (events && events.length > 0) {
    events.forEach((evt) => {
      let span = document.createElement("span");
      span.className = "event-tag";
      if (evt.includes("⚠️")) {
        span.style.background = "#ffe5e5";
        span.style.color = "#ff453a";
      }
      span.innerText = evt;
      eventDiv.appendChild(span);
    });
  } else {
    eventDiv.innerText = "--";
  }
}

function renderCharts(result) {
  for (let i = 0; i < LEAD_COUNT; i++) {
    const rawData = result.raw_signals ? result.raw_signals[`lead_${i}`] : null;
    const cleanData = result.clean_signals ? result.clean_signals[`lead_${i}`] : null;

    updateChartData(charts[`raw_${i}`], rawData);
    updateChartData(charts[`clean_${i}`], cleanData);
  }
}

// =========================================================================
// 9. SIMULATOR HARDWARE DSP API & RENDERING (TAB 2)
// =========================================================================
async function loadSimulatorFolders() {
  setText("sim_placeholder_text", "Loading daftar rekaman dari hardware simulator...");
  try {
    const res = await fetch(`${API_BASE}/api/simulator/folders`);
    const folders = await res.json();
    const selectNode = document.getElementById("sim_folder_select");
    if (!selectNode) return;

    selectNode.innerHTML = "";
    folders.forEach((f) => {
      let opt = document.createElement("option");
      opt.value = f;
      opt.innerText = f;
      selectNode.appendChild(opt);
    });
    setText("sim_placeholder_text", "Silakan pilih folder rekaman kemudian jalankan analisis.");
  } catch (err) {
    console.error("Gagal memuat folder simulator:", err);
    setText("sim_placeholder_text", "Gagal memuat folder: Periksa koneksi backend Hardware API.");
  }
}

async function triggerSimulatorAnalysis() {
  const getVal = (id) => document.getElementById(id)?.value || "";
  const folder = getVal("sim_folder_select");
  if (!folder) return;

  setText("sim_placeholder_text", "Sedang menghitung transformasi Fourier (FFT) dan melacak koordinat puncak R... Mohon tunggu.");
  hide("sim_report_img");

  try {
    const fs = getVal("sim_fs");
    const wav = getVal("sim_wavelet");
    const lvl = getVal("sim_w_level");
    const med = getVal("sim_median_kernel");
    const low = getVal("sim_lowcut");
    const high = getVal("sim_highcut");

    const url = `${API_BASE}/api/simulator/analyze?folder_name=${folder}&target_fs=${fs}&wavelet=${wav}&w_level=${lvl}&median_kernel=${med}&lowcut=${low}&highcut=${high}`;
    const res = await fetch(url);
    const result = await res.json();

    if (result.status === "success") {
      hide("sim_placeholder_text");
      show("sim_metrics_grid", "grid");

      renderSimulatorMetrics(result);
      renderSimulatorImage(result.image);
      renderRecommendation(result);
    } else {
      alert(result.message);
      setText("sim_placeholder_text", "Terjadi kegagalan analisis: " + result.message);
    }
  } catch (err) {
    console.error("Error pada modul simulator:", err);
    alert("Gagal terhubung ke modul komparasi hardware simulator.");
  }
}

function renderSimulatorMetrics(result) {
  setHTML("sim_val_bpm", `${result.calculated_bpm}<span class="holter-unit"> BPM</span>`);
  setHTML("sim_val_rr", `${result.avg_rr_seconds}<span class="holter-unit"> detik</span>`);
  setHTML("sim_val_noise", `${result.dominant_noise_freq}<span class="holter-unit"> Hz</span>`);
  setHTML("sim_val_attenuation", `${result.attenuation_median_pct}<span class="holter-unit"> %</span>`);
}

function renderSimulatorImage(base64Image) {
  const imgNode = document.getElementById("sim_report_img");
  if (imgNode) {
    imgNode.src = "data:image/png;base64," + base64Image;
    show("sim_report_img");
  }
}

function renderRecommendation(result) {
  const recCard = document.getElementById("sim_recommendation_card");
  if (!recCard) return;

  show("sim_recommendation_card");
  const bpm = parseFloat(result.calculated_bpm);
  const atten = parseFloat(result.attenuation_median_pct);

  if (bpm < 65.0 && atten > 4.0) {
    recCard.style.background = "#fff2e6";
    recCard.style.color = "#ff9f0a";
    recCard.style.borderLeft = "5px solid #ff9f0a";
    setHTML(
      "sim_recommendation_text",
      `⚠️ <b>Rekomendasi Deteksi Bradikardia:</b> Sinyal terdeteksi sebagai Denyut Jantung Lambat (${result.calculated_bpm} BPM) dan filter median mereduksi amplitudo puncak R sebesar ${result.attenuation_median_pct}%. Filter terlalu agresif memotong fase isoelektrik. <br><b>Saran Tindakan:</b> Ubah parameter Median Filter Kernel di Tab 1 menjadi 101 atau 151 sampel sebelum grid search massal dilakukan.`,
    );
  } else {
    recCard.style.background = "#e6f9ed";
    recCard.style.color = "#34c759";
    recCard.style.borderLeft = "5px solid #34c759";
    setHTML(
      "sim_recommendation_text",
      `✅ <b>Status Filter Stabil:</b> Redaman amplitudo puncak r-wave berada pada rentang batas aman (${result.attenuation_median_pct}%). Segmentasi morfologi dan interval waktu spasio-temporal EKG lulus uji distorsi klinis AHA.`,
    );
  }
}

// =========================================================================
// 10. UTILITY FUNCTIONS
// =========================================================================
function updateChartData(chartInstance, dataArray) {
  if (!chartInstance || !dataArray || dataArray.length === 0) return;
  chartInstance.data.labels = Array.from({ length: dataArray.length }, (_, idx) => idx);
  chartInstance.data.datasets[0].data = dataArray;
  chartInstance.update("none"); // Mencegah pemborosan resource rendering animasi berlebih
}

// =========================================================================
// 11. STARTUP / APPLICATION ENTRY POINT
// =========================================================================
window.onload = async () => {
  initializeCharts();
  await loadRecordOptions();
};
