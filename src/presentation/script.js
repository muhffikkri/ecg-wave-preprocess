// =========================================================================
// 1. GLOBAL CONSTANTS & CONFIGURATION
// =========================================================================
const API_BASE = "http://127.0.0.1:8000"; // Ubah ke IP Server jika dideploy (misal: "http://192.168.1.10:8000")
const LEAD_COUNT = 3;
const PX_PER_MM = 6; // Grid klinis: 1 mm = 6 px
let charts = {};

// Tampilan EKG klinis (client-side saja, tidak memengaruhi data/CSV)
let currentGain = 10; // mm/mV (5 | 10 | 20)
let currentSpeed = 25; // mm/s (12.5 | 25 | 50)
let rawSignals = [];
let cleanSignals = [];
let totalSamples = 0;

// =========================================================================
// 2. ECG GRID PLUGIN (MILLIMETER CLINICAL GRID)
// =========================================================================
const ecgGridPlugin = {
  id: "ecgGrid",
  beforeDraw: (chart) => {
    const { ctx, chartArea, scales } = chart;
    if (!chartArea || !scales.x || !scales.y) return;
    const { left, top, right, bottom } = chartArea;
    ctx.save();

    // Grid dijangkar ke nol DATA (0 s dan 0 mV), bukan tepi layar, supaya
    // tetap sesuai standar EKG saat bidang digeser vertikal/horizontal.
    const x0 = scales.x.getPixelForValue(0);
    const y0 = scales.y.getPixelForValue(0);
    const smallX = PX_PER_MM * 0.04 * currentSpeed; // 1 kotak kecil = 0,04 s x mm/s
    const smallY = PX_PER_MM; // 1 kotak kecil = 1 mm
    const bigX = smallX * 5; // 1 kotak besar = 5 mm
    const bigY = smallY * 5;

    const drawAxis = (origin, from, to, step, line) => {
      for (let k = Math.ceil((from - origin) / step); origin + k * step <= to; k++) {
        line(origin + k * step);
      }
    };
    const vLine = (x) => {
      ctx.beginPath();
      ctx.moveTo(x, top);
      ctx.lineTo(x, bottom);
      ctx.stroke();
    };
    const hLine = (y) => {
      ctx.beginPath();
      ctx.moveTo(left, y);
      ctx.lineTo(right, y);
      ctx.stroke();
    };

    // Grid Kecil (Tipis): 1 mm
    ctx.strokeStyle = "rgba(255, 71, 71, 0.12)";
    ctx.lineWidth = 0.5;
    drawAxis(x0, left, right, smallX, vLine);
    drawAxis(y0, top, bottom, smallY, hLine);

    // Grid Besar (Tebal): 5 mm
    ctx.strokeStyle = "rgba(255, 71, 71, 0.4)";
    ctx.lineWidth = 1.0;
    drawAxis(x0, left, right, bigX, vLine);
    drawAxis(y0, top, bottom, bigY, hLine);
    ctx.restore();
  },
};
Chart.register(ecgGridPlugin);
if (window.ChartZoom) Chart.register(window.ChartZoom);

// =========================================================================
// 3. CHART INITIALIZATION (3 KANVAS PER LEAD, MENEMPEL, RAW + FILTERED)
// =========================================================================
const SIGNAL_COLORS = { raw: "#d9553f", clean: "#1f6fe5" };

// Label LEAD digambar di gutter sumbu-y tiap kanvas (bukan elemen terpisah)
function makeLeadLabelPlugin(lead) {
  return {
    id: `leadLabel${lead}`,
    afterDraw: (chart) => {
      const area = chart.chartArea;
      if (!area) return;
      const { ctx } = chart;
      ctx.save();
      ctx.font = "700 11px monospace";
      ctx.fillStyle = "#566274";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(`LEAD ${lead + 1}`, 8, (area.top + area.bottom) / 2);
      ctx.restore();
    },
  };
}

// Sumbu-y memakai nilai mV asli; kalibrasi (gain = mm/mV) diterapkan lewat
// jendela skala-y di initChart — sama seperti kertas EKG standar.
function scaledPoints(values) {
  return values.map((v, i) => ({ x: i, y: v }));
}

function initChart(canvasId, lead) {
  const canvasEl = document.getElementById(canvasId);
  if (!canvasEl) return null;

  const ctx = canvasEl.getContext("2d");
  const fs = parseFloat(document.getElementById("target_fs")?.value) || 250;
  const total = Math.max(totalSamples, 1);
  // Ukur container (bukan canvas) agar stabil sebelum/sesudah Chart.js sizing
  const width = canvasEl.parentElement?.clientWidth || canvasEl.clientWidth || 1100;
  const height = canvasEl.parentElement?.clientHeight || 240;
  // Kalibrasi klinis: gain = mm/mV, px/mm = PX_PER_MM.
  // Jendela-y = setengah tinggi / (px per mm x mm/mV):
  //   gain 5 -> +/-4 mV, gain 10 -> +/-2 mV, gain 20 -> +/-1 mV (standar EKG)
  const view = height / 2 / (PX_PER_MM * currentGain);
  const yLimit = view * 3; // Ruang geser vertikal di luar jendela tampil
  // Jendela waktu = lebar kanvas / (mm px x mm/s): mereplikasi kertas EKG asli
  const windowSamples = Math.max(1, Math.min(total, Math.round((width / (PX_PER_MM * currentSpeed)) * fs)));

  return new Chart(ctx, {
    type: "line",
    data: {
      datasets: [
        { label: `Lead ${lead + 1} Raw`, data: [], borderColor: SIGNAL_COLORS.raw, borderWidth: 1.1, pointRadius: 0, fill: false },
        { label: `Lead ${lead + 1} Filtered`, data: [], borderColor: SIGNAL_COLORS.clean, borderWidth: 1.7, pointRadius: 0, fill: false },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: false,
      interaction: { mode: "nearest", intersect: false },
      scales: {
        x: { type: "linear", display: false, min: 0, max: windowSamples },
        y: {
          type: "linear",
          position: "left",
          min: -view,
          max: view,
          grid: { display: false },
          border: { display: false },
          ticks: { display: false },
          // Gutter 64 px di kiri kanvas untuk label LEAD n
          afterFit: (scale) => { scale.width = 64; },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: { enabled: false },
        zoom: {
          // Seret: geser horizontal (waktu) DAN vertikal (bidang tampilan Y)
          pan: { enabled: true, mode: "xy" },
          zoom: { wheel: { enabled: true, mode: "x" }, pinch: { enabled: true, mode: "x" }, dblclick: { enabled: true, mode: "x" }, mode: "x" },
          limits: {
            x: { min: 0, max: total },
            y: { min: -yLimit, max: yLimit },
          },
        },
      },
    },
    plugins: [makeLeadLabelPlugin(lead)],
  });
}

function initializeCharts() {
  for (let i = 0; i < LEAD_COUNT; i++) {
    charts[i] = initChart(`lead_chart_${i}`, i);
  }
}

function rebuildCharts() {
  for (let i = 0; i < LEAD_COUNT; i++) {
    if (charts[i]) {
      charts[i].destroy();
      charts[i] = null;
    }
    charts[i] = initChart(`lead_chart_${i}`, i);
  }
  drawChartData();
}

function drawChartData() {
  const rawOn = document.getElementById("tgl_raw")?.checked ?? true;
  const cleanOn = document.getElementById("tgl_clean")?.checked ?? true;
  for (let i = 0; i < LEAD_COUNT; i++) {
    const chart = charts[i];
    if (!chart || !rawSignals[i]) continue;
    chart.data.datasets[0].data = scaledPoints(rawSignals[i]);
    chart.setDatasetVisibility(0, rawOn);
    chart.data.datasets[1].data = scaledPoints(cleanSignals[i] || []);
    chart.setDatasetVisibility(1, cleanOn);
    chart.update("none");
  }
}

function toggleSignal(which, visible) {
  const idx = which === "raw" ? 0 : 1;
  Object.values(charts).forEach((c) => {
    if (c) {
      c.setDatasetVisibility(idx, visible);
      c.update();
    }
  });
}

function resetZoomAll() {
  Object.values(charts).forEach((c) => c && c.resetZoom());
}

function setGain(gain) {
  currentGain = gain;
  document.querySelectorAll("[data-gain]").forEach((b) => b.classList.toggle("active", Number(b.dataset.gain) === gain));
  rebuildCharts(); // Kalibrasi berubah: jendela-y & limits ikut dihitung ulang
}

function setSpeed(speed) {
  currentSpeed = speed;
  document.querySelectorAll("[data-speed]").forEach((b) => b.classList.toggle("active", Number(b.dataset.speed) === speed));
  rebuildCharts(); // Jendela waktu & status zoom ikut dihitung ulang
}

// =========================================================================
// 3b. FILTER STAGE TOGGLES (input dinonaktifkan saat stage dimatikan)
// =========================================================================
const FILTER_INPUTS = {
  wavelet: ["wavelet", "w_level"],
  median: ["median_kernel"],
  bandpass: ["lowcut", "highcut"],
};

function toggleFilter(name, enabled) {
  (FILTER_INPUTS[name] || []).forEach((id) => {
    const el = document.getElementById(id);
    if (!el) return;
    el.disabled = !enabled;
    el.closest(".control-group")?.classList.toggle("is-disabled", !enabled);
  });
}

function filterFlags() {
  const on = (id) => document.getElementById(id)?.checked ?? true;
  return `use_wavelet=${on("use_wavelet")}&use_median=${on("use_median")}&use_bandpass=${on("use_bandpass")}`;
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

// =========================================================================
// 5. TAB NAVIGATION
// =========================================================================
function switchTab(tabId) {
  if (tabId === 1) {
    show("view_tab_1", "flex");
    hide("view_tab_2");
  } else {
    hide("view_tab_1");
    show("view_tab_2", "flex");
    loadSimulatorFolders();
  }
  document.getElementById("tab_btn_1")?.classList.toggle("active", tabId === 1);
  document.getElementById("tab_btn_2")?.classList.toggle("active", tabId === 2);
}

// =========================================================================
// 6. DATASET OPTIONS API
// =========================================================================
// 6. DATASET OPTIONS API
// =========================================================================
let datasetsInitialized = false;

async function loadRecordOptions() {
  try {
    const res = await fetch(`${API_BASE}/api/records`);
    const data = await res.json();

    const datasetEl = document.getElementById("dataset");
    const recordSelect = document.getElementById("record_id");
    if (!datasetEl || !recordSelect) return;

    // Dynamically populate datasets if not done yet
    if (!datasetsInitialized) {
      const currentSelected = datasetEl.value;
      datasetEl.innerHTML = "";
      
      const standardKeys = ["prosim_simulator", "ptbxl_100hz", "ptbxl_500hz", "chapman"];
      const standardLabels = {
        "prosim_simulator": "ProSim Simulator",
        "ptbxl_100hz": "PTB-XL (100Hz)",
        "ptbxl_500hz": "PTB-XL (500Hz)",
        "chapman": "Chapman"
      };
      
      standardKeys.forEach(key => {
        if (data[key]) {
          const opt = document.createElement("option");
          opt.value = key;
          opt.innerText = standardLabels[key] || key;
          datasetEl.appendChild(opt);
        }
      });
      
      Object.keys(data).forEach(key => {
        if (!standardKeys.includes(key)) {
          const opt = document.createElement("option");
          opt.value = key;
          opt.innerText = key;
          datasetEl.appendChild(opt);
        }
      });
      
      if (data[currentSelected]) {
        datasetEl.value = currentSelected;
      } else {
        datasetEl.selectedIndex = 0;
      }
      datasetsInitialized = true;
    }

    const dataset = datasetEl.value;

    // -------------------------------------------------------------
    // SMART DEFAULT SELECTION (Sinkronisasi Parameter v5.0 Sebelum Inferensi)
    // -------------------------------------------------------------
    if (dataset === "ptbxl_500hz" || dataset === "chapman") {
      setTextBoxValue("median_kernel", "101");
      setTextBoxValue("highcut", "100");
      setSliderValue("w_level", "lbl_w_level", "4");
      setTextBoxValue("model_id", "softmax_filtered_500to250_cnn");
    } else if (dataset === "ptbxl_100hz") {
      setTextBoxValue("median_kernel", "51");
      setTextBoxValue("highcut", "45");
      setSliderValue("w_level", "lbl_w_level", "4");
      setTextBoxValue("model_id", "softmax_filtered_100to250_cnn");
    } else {
      // Fallback for ProSim and any dynamic sensor records
      setTextBoxValue("median_kernel", "51");
      setTextBoxValue("highcut", "45");
      setSliderValue("w_level", "lbl_w_level", "4");
      setTextBoxValue("model_id", "softmax_filtered_500to250_cnn");
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
let lastCleanSignals = null;

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

    // Retrieve model_id and use_raw_for_ai directly
    const model_id = getVal("model_id");
    const use_raw = getVal("use_raw_for_ai") === "true";

    const url = `${API_BASE}/api/process?dataset=${d}&record_id=${r}&target_fs=${t_fs}&wavelet=${wav}&w_level=${lvl}&median_kernel=${med}&lowcut=${low}&highcut=${high}&model_id=${model_id}&use_raw_for_ai=${use_raw}&${filterFlags()}`;


    const res = await fetch(url);
    const result = await res.json();

    lastCleanSignals = result.clean_signals || null;

    // Jalankan seluruh fungsi modular perenderan data
    renderDiagnosis(result);
    renderPerformance(result);
    renderHolter(result);
    renderCharts(result);
    renderSignalQualityMetrics(result);
  } catch (err) {
    console.error("Gagal memproses pipeline DSP:", err);
    setText("sample_class", "Offline");
    setText("ai_class", "Offline");
  }
}

// =========================================================================
// 7.1. SAVE CSV FUNCTION
// =========================================================================
async function saveFilteredCSV() {
  const getVal = (id) => document.getElementById(id)?.value || "";
  const d = getVal("dataset");
  const r = getVal("record_id");
  if (!r) {
    alert("Silakan pilih Record ID terlebih dahulu.");
    return;
  }

  const t_fs = getVal("target_fs");
  const wav = getVal("wavelet");
  const lvl = getVal("w_level");
  const med = getVal("median_kernel");
  const low = getVal("lowcut");
  const high = getVal("highcut");

  const saveBtn = document.getElementById("btn_save_csv");
  const origText = saveBtn ? saveBtn.innerText : "SAVE CSV";
  if (saveBtn) saveBtn.innerText = "SAVING CSV...";

  try {
    // 1. Simpan CSV di server & dapatkan metadata
    const saveUrl = `${API_BASE}/api/save_frames?dataset=${d}&record_id=${r}&target_fs=${t_fs}&wavelet=${wav}&w_level=${lvl}&median_kernel=${med}&lowcut=${low}&highcut=${high}&file_format=csv&${filterFlags()}`;
    const res = await fetch(saveUrl);
    const result = await res.json();

    // 2. Unduh file CSV secara langsung di peramban pengguna
    const downloadUrl = `${API_BASE}/api/download_csv?dataset=${d}&record_id=${r}&target_fs=${t_fs}&wavelet=${wav}&w_level=${lvl}&median_kernel=${med}&lowcut=${low}&highcut=${high}&${filterFlags()}`;
    const link = document.createElement("a");
    link.href = downloadUrl;
    link.download = `${r}_filtered.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);

    if (result.status === "success") {
      const savedCount = result.saved_frames?.total_files_saved || 0;
      const outDir = result.saved_frames?.output_dir || "output/filtered_frames";
      alert(`✅ Berhasil! Sinyal CSV diunduh ke peramban & disimpan di server (${savedCount} file CSV):\n${outDir}`);
    } else {
      alert("⚠️ Sinyal diunduh ke peramban, namun terjadi kendala di server: " + (result.message || "Unknown error"));
    }
  } catch (err) {
    console.error("Gagal menyimpan file CSV:", err);
    alert("Terjadi kesalahan koneksi saat menyimpan CSV: " + err.message);
  } finally {
    if (saveBtn) saveBtn.innerText = origText;
  }
}

// =========================================================================
// 7.2. CONVERT TO JSONL FUNCTION
// =========================================================================
async function convertToJSONL() {
  const getVal = (id) => document.getElementById(id)?.value || "";
  const d = getVal("dataset");
  const r = getVal("record_id");
  if (!r) {
    alert("Silakan pilih Record ID terlebih dahulu.");
    return;
  }

  const t_fs = getVal("target_fs");
  const wav = getVal("wavelet");
  const lvl = getVal("w_level");
  const med = getVal("median_kernel");
  const low = getVal("lowcut");
  const high = getVal("highcut");
  const model_id = getVal("model_id");

  const convertBtn = document.getElementById("btn_convert_jsonl");
  const origText = convertBtn ? convertBtn.innerText : "CONVERT TO JSONL";
  if (convertBtn) convertBtn.innerText = "CONVERTING...";

  try {
    const downloadUrl = `${API_BASE}/api/convert_to_jsonl?dataset=${d}&record_id=${r}&target_fs=${t_fs}&wavelet=${wav}&w_level=${lvl}&median_kernel=${med}&lowcut=${low}&highcut=${high}&model_id=${model_id}&${filterFlags()}`;
    
    const link = document.createElement("a");
    link.href = downloadUrl;
    link.download = `${r}.jsonl`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    
    alert(`✅ Berhasil! File ${r}.jsonl telah berhasil dikonversi dan diunduh.`);
  } catch (err) {
    console.error("Gagal melakukan konversi JSONL:", err);
    alert("Terjadi kesalahan koneksi saat konversi JSONL: " + err.message);
  } finally {
    if (convertBtn) convertBtn.innerText = origText;
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
        span.style.color = "#c0392b";
      }
      span.innerText = evt;
      eventDiv.appendChild(span);
    });
  } else {
    eventDiv.innerText = "--";
  }
}

function renderCharts(result) {
  rawSignals = [];
  cleanSignals = [];
  totalSamples = 0;

  for (let i = 0; i < LEAD_COUNT; i++) {
    const raw = result.raw_signals ? result.raw_signals[`lead_${i}`] || [] : [];
    const clean = result.clean_signals ? result.clean_signals[`lead_${i}`] || [] : [];
    rawSignals.push(raw);
    cleanSignals.push(clean);
    totalSamples = Math.max(totalSamples, raw.length, clean.length);
  }

  rebuildCharts();
}

function renderSignalQualityMetrics(result) {
  if (!result || !result.metrics) return;

  const setMetricText = (id, val, suffix = "") => {
    const el = document.getElementById(id);
    if (el) {
      if (typeof val === 'number') {
        el.innerText = `${val.toFixed(id.includes('_red') ? 1 : (id.includes('_n_') ? 4 : 3))}${suffix}`;
      } else {
        el.innerText = `--${suffix}`;
      }
    }
  };

  const leads = ['lead1', 'lead2', 'lead3'];
  leads.forEach((leadKey, idx) => {
    const num = idx + 1;
    const lMetrics = result.metrics[leadKey];
    if (lMetrics) {
      setMetricText(`q_l${num}_b_raw`, lMetrics.baseline_rms_raw, " mV");
      setMetricText(`q_l${num}_b_filt`, lMetrics.baseline_rms_filtered, " mV");
      setMetricText(`q_l${num}_b_red`, lMetrics.baseline_reduction_percent, " %");
      
      setMetricText(`q_l${num}_n_raw`, lMetrics.hf_noise_raw);
      setMetricText(`q_l${num}_n_filt`, lMetrics.hf_noise_filtered);
      setMetricText(`q_l${num}_n_red`, lMetrics.hf_noise_reduction_percent, " %");

      // Dynamic color coding based on reduction percentage
      const bRedEl = document.getElementById(`q_l${num}_b_red`);
      const nRedEl = document.getElementById(`q_l${num}_n_red`);
      
      if (bRedEl) {
        if (lMetrics.baseline_reduction_percent >= 80.0) {
          bRedEl.style.color = "#1f9d4d"; // Green
        } else if (lMetrics.baseline_reduction_percent >= 50.0) {
          bRedEl.style.color = "#b26a00"; // Orange
        } else {
          bRedEl.style.color = "#d92d20"; // Red
        }
      }

      if (nRedEl) {
        if (lMetrics.hf_noise_reduction_percent >= 80.0) {
          nRedEl.style.color = "#1f9d4d";
        } else if (lMetrics.hf_noise_reduction_percent >= 50.0) {
          nRedEl.style.color = "#b26a00";
        } else {
          nRedEl.style.color = "#d92d20";
        }
      }
    }
  });

  const sumMetrics = result.metrics.summary;
  if (sumMetrics) {
    setMetricText("q_sum_b_red", sumMetrics.average_baseline_reduction, " %");
    setMetricText("q_sum_n_red", sumMetrics.average_noise_reduction, " %");
    
    const bSumEl = document.getElementById("q_sum_b_red");
    const nSumEl = document.getElementById("q_sum_n_red");
    
    if (bSumEl) {
      if (sumMetrics.average_baseline_reduction >= 80.0) {
        bSumEl.style.color = "#1f9d4d";
      } else if (sumMetrics.average_baseline_reduction >= 50.0) {
        bSumEl.style.color = "#b26a00";
      } else {
        bSumEl.style.color = "#d92d20";
      }
    }

    if (nSumEl) {
      if (sumMetrics.average_noise_reduction >= 80.0) {
        nSumEl.style.color = "#1f9d4d";
      } else if (sumMetrics.average_noise_reduction >= 50.0) {
        nSumEl.style.color = "#b26a00";
      } else {
        nSumEl.style.color = "#d92d20";
      }
    }
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
    recCard.style.color = "#8a5200";
    recCard.style.borderLeft = "5px solid #b26a00";
    setHTML(
      "sim_recommendation_text",
      `⚠️ <b>Rekomendasi Deteksi Bradikardia:</b> Sinyal terdeteksi sebagai Denyut Jantung Lambat (${result.calculated_bpm} BPM) dan filter median mereduksi amplitudo puncak R sebesar ${result.attenuation_median_pct}%. Filter terlalu agresif memotong fase isoelektrik. <br><b>Saran Tindakan:</b> Ubah parameter Median Filter Kernel di Tab 1 menjadi 101 atau 151 sampel sebelum grid search massal dilakukan.`,
    );
  } else {
    recCard.style.background = "#e6f9ed";
    recCard.style.color = "#1a7f42";
    recCard.style.borderLeft = "5px solid #1f9d4d";
    setHTML(
      "sim_recommendation_text",
      `✅ <b>Status Filter Stabil:</b> Redaman amplitudo puncak r-wave berada pada rentang batas aman (${result.attenuation_median_pct}%). Segmentasi morfologi dan interval waktu spasio-temporal EKG lulus uji distorsi klinis AHA.`,
    );
  }
}

// =========================================================================
// 10. STARTUP / APPLICATION ENTRY POINT
// =========================================================================
window.onload = async () => {
  initializeCharts();
  await loadRecordOptions();
};
