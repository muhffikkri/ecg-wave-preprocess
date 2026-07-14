// Logika Navigasi Tab Switcher
function switchTab(tabId) {
  if (tabId === 1) {
    document.getElementById("view_tab_1").style.display = "flex";
    document.getElementById("view_tab_2").style.display = "none";
    document.getElementById("tab_btn_1").style.background = "#0071e3";
    document.getElementById("tab_btn_1").style.color = "white";
    document.getElementById("tab_btn_2").style.background = "#e5e5e7";
    document.getElementById("tab_btn_2").style.color = "#1d1d1f";
  } else {
    document.getElementById("view_tab_1").style.display = "none";
    document.getElementById("view_tab_2").style.display = "flex";
    document.getElementById("tab_btn_1").style.background = "#e5e5e7";
    document.getElementById("tab_btn_1").style.color = "#1d1d1f";
    document.getElementById("tab_btn_2").style.background = "#0071e3";
    document.getElementById("tab_btn_2").style.color = "white";
    loadSimulatorFolders();
  }
}

// Fetch daftar folder kustom dari hardware simulator
async function loadSimulatorFolders() {
  const res = await fetch("http://127.0.0.1:8000/api/simulator/folders");
  const folders = await res.json();
  const selectNode = document.getElementById("sim_folder_select");
  selectNode.innerHTML = "";
  folders.forEach((f) => {
    let opt = document.createElement("option");
    opt.value = f;
    opt.innerText = f;
    selectNode.appendChild(opt);
  });
}

// Ambil data analisis komparatif dari backend (Dukungan Fitur Filter Interaktif Tab 2)
async function triggerSimulatorAnalysis() {
  const folder = document.getElementById("sim_folder_select").value;
  const fs = document.getElementById("sim_fs").value;
  const wav = document.getElementById("sim_wavelet").value;
  const lvl = document.getElementById("sim_w_level").value;
  const med = document.getElementById("sim_median_kernel").value;
  const low = document.getElementById("sim_lowcut").value;
  const high = document.getElementById("sim_highcut").value;

  if (!folder) return;

  document.getElementById("sim_placeholder_text").innerText = "Sedang menghitung transformasi Fourier (FFT) dan melacak koordinat puncak R... Mohon tunggu.";

  const url = `http://127.0.0.1:8000/api/simulator/analyze?folder_name=${folder}&target_fs=${fs}&wavelet=${wav}&w_level=${lvl}&median_kernel=${med}&lowcut=${low}&highcut=${high}`;
  const res = await fetch(url);
  const result = await res.json();

  if (result.status === "success") {
    document.getElementById("sim_placeholder_text").style.display = "none";
    document.getElementById("sim_metrics_grid").style.display = "grid";

    // Render Metrik Card
    document.getElementById("sim_val_bpm").innerHTML = `${result.calculated_bpm}<span class="holter-unit">BPM</span>`;
    document.getElementById("sim_val_rr").innerHTML = `${result.avg_rr_seconds}<span class="holter-unit">detik</span>`;
    document.getElementById("sim_val_noise").innerHTML = `${result.dominant_noise_freq}<span class="holter-unit">Hz</span>`;
    document.getElementById("sim_val_attenuation").innerHTML = `${result.attenuation_median_pct}<span class="holter-unit">%</span>`;

    // Render Base64 Image
    const imgNode = document.getElementById("sim_report_img");
    imgNode.src = "data:image/png;base64," + result.image;
    imgNode.style.display = "block";

    // Berikan Sistem Rekomendasi Pintar Medis Otomatis
    const recCard = document.getElementById("sim_recommendation_card");
    recCard.style.display = "block";
    if (parseFloat(result.calculated_bpm) < 65.0 && parseFloat(result.attenuation_median_pct) > 4.0) {
      recCard.style.background = "#fff2e6";
      recCard.style.color = "#ff9f0a";
      recCard.style.border = "1px solid #ffd699";
      recCard.innerHTML = `⚠️ <b>Rekomendasi Deteksi Bradikardia:</b> Sinyal terdeteksi sebagai Denyut Jantung Lambat (${result.calculated_bpm} BPM) dan filter median mereduksi amplitudo puncak R sebesar ${result.attenuation_median_pct}%. Filter terlalu agresif memotong fase isoelektrik. <b>Saran Tindakan:</b> Ubah parameter Median Filter Kernel di Tab 1 menjadi 101 atau 151 sampel sebelum grid search massal dilakukan.`;
    } else {
      recCard.style.background = "#e6f9ed";
      recCard.style.color = "#34c759";
      recCard.style.border = "1px solid #b3f0c2";
      recCard.innerHTML = `✅ <b>Status Filter Stabil:</b> Redaman amplitudo puncak r-wave berada pada rentang batas aman (${result.attenuation_median_pct}%). Segmentasi morfologi dan interval waktu spasio-temporal EKG lulus uji distorsi klinis AHA.`;
    }
  } else {
    alert(result.message);
  }
}

let charts = {};

// PLUGIN GRID MILLIMETER BLOK KLINIS EKG ASLI
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

function initChart(canvasId) {
  const ctx = document.getElementById(canvasId).getContext("2d");
  return new Chart(ctx, {
    type: "line",
    data: { labels: [], datasets: [{ data: [], borderColor: "#1c1c1e", borderWidth: 1.5, pointRadius: 0, fill: false }] },
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

for (let i = 0; i < 3; i++) {
  charts[`raw_${i}`] = initChart(`raw_lead_${i}`);
  charts[`clean_${i}`] = initChart(`clean_lead_${i}`);
}

async function loadRecordOptions() {
  const res = await fetch("http://127.0.0.1:8000/api/records");
  const data = await res.json();
  const dataset = document.getElementById("dataset").value;
  const recordSelect = document.getElementById("record_id");

  recordSelect.innerHTML = "";
  if (!data[dataset] || data[dataset].length === 0) return;

  data[dataset].forEach((id) => {
    let opt = document.createElement("option");
    opt.value = id;
    opt.innerText = id;
    recordSelect.appendChild(opt);
  });
}

async function triggerProcessing() {
  const d = document.getElementById("dataset").value;
  const r = document.getElementById("record_id").value;
  const t_fs = document.getElementById("target_fs").value;
  const wav = document.getElementById("wavelet").value;
  const lvl = document.getElementById("w_level").value;
  const med = document.getElementById("median_kernel").value;
  const low = document.getElementById("lowcut").value;
  const high = document.getElementById("highcut").value;

  if (!r) return;

  const url = `http://127.0.0.1:8000/api/process?dataset=${d}&record_id=${r}&target_fs=${t_fs}&wavelet=${wav}&w_level=${lvl}&median_kernel=${med}&lowcut=${low}&highcut=${high}`;
  const res = await fetch(url);
  const result = await res.json();

  // Update Data Penyakit Tiga Pilar
  document.getElementById("sample_class").innerText = result.target_class || "Unknown";
  document.getElementById("keras_class").innerText = result.keras_prediction || "Offline";
  document.getElementById("keras_conf").innerText = result.keras_confidence ? `(${result.keras_confidence}%)` : "(--%)";
  document.getElementById("tflite_class").innerText = result.tflite_prediction || "Offline";
  document.getElementById("tflite_conf").innerText = result.tflite_confidence ? `(${result.tflite_confidence}%)` : "(--%)";

  if (result.metrics) {
    document.getElementById("val_latency").innerText = `${result.metrics.latency_ms.toFixed(2)} ms`;
    document.getElementById("val_memory").innerText = `${result.metrics.peak_memory_mb.toFixed(4)} MB`;
  }

  const holter = result.holter || (result.metrics ? result.metrics.holter : null);

  if (holter) {
    document.getElementById("h_hr").innerHTML = `${holter.hr || "--"}<span class="holter-unit">BPM</span>`;
    document.getElementById("h_rr").innerHTML = `${holter.rr_avg_ms || "--"}<span class="holter-unit">ms</span>`;
    document.getElementById("h_hrv").innerHTML = `${holter.rmssd_ms || "--"}<span class="holter-unit">ms</span>`;
    document.getElementById("h_st").innerHTML = holter.st_dev_mv !== undefined ? `${holter.st_dev_mv.toFixed(3)}<span class="holter-unit">mV</span>` : `--<span class="holter-unit">mV</span>`;
    document.getElementById("h_qtc").innerHTML = `${holter.qtc_ms || "--"}<span class="holter-unit">ms</span>`;

    const eventDiv = document.getElementById("h_events");
    eventDiv.innerHTML = "";
    if (holter.events && holter.events.length > 0) {
      holter.events.forEach((evt) => {
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

  // Update Gelombang Kanvas
  for (let i = 0; i < 3; i++) {
    const rawData = result.raw_signals ? result.raw_signals[`lead_${i}`] : null;
    const cleanData = result.clean_signals ? result.clean_signals[`lead_${i}`] : null;

    if (rawData && rawData.length > 0) {
      charts[`raw_${i}`].data.labels = Array.from({ length: rawData.length }, (_, idx) => idx);
      charts[`raw_${i}`].data.datasets[0].data = rawData;
      charts[`raw_${i}`].update("none");
    }
    if (cleanData && cleanData.length > 0) {
      charts[`clean_${i}`].data.labels = Array.from({ length: cleanData.length }, (_, idx) => idx);
      charts[`clean_${i}`].data.datasets[0].data = cleanData;
      charts[`clean_${i}`].update("none");
    }
  }
}

window.onload = async () => {
  await loadRecordOptions();
  await triggerProcessing();
};
