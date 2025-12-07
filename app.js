// --------- Tab navigation ----------
document.addEventListener("DOMContentLoaded", () => {
  const tabButtons = document.querySelectorAll(".tab-button");
  const tabContents = document.querySelectorAll(".tab-content");

  tabButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetId = btn.getAttribute("data-tab");

      tabButtons.forEach((b) => b.classList.remove("active"));
      tabContents.forEach((c) => c.classList.remove("active"));

      btn.classList.add("active");
      const target = document.getElementById(targetId);
      if (target) target.classList.add("active");
    });
  });

  setupPredictionForm();
  setupEDA();
});

// --------- Job Check logic ----------
function setupPredictionForm() {
  const form = document.getElementById("job-form");
  if (!form) return;

  const predictButton = document.getElementById("predict-button");
  const statusEl = document.getElementById("predict-status");
  const resultEl = document.getElementById("predict-result");
  const messageEl = document.getElementById("predict-message");
  const probEl = document.getElementById("predict-prob");
  const errorEl = document.getElementById("predict-error");

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    errorEl.classList.add("hidden");
    resultEl.classList.add("hidden");

    const formData = new FormData(form);

    const title = (formData.get("title") || "").toString();
    const company_profile = (formData.get("company_profile") || "").toString();
    const description = (formData.get("description") || "").toString();
    const requirements = (formData.get("requirements") || "").toString();
    const benefits = (formData.get("benefits") || "").toString();
    const location = (formData.get("location") || "").toString();
    const salary_range = (formData.get("salary_range") || "").toString();
    const employment_type = (formData.get("employment_type") || "").toString();
    const industry = (formData.get("industry") || "").toString();

    const full_text =
      title +
      " " +
      company_profile +
      " " +
      description +
      " " +
      requirements +
      " " +
      benefits +
      " " +
      location +
      " " +
      salary_range +
      " " +
      employment_type +
      " " +
      industry;

    const payload = {
      full_text: full_text
    };

    predictButton.disabled = true;
    statusEl.textContent = "Sending request...";
    try {
      const response = await fetch("/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify(payload)
      });

      if (!response.ok) {
        throw new Error("Server returned status " + response.status);
      }

      const data = await response.json();
      // Ожидаем объект вида { fraud_proba: 0.0 - 1.0 } или { probability: ... }
      const probaRaw =
        typeof data.fraud_proba === "number"
          ? data.fraud_proba
          : typeof data.probability === "number"
          ? data.probability
          : null;

      if (probaRaw === null || isNaN(probaRaw)) {
        throw new Error("Unexpected response format");
      }

      const fraudProba = Math.min(Math.max(probaRaw, 0), 1);
      const fraudPct = (fraudProba * 100).toFixed(1);
      const legitPct = (100 - fraudProba * 100).toFixed(1);

      resultEl.classList.remove("hidden");
      if (fraudProba < 0.5) {
        resultEl.classList.remove("danger");
        resultEl.classList.add("success");
        messageEl.textContent =
          "This job posting appears legitimate.";
        probEl.textContent =
          "Fraud probability: " +
          fraudPct +
          "% · Legitimate probability: " +
          legitPct +
          "%";
      } else {
        resultEl.classList.remove("success");
        resultEl.classList.add("danger");
        messageEl.textContent =
          "Warning: high fraud probability.";
        probEl.textContent =
          "Fraud probability: " +
          fraudPct +
          "% · Legitimate probability: " +
          legitPct +
          "%";
      }

      statusEl.textContent = "Prediction received.";
    } catch (err) {
      console.error(err);
      errorEl.textContent =
        "Network or server error while calling /predict. Please try again later.";
      errorEl.classList.remove("hidden");
      statusEl.textContent = "Error.";
    } finally {
      predictButton.disabled = false;
      setTimeout(() => {
        statusEl.textContent = "";
      }, 2000);
    }
  });
}

// --------- EDA logic ----------
let fraudChart = null;
let lengthChart = null;

function setupEDA() {
  const csvPath = "data/combined_dataset_clean_sample.csv";

  const edaError = document.getElementById("eda-error");
  const totalEl = document.getElementById("total-count");
  const realEl = document.getElementById("real-count");
  const fraudEl = document.getElementById("fraud-count");

  if (!window.Papa) {
    if (edaError) {
      edaError.textContent =
        "PapaParse failed to load. Check CDN connection.";
      edaError.classList.remove("hidden");
    }
    return;
  }

  Papa.parse(csvPath, {
    download: true,
    header: true,
    skipEmptyLines: true,
    complete: function (results) {
      const rows = results.data || [];
      if (!rows.length) {
        if (edaError) {
          edaError.textContent =
            "Dataset file loaded but contains no rows.";
          edaError.classList.remove("hidden");
        }
        return;
      }

      // fraud counts
      let countReal = 0;
      let countFraud = 0;

      // length buckets: short < 300, 300-800, > 800 chars of description
      let shortCount = 0;
      let mediumCount = 0;
      let longCount = 0;

      rows.forEach((row) => {
        const fraudVal = row["fraudulent"];
        if (fraudVal === "1" || fraudVal === 1) {
          countFraud += 1;
        } else {
          countReal += 1;
        }

        const desc = (row["description"] || "").toString();
        const len = desc.length;
        if (len < 300) {
          shortCount += 1;
        } else if (len < 800) {
          mediumCount += 1;
        } else {
          longCount += 1;
        }
      });

      const total = countReal + countFraud;
      if (totalEl) totalEl.textContent = total.toString();
      if (realEl) realEl.textContent = countReal.toString();
      if (fraudEl) fraudEl.textContent = countFraud.toString();

      renderFraudChart(countReal, countFraud);
      renderLengthChart(shortCount, mediumCount, longCount);
    },
    error: function (error) {
      console.error(error);
      if (edaError) {
        edaError.textContent =
          "Failed to load dataset CSV. Ensure data/combined_dataset_clean_sample.csv exists.";
        edaError.classList.remove("hidden");
      }
    }
  });
}

function renderFraudChart(realCount, fraudCount) {
  const ctx = document.getElementById("fraud-chart");
  if (!ctx || !window.Chart) return;

  if (fraudChart) fraudChart.destroy();

  fraudChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Legitimate (0)", "Fraudulent (1)"],
      datasets: [
        {
          label: "Number of postings",
          data: [realCount, fraudCount],
          backgroundColor: ["#22c55e", "#f97373"]
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => ctx.raw + " postings"
          }
        }
      },
      scales: {
        x: {
          ticks: {
            color: "#9ca3af"
          }
        },
        y: {
          ticks: {
            color: "#9ca3af",
            precision: 0
          },
          beginAtZero: true
        }
      }
    }
  });
}

function renderLengthChart(shortCount, mediumCount, longCount) {
  const ctx = document.getElementById("length-chart");
  if (!ctx || !window.Chart) return;

  if (lengthChart) lengthChart.destroy();

  lengthChart = new Chart(ctx, {
    type: "bar",
    data: {
      labels: ["Short (<300 chars)", "Medium (300–800)", "Long (>800)"],
      datasets: [
        {
          label: "Description length buckets",
          data: [shortCount, mediumCount, longCount],
          backgroundColor: ["#38bdf8", "#0ea5e9", "#0369a1"]
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: (ctx) => ctx.raw + " postings"
          }
        }
      },
      scales: {
        x: {
          ticks: {
            color: "#9ca3af"
          }
        },
        y: {
          ticks: {
            color: "#9ca3af",
            precision: 0
          },
          beginAtZero: true
        }
      }
    }
  });
}
