/* eslint-disable no-console */
const express = require("express");
const axios = require("axios");

// ================== CONFIG ===================
const HOST = "0.0.0.0";
const PORT = process.env.PORT || 8000;

const POLL_INTERVAL = 5 * 1000; // ms
const RETRY_DELAY = 5 * 1000;   // ms
const MAX_HISTORY = 200;
const MAX_FEATURES = 16;

// ML hyperparams
const LR = 0.08;   // learning rate
const L2 = 1e-4;   // L2 regularization

// ================== STORES ===================
const latestResult100 = {
  Phien_truoc: 0, Xuc_xac: "0-0-0", Tong: 0, Ket_qua: "Chưa có",
  Phien_sau: 0, Du_doan: "", Do_tin_cay: 0, Giai_thich: "",
  id: "Tele@idol_vannhat"
};
const latestResult101 = { ...latestResult100 };

const history100 = []; // [{Phien, d:[d1,d2,d3], Tong, Ket_qua}]
const history101 = [];

let lastSid100 = null;
let lastSid101 = null;
let sidForTX = null;
let lastConfidence100 = null;
let lastConfidence101 = null;

// ================ HELPERS ====================
function getTaiXiu(d1, d2, d3) {
  const total = d1 + d2 + d3;
  return total <= 10 ? "Xỉu" : "Tài";
}
function txToInt(kq) {
  return kq === "Tài" ? 1 : 0;
}
function safeMean(arr) {
  return arr.length ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
}
function streakLen(hist) {
  if (!hist.length) return 0;
  const first = hist[0].Ket_qua;
  let n = 0;
  for (const h of hist) {
    if (h.Ket_qua === first) n++;
    else break;
  }
  return n;
}

// ============ Online Logistic Regression =====
class OnlineLogReg {
  constructor(dim, lr = LR, l2 = L2) {
    this.dim = dim;
    this.lr = lr;
    this.l2 = l2;
    this.w = Array(dim).fill(0);
  }
  _sigmoid(z) {
    if (z >= 0) {
      const ez = Math.exp(-z);
      return 1 / (1 + ez);
    } else {
      const ez = Math.exp(z);
      return ez / (1 + ez);
    }
  }
  predictProba(x) {
    const z = this.w.reduce((acc, wi, i) => acc + wi * x[i], 0);
    return this._sigmoid(z);
  }
  update(x, y) {
    const p = this.predictProba(x);
    const g = p - y; // gradient for logistic loss
    for (let i = 0; i < this.dim; i++) {
      this.w[i] -= this.lr * (g * x[i] + this.l2 * this.w[i]);
    }
  }
}
const model100 = new OnlineLogReg(MAX_FEATURES);
const model101 = new OnlineLogReg(MAX_FEATURES);

// pending feature vector used for previous prediction, to train when label arrives
let pendingSample100 = null; // {x, nextPhien}
let pendingSample101 = null;

// ============== FEATURE ENGINEERING ==========
function buildFeatures(hist) {
  const totals = hist.map(h => h.Tong);
  const res = hist.map(h => h.Ket_qua);

  const lastTotal = totals.length ? totals[0] : 10.5;
  const mean3 = safeMean(totals.slice(0, 3));
  const mean5 = safeMean(totals.slice(0, 5));
  const mean10 = safeMean(totals.slice(0, 10));
  const mean20 = safeMean(totals.slice(0, 20));

  const shortLong = mean5 - mean20;

  const cntTai20 = res.slice(0, 20).filter(r => r === "Tài").length;
  const cntXiu20 = res.slice(0, 20).filter(r => r === "Xỉu").length;
  const n20 = Math.max(1, Math.min(20, res.length));
  const ratioTai20 = cntTai20 / n20;
  const ratioXiu20 = cntXiu20 / n20;

  const s = streakLen(hist);
  const parityLast = lastTotal % 2 === 1 ? 1 : 0;
  const highLast = lastTotal >= 11 ? 1 : 0;
  const slope = mean3 - mean10;

  const hasLongRun = s >= 4 ? 1 : 0;
  const hasShortRun = s >= 2 ? 1 : 0;

  const zn = (x, m = 10.5, sd = 4.0) => (x - m) / sd;

  let feats = [
    1.0,                   // bias
    zn(lastTotal),
    zn(mean3),
    zn(mean5),
    zn(mean10),
    zn(mean20),
    shortLong / 6,
    ratioTai20,
    ratioXiu20,
    s / 10,
    parityLast,
    highLast,
    slope / 6,
    hasLongRun,
    hasShortRun,
    hist.length >= 15 ? 1 : 0
  ];

  if (feats.length < MAX_FEATURES) {
    feats = feats.concat(Array(MAX_FEATURES - feats.length).fill(0));
  } else if (feats.length > MAX_FEATURES) {
    feats = feats.slice(0, MAX_FEATURES);
  }
  return feats;
}

// ================ RULE-BASED AI ==============
function aiRuleBased(history) {
  const explain = [];
  const votes = [];

  if (!history.length) {
    return { final: "Tài", exp: "Chưa có dữ liệu → mặc định theo Tài", votes: ["Tài(mặc định)"] };
  }

  // 1) Short vs Long means
  const m5 = safeMean(history.slice(0, 5).map(h => h.Tong));
  const m20 = safeMean(history.slice(0, 20).map(h => h.Tong));
  if (m5 > m20) { votes.push("Tài"); explain.push(`Short(${m5.toFixed(1)}) > Long(${m20.toFixed(1)})`); }
  else { votes.push("Xỉu"); explain.push(`Short(${m5.toFixed(1)}) ≤ Long(${m20.toFixed(1)})`); }

  // 2) Tỉ lệ Tài/Xỉu 20 phiên
  const cntTai20 = history.slice(0, 20).filter(h => h.Ket_qua === "Tài").length;
  const cntXiu20 = history.slice(0, 20).filter(h => h.Ket_qua === "Xỉu").length;
  const n20 = Math.max(1, Math.min(20, history.length));
  const pTai = (cntTai20 / n20) * 100;
  const pXiu = (cntXiu20 / n20) * 100;
  if (pTai > 60) { votes.push("Tài"); explain.push(`Tài ${pTai.toFixed(0)}%/20 phiên`); }
  else if (pXiu > 60) { votes.push("Xỉu"); explain.push(`Xỉu ${pXiu.toFixed(0)}%/20 phiên`); }

  // 3) Bẻ cầu dài (>=4)
  if (history.length >= 4 && history.slice(0, 4).every(h => h.Ket_qua === history[0].Ket_qua)) {
    const opposite = history[0].Ket_qua === "Xỉu" ? "Tài" : "Xỉu";
    votes.push(opposite); explain.push(`Cầu ${history[0].Ket_qua} 4 phiên → bẻ cầu`);
  }

  // 4) Theo cầu ngắn (>=2)
  if (history.length >= 2 && history[0].Ket_qua === history[1].Ket_qua) {
    votes.push(history[0].Ket_qua); explain.push("Cầu ngắn 2 phiên → theo cầu");
  }

  // 5) Ngưỡng tổng gần nhất
  const lastTotal = history[0].Tong;
  if (lastTotal >= 12) { votes.push("Tài"); explain.push(`Tổng gần nhất cao (${lastTotal}) → đà Tài`); }
  else if (lastTotal <= 9) { votes.push("Xỉu"); explain.push(`Tổng gần nhất thấp (${lastTotal}) → đà Xỉu`); }

  const final = votes.length
    ? votes.sort((a, b) => (votes.filter(v => v === a).length) - (votes.filter(v => v === b).length)).pop()
    : "Tài";

  return { final, exp: explain.join(" | "), votes };
}

// ============== ENSEMBLE (RULE + ML) =========
function aiPredictEnsemble(history, model) {
  const { final: rulePred, exp: ruleExp } = aiRuleBased(history);

  const x = history.length ? buildFeatures(history) : Array(MAX_FEATURES).fill(0);
  const mlP = model.predictProba(x);
  const mlPred = mlP >= 0.5 ? "Tài" : "Xỉu";

  let final, reason;
  if (history.length >= 15) {
    if (mlPred === rulePred) {
      final = mlPred;
      reason = `[ML ${mlP.toFixed(2)}] & [Rule] đồng thuận → ${final}`;
    } else if (mlP >= 0.45 && mlP <= 0.55) {
      final = rulePred;
      reason = `[ML ${mlP.toFixed(2)}] do dự → theo Rule → ${final}`;
    } else {
      final = mlPred;
      reason = `[ML ${mlP.toFixed(2)}] mạnh hơn Rule → ${final}`;
    }
  } else {
    if (mlPred === rulePred) {
      final = rulePred;
      reason = `Dữ liệu ít, ML ${mlP.toFixed(2)} & Rule trùng → ${final}`;
    } else {
      final = rulePred;
      reason = "Dữ liệu ít, ưu tiên Rule → " + final;
    }
  }

  const explain = `${reason} | Rule: ${ruleExp}`;
  return { duDoan: final, giaiThich: explain, x };
}

// ============= UPDATE PUBLIC STORE ===========
function updatePublicStore(store, history, sid, d1, d2, d3, isMd5) {
  const total = d1 + d2 + d3;
  const kq = getTaiXiu(d1, d2, d3);

  // 1) push history (đầu mảng là phiên mới nhất)
  history.unshift({ Phien: sid, d: [d1, d2, d3], Tong: total, Ket_qua: kq });
  if (history.length > MAX_HISTORY) history.pop();

  // 2) Training online: nếu có pending sample cho phiên hiện tại
  if (isMd5) {
    if (pendingSample101) {
      const { x } = pendingSample101;
      model101.update(x, txToInt(kq));
      pendingSample101 = null;
    }
  } else {
    if (pendingSample100) {
      const { x } = pendingSample100;
      model100.update(x, txToInt(kq));
      pendingSample100 = null;
    }
  }

  // 3) Ensemble predict cho phiên sau
  const { duDoan, giaiThich, x } = isMd5 ? aiPredictEnsemble(history, model101)
                                         : aiPredictEnsemble(history, model100);

  // 4) Random Do_tin_cay 55–99 *mỗi phiên mới*
  let doTinCay;
  if (isMd5) {
    if (sid !== store.Phien_truoc) lastConfidence101 = Math.floor(Math.random() * (99 - 55 + 1)) + 55;
    doTinCay = lastConfidence101;
  } else {
    if (sid !== store.Phien_truoc) lastConfidence100 = Math.floor(Math.random() * (99 - 55 + 1)) + 55;
    doTinCay = lastConfidence100;
  }

  // 5) Lưu pending sample để train khi nhãn của phiên kế đến xuất hiện
  if (isMd5) pendingSample101 = { x, nextPhien: sid + 1 };
  else pendingSample100 = { x, nextPhien: sid + 1 };

  // 6) Cập nhật store public đúng format
  Object.assign(store, {
    Phien_truoc: sid,
    Xuc_xac: `${d1}-${d2}-${d3}`,
    Tong: total,
    Ket_qua: kq,
    Phien_sau: sid + 1,
    Du_doan: duDoan,
    Do_tin_cay: doTinCay,
    Giai_thich: giaiThich,
    id: "Tele@idol_vannhat"
  });
}

// ================= POLLING ===================
async function pollLoop(gid, store, history, isMd5) {
  const url = `https://api-agent.gowsazhjo.net/glms/v1/notify/taixiu?platform_id=b5&gid=${gid}`;

  while (true) {
    try {
      const { data } = await axios.get(url, {
        timeout: 10000,
        headers: { "User-Agent": "Node-Proxy/1.0" }
      });

      if (data && data.status === "OK" && Array.isArray(data.data)) {
        // Với gid=100 cần lấy sid từ cmd=1008 trước để ghép cho 1003
        for (const game of data.data) {
          const cmd = game.cmd;
          if (!isMd5 && cmd === 1008) {
            sidForTX = game.sid;
          }
        }

        for (const game of data.data) {
          const cmd = game.cmd;

          if (isMd5 && cmd === 2006) {
            const sid = game.sid;
            const d1 = game.d1, d2 = game.d2, d3 = game.d3;
            if (sid && sid !== lastSid101 && [d1, d2, d3].every(v => v !== null && v !== undefined)) {
              lastSid101 = sid;
              updatePublicStore(store, history, sid, d1, d2, d3, true);
              console.log(`[MD5] Phiên ${sid} - Tổng: ${d1 + d2 + d3}, KQ: ${getTaiXiu(d1, d2, d3)}`);
            }
          } else if (!isMd5 && cmd === 1003) {
            const d1 = game.d1, d2 = game.d2, d3 = game.d3;
            const sid = sidForTX;
            if (sid && sid !== lastSid100 && [d1, d2, d3].every(v => v !== null && v !== undefined)) {
              lastSid100 = sid;
              updatePublicStore(store, history, sid, d1, d2, d3, false);
              console.log(`[TX ] Phiên ${sid} - Tổng: ${d1 + d2 + d3}, KQ: ${getTaiXiu(d1, d2, d3)}`);
              sidForTX = null;
            }
          }
        }
      }
      await waitMs(POLL_INTERVAL);
    } catch (err) {
      console.error(`Lỗi khi lấy dữ liệu API ${gid}:`, err?.message || err);
      await waitMs(RETRY_DELAY);
    }
  }
}
function waitMs(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

// ================== SERVER ===================
const app = express();

app.get("/api/taixiu", (req, res) => {
  return res.json(latestResult100);
});

app.get("/api/taixiumd5", (req, res) => {
  return res.json(latestResult101);
});

app.get("/api/history", (req, res) => {
  return res.json({
    taixiu: history100,
    taixiumd5: history101
  });
});

app.get("/healthz", (req, res) => res.send("ok"));
app.get("/", (req, res) => {
  res.send("API Server for TaiXiu is running. Endpoints: /api/taixiu, /api/taixiumd5, /api/history");
});

// Khởi động polling và server
app.listen(PORT, HOST, () => {
  console.log(`Server listening on ${HOST}:${PORT}`);
  console.log("Khởi động hệ thống API Tài Xỉu (Rule+ML Ensemble)...");
  // vgm n_100 / vgm n_101
  pollLoop("vgmn_100", latestResult100, history100, false);
  pollLoop("vgmn_101", latestResult101, history101, true);
});

// Đảm bảo không crash khi lỗi mạng
process.on("unhandledRejection", (r) => console.error("UnhandledRejection:", r));
process.on("uncaughtException", (e) => console.error("UncaughtException:", e));
