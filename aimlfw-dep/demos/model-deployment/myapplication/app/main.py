import os, time, csv
import requests
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# =========================
# ENV
# =========================
KSERVE_HOST = os.environ["KSERVE_HOST"].rstrip("/")
MODEL_NAME = os.environ["MODEL_NAME"]

# CSV_PATH = os.environ.get("CSV_PATH", "/app/data/balanced_CU.csv")
# OUT_CSV = os.environ.get("OUT_CSV", "/app/results_cu_cnn_10.csv")
CSV_PATH = os.environ.get("CSV_PATH", "/app/data/DU_sampled_80000.csv")
OUT_CSV = os.environ.get("OUT_CSV", "/app/results_du_transformer_10_ver2.csv")

# sweep range
N_MIN = int(os.environ.get("N_MIN", "100"))
N_MAX = int(os.environ.get("N_MAX", "80000"))
N_STEP = int(os.environ.get("N_STEP", "100"))

# latency config
N_RUNS = int(os.environ.get("N_RUNS", "100"))
WARMUP_RUNS = int(os.environ.get("WARMUP_RUNS", "5"))
TIMEOUT_SEC = int(os.environ.get("TIMEOUT_SEC", "60"))
THR = float(os.environ.get("THRESHOLD", "0.5"))

SEED = int(os.environ.get("SEED", "7"))

URL = f"{KSERVE_HOST}/v1/models/{MODEL_NAME}:predict"
S = requests.Session()

# FEATURES = [
#     "src_port", "dst_port", "duration", "src_bytes", "dst_bytes", "missed_bytes",
#     "src_pkts", "src_ip_bytes", "dst_pkts", "dst_ip_bytes", "ip_proto", "http_trans_depth",
#     "files_total_bytes", "is_GET_mthd", "http_status_error", "is_file_transfered"
# ]
FEATURES = [
    "dlbytes", "dlmcs", "dlbler", "ulbytes", "ulmcs", "ulbler",
    "ri", "phr", "pcmax", "rsrq", "sinr", "rsrp", "rssi",
    "cqi", "pucchsnr", "puschsnr", "ue_id", "timestamp", "cellid", "in_sync"
]
LABEL_COL = "traffic_type"


def pctl(vals, p):
    vals = sorted(vals)
    if not vals:
        return None
    k = (len(vals) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    if f == c:
        return vals[f]
    return vals[f] + (vals[c] - vals[f]) * (k - f)


def post(instances, retries=5, backoff=1.0):
    last_err = None
    for i in range(retries):
        try:
            t0 = time.perf_counter()
            r = S.post(URL, json={"instances": instances}, timeout=TIMEOUT_SEC)
            dt = time.perf_counter() - t0
            if r.status_code != 200:
                raise RuntimeError(f"HTTP {r.status_code}: {r.text}")
            return dt, r.json()
        except Exception as e:
            last_err = e
            time.sleep(backoff * (2 ** i))
    raise RuntimeError(f"POST failed after {retries} retries: {last_err}")


def calculate_recon_error(original_X, resp):
    predicted_X = resp.get("predictions", resp.get("outputs"))
    orig = np.array(original_X)
    pred = np.array(predicted_X)
    errors = np.mean(np.square(orig - pred), axis=1)
    return errors.tolist()


def append_csv(row):
    exists = os.path.exists(OUT_CSV)
    with open(OUT_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            w.writeheader()
        w.writerow(row)


def find_optimal_threshold(df, features, label_col, n_sample=5000):
    """Validation set에서 F1-score를 최대화하는 threshold 탐색"""
    print("=" * 60)
    print("[THRESHOLD SEARCH] Starting optimal threshold search...")
    print("=" * 60)

    sub = df.sample(n=min(n_sample, len(df)), random_state=SEED)
    X = sub[features].values.tolist()
    y_true = sub[label_col].astype(int).values.tolist()

    # 모델에 추론 요청
    _, resp = post(X)
    recon_errors = calculate_recon_error(X, resp)

    print(f"Recon error stats: min={min(recon_errors):.6f}, max={max(recon_errors):.6f}, mean={np.mean(recon_errors):.6f}")

    # threshold sweep: 0.001 ~ 0.1 (step 0.001)
    best_thr = 0.0
    best_f1 = 0.0
    best_acc = 0.0
    best_prec = 0.0
    best_rec = 0.0

    print(f"\n{'threshold':>12} | {'accuracy':>10} | {'f1':>10} | {'precision':>10} | {'recall':>10}")
    print("-" * 65)

    for t_int in range(1, 101):
        t = t_int * 0.001
        y_pred = [1 if err >= t else 0 for err in recon_errors]
        acc = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)

        if t_int % 5 == 0 or f1 > best_f1:
            print(f"{t:>12.3f} | {acc:>10.6f} | {f1:>10.6f} | {prec:>10.6f} | {rec:>10.6f}")

        if f1 > best_f1:
            best_thr = t
            best_f1 = f1
            best_acc = acc
            best_prec = prec
            best_rec = rec

    print("-" * 65)
    print(f"[RESULT] Optimal threshold = {best_thr:.4f}")
    print(f"         Accuracy  = {best_acc:.6f}")
    print(f"         F1-score  = {best_f1:.6f}")
    print(f"         Precision = {best_prec:.6f}")
    print(f"         Recall    = {best_rec:.6f}")
    print("=" * 60)

    return best_thr


def main():
    model_lower = MODEL_NAME.lower()
    if "autoencoder" in model_lower:
        model_type = "Autoencoder"
    elif "lstm" in model_lower:
        model_type = "LSTM"
    elif "cnn" in model_lower:
        model_type = "CNN"
    elif "transformer" in model_lower:
        model_type = "Transformer"
    else:
        model_type = "Unknown"

    print(f"URL={URL} | MODEL={MODEL_NAME} ({model_type})")
    print(f"CSV_PATH={CSV_PATH}")

    df = pd.read_csv(CSV_PATH)

    # Column rename
    if "is_get_mthd" in df.columns and "is_GET_mthd" not in df.columns:
        df = df.rename(columns={"is_get_mthd": "is_GET_mthd"})

    # Check missing columns and fill with 0
    need = set(FEATURES + [LABEL_COL])
    miss = [c for c in need if c not in df.columns]
    if miss:
        print(f"[WARNING] Missing columns: {miss}. Filling with 0.")
        for c in miss:
            df[c] = 0

    print(f"Successfully loaded CSV. Total rows: {len(df)}")

    # Data scaling (0~1 normalization)
    sub_features = df[FEATURES].astype(float)
    df[FEATURES] = (sub_features - sub_features.min()) / (sub_features.max() - sub_features.min() + 1e-12)
    print("Data scaling completed.")

    # Autoencoder: find optimal threshold before sweep
    if model_type == "Autoencoder":
        active_thr = find_optimal_threshold(df, FEATURES, LABEL_COL)
        print(f"\n>>> Using searched optimal threshold: {active_thr:.4f}")
    else:
        active_thr = THR
        print(f"\n>>> Using ENV threshold: {active_thr}")

    print(f"Starting Sweep: N_MIN={N_MIN}, N_MAX={N_MAX}, N_STEP={N_STEP}")

    for n in range(N_MIN, N_MAX + 1, N_STEP):
        sub = df.sample(n=n, random_state=SEED)
        X = sub[FEATURES].values.tolist()
        y_true = sub[LABEL_COL].astype(int).values.tolist()

        print(f"Current Batch: {n} | Measuring {N_RUNS} runs...")

        # LSTM, CNN, Transformer input shape: (N, 1, F)
        if model_type in ("LSTM", "CNN", "Transformer"):
            X = [[row] for row in X]

        # warmup
        for _ in range(WARMUP_RUNS):
            try:
                post(X)
            except Exception:
                pass

        # measure latency
        lats = []
        first_resp = None
        for _ in range(N_RUNS):
            dt, resp = post(X)
            lats.append(dt)
            if first_resp is None:
                first_resp = resp

        mean_s = sum(lats) / len(lats)
        p95_s = pctl(lats, 95)
        p99_s = pctl(lats, 99)

        # metrics
        if model_type == "Autoencoder":
            flat_X = X
            recon_errors = calculate_recon_error(flat_X, first_resp)
            y_pred = [1 if err >= active_thr else 0 for err in recon_errors]
        else:
            raw = first_resp.get("predictions", first_resp.get("outputs"))
            probs = []
            for r in raw:
                if isinstance(r, (int, float)):
                    probs.append(float(r))
                elif isinstance(r, (list, tuple)):
                    probs.append(float(r[1]) if len(r) == 2 else float(r[0]))
                else:
                    probs.append(0.0)
            y_pred = [1 if p >= active_thr else 0 for p in probs]

        row = {
            "model": MODEL_NAME,
            "dataset": "DU",
            "instances": n,
            "mean_ms": round(mean_s * 1000.0, 3),
            "p95_ms": round(p95_s * 1000.0, 3),
            "p99_ms": round(p99_s * 1000.0, 3),
            "throughput_inst_per_s": round(n / mean_s, 2),
            "acc": round(accuracy_score(y_true, y_pred), 6),
            "f1": round(f1_score(y_true, y_pred, zero_division=0), 6),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 6),
            "recall": round(recall_score(y_true, y_pred, zero_division=0), 6),
        }

        print(row)
        append_csv(row)

    print("\n[DONE] sweep finished.")


if __name__ == "__main__":
    main()