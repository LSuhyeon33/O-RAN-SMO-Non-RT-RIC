import pandas as pd
import json
import os
import numpy as np

# =========================
# 설정
# =========================
CSV_PATH = "/usr/local/o-ran/aimlfw-dep/demos/model-deployment/myapplication/app/balanced_CU.csv"
OUTPUT_DIR = "/usr/local/o-ran/aimlfw-dep/demos/model-deployment/myapplication/app/input"

# 생성할 샘플 개수
SIZES = [1709, 3417, 6835, 17087, 34173, 68346]

# =========================
# CSV 로드
# =========================
df = pd.read_csv(CSV_PATH)

print("Loaded:", CSV_PATH)
print("Rows:", len(df), "Cols:", len(df.columns))

# 컬럼 정리
df = df.rename(columns={"is_GET_mthd": "is_get_mthd"})

# 필요한 컬럼
FEATURE_COLUMNS = [
    'src_port', 'dst_port', 'duration', 'src_bytes', 'dst_bytes',
    'missed_bytes', 'src_pkts', 'src_ip_bytes', 'dst_pkts',
    'dst_ip_bytes', 'ip_proto', 'http_trans_depth',
    'files_total_bytes', 'is_get_mthd',
    'http_status_error', 'is_file_transfered'
]

LABEL_COLUMN = "traffic_type"

# 컬럼 존재 확인
missing = [c for c in FEATURE_COLUMNS + [LABEL_COLUMN] if c not in df.columns]
if missing:
    raise ValueError(f"CSV에 필요한 컬럼이 없습니다: {missing}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# JSON 생성
# =========================
for size in SIZES:
    if size > len(df):
        print(f"[SKIP] size {size} > total rows")
        continue

    sample_df = df.sample(n=size, random_state=42)

    data = []

    for _, row in sample_df.iterrows():
        features = row[FEATURE_COLUMNS].astype(float).tolist()
        label = int(row[LABEL_COLUMN])
        data.append(features + [label])

    output_path = os.path.join(OUTPUT_DIR, f"input_cu_{size}.json")

    with open(output_path, "w") as f:
        json.dump(data, f)

    print(f"[OK] Saved {size} -> {output_path}")

print("Done.")

