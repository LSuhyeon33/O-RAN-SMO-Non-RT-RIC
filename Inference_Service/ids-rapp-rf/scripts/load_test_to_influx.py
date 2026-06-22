#!/usr/bin/env python3
"""
Load the IDS test set into InfluxDB v2 (bucket=oom_ids, measurement=oom_test).

Source: balanced_test_classification_dataset.csv — the time-based 8:2 split test
portion produced by build_balanced_dataset.py (6 classes x 15,449 = 92,694 rows).

The existing oom_test measurement data is DELETED before loading.

Schema:
  measurement: oom_test
  tags: scenario (label에서 파생), label
  fields: 20 features + label_int
  time: ts_ns
"""
import os
import sys
import argparse
import time
from datetime import datetime, timezone

import pandas as pd
from influxdb_client import InfluxDBClient, Point, WritePrecision
from influxdb_client.client.write_api import SYNCHRONOUS

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'rapp', 'app'))
from features import FEATURES, LABEL_TO_SCENARIO

TEST_CSV = "/usr/local/o-ran/Dataset/OOM dataset/processed/balanced_test_classification_dataset.csv"
DEFAULT_BUCKET = "oom_ids"
DEFAULT_MEASUREMENT = "oom_test"
DEFAULT_ORG = "est"
DEFAULT_URL = "http://localhost:8086"
WRITE_CHUNK = 5_000


def dedup_timestamps(df: pd.DataFrame) -> pd.DataFrame:
    """동일 (scenario/label) 그룹 내에서 ts_ns가 엄격히 증가하도록 보정.

    InfluxDB는 measurement+태그셋+timestamp가 같으면 한 포인트로 합쳐(나중 쓰기가
    덮어씀) 행이 조용히 사라진다. label↔scenario는 1:1이므로 label 단위로 그룹지어,
    직전 행 이하의 ts_ns를 (직전+1)ns로 밀어 충돌을 제거한다. 연쇄 충돌도 루프가
    자연히 흡수한다 (밀린 값이 다음 행과 또 겹치면 그 행도 다시 밀림).
    """
    df = df.sort_values(['label', 'ts_ns']).reset_index(drop=True)
    bumped = 0
    for _, idx in df.groupby('label').groups.items():
        prev = None
        for i in idx:
            ts = int(df.at[i, 'ts_ns'])
            if prev is not None and ts <= prev:
                ts = prev + 1
                df.at[i, 'ts_ns'] = ts
                bumped += 1
            prev = ts
    if bumped:
        print(f"[dedup] {bumped}개 중복/비증가 ts_ns를 +1ns씩 밀어 보정")
    else:
        print("[dedup] ts_ns 중복 없음")
    return df


def df_to_points(df: pd.DataFrame, measurement: str):
    for row in df.itertuples(index=False):
        label = int(row.label)
        scenario = LABEL_TO_SCENARIO[label]          # label -> scenario 태그 파생
        p = (Point(measurement)
             .tag("scenario", scenario)
             .tag("label", str(label))
             .time(int(row.ts_ns), WritePrecision.NS)
             .field("label_int", label))
        for f in FEATURES:
            p = p.field(f, float(getattr(row, f)))
        yield p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', default=os.environ.get('INFLUX_URL', DEFAULT_URL))
    ap.add_argument('--org', default=os.environ.get('INFLUX_ORG', DEFAULT_ORG))
    ap.add_argument('--bucket', default=os.environ.get('INFLUX_BUCKET', DEFAULT_BUCKET))
    ap.add_argument('--measurement', default=os.environ.get('INFLUX_MEASUREMENT', DEFAULT_MEASUREMENT))
    ap.add_argument('--csv', default=TEST_CSV)
    args = ap.parse_args()

    token = os.environ.get('INFLUX_TOKEN')
    if not token:
        sys.exit("[error] INFLUX_TOKEN 환경변수 필요")

    print(f"[connect] url={args.url} org={args.org} bucket={args.bucket}")
    client = InfluxDBClient(url=args.url, token=token, org=args.org, timeout=120_000)

    # 1) 기존 oom_test 데이터 삭제
    start = "1970-01-01T00:00:00Z"
    stop = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    print(f"[delete] _measurement=\"{args.measurement}\" in [{start}, {stop}]")
    client.delete_api().delete(start, stop, f'_measurement="{args.measurement}"',
                               bucket=args.bucket, org=args.org)

    # 2) 새 test set 로드
    print(f"[load] {args.csv}")
    df = pd.read_csv(args.csv)
    print(f"[load]   rows={len(df)}, cols={len(df.columns)}")
    missing = [c for c in ['ts_ns', 'label'] + FEATURES if c not in df.columns]
    if missing:
        sys.exit(f"[error] missing columns: {missing}")

    df = dedup_timestamps(df)

    write_api = client.write_api(write_options=SYNCHRONOUS)
    points = list(df_to_points(df, args.measurement))
    t0 = time.time()
    for i in range(0, len(points), WRITE_CHUNK):
        write_api.write(bucket=args.bucket, org=args.org, record=points[i:i + WRITE_CHUNK])
    print(f"[write] {len(points)} points in {time.time() - t0:.1f}s")

    # 3) 라벨별 분포 출력
    print("[dist] label distribution:")
    for label, scen in sorted(LABEL_TO_SCENARIO.items()):
        n = int((df['label'] == label).sum())
        print(f"  label {label} ({scen}): {n}")

    client.close()
    print("[done]")


if __name__ == '__main__':
    main()
