import pandas as pd
from influxdb_client import InfluxDBClient

import config
from features import FEATURES


def _client():
    if not config.INFLUX_TOKEN:
        raise RuntimeError("INFLUX_TOKEN not set")
    return InfluxDBClient(
        url=config.INFLUX_URL,
        token=config.INFLUX_TOKEN,
        org=config.INFLUX_ORG,
        timeout=60_000,
    )


def health() -> bool:
    try:
        return _client().ping()
    except Exception:
        return False


def fetch_test_by_scenario(scenario: str) -> pd.DataFrame:
    flux = f'''
from(bucket: "{config.INFLUX_BUCKET}")
  |> range(start: 0)
  |> filter(fn: (r) => r._measurement == "{config.INFLUX_MEASUREMENT}" and r.scenario == "{scenario}")
  |> pivot(rowKey: ["_time"], columnKey: ["_field"], valueColumn: "_value")
  |> sort(columns: ["_time"])
'''
    with _client() as client:
        df = client.query_api().query_data_frame(flux)
    if isinstance(df, list):
        df = pd.concat(df, ignore_index=True) if df else pd.DataFrame()
    if df.empty:
        return pd.DataFrame(columns=['ts_ns', 'label'] + FEATURES)

    df['ts_ns'] = df['_time'].astype('int64')
    df['label'] = df['label_int'].astype(int)
    keep = ['ts_ns', 'label'] + FEATURES
    return df[keep]
