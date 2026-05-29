FEATURES = [
    'inter_arrival_time_ms', 'request_response_latency_ms',
    'timeouts_last_30s', 'timeouts_last_60s', 'total_correlation_errors',
    'errors_last_30s', 'subscription_mismatch', 'subscription_mismatches_last_30s',
    'ssn_retransmission', 'ssn_retransmissions_last_30s',
    'tsn_out_of_order', 'tsn_out_of_order_last_30s',
    'message_size_anomaly', 'msg_size_anomalies_last_30s',
    'consecutive_tx', 'consecutive_rx',
    'arrival_time_cv_30', 'msg_size_cv_30',
    'event_rate_30s', 'tx_rx_ratio_30',
]

LABEL = 'label'
N_CLASSES = 6
N_FEATURES = len(FEATURES)

SCENARIO_TO_LABEL = {
    'oran_logs_baseline': 0,
    'oran_logs_S1_Normal': 1,
    'oran_logs_S1_Terminattion': 2,
    'oran_logs_S2': 3,
    'oran_logs_S3': 4,
    'oran_logs_S4': 5,
}

LABEL_TO_SCENARIO = {v: k for k, v in SCENARIO_TO_LABEL.items()}
