import json

import numpy as np
import pandas as pd
import pytest

from profrca.errors import InputError
from profrca.inputs import DiagnosisInput, normalize_metrics


def test_dbot_long_schema_ignores_metadata_and_gold_fields():
    frame = pd.DataFrame({'timestamp': ['2026-11-19T10:00:0%dZ' % n for n in range(4)],
                          'phase': ['baseline', 'fault', 'fault', 'recovery'],
                          'metric_name': ['cpu'] * 4, 'metric_value': [1, 2, 3, 4],
                          'unit': ['%'] * 4, 'source_scope': ['postgresql'] * 4,
                          'ground_truth': ['DO_NOT_READ'] * 4})
    wide, info = normalize_metrics(frame)
    assert list(wide.columns) == ['time', 'PostgreSQL|cpu']
    assert info['metrics_format'] == 'dbot_long'
    assert 'DO_NOT_READ' not in wide.to_csv()
    assert len(wide) == 4  # Phase is not used to select a gold fault window.


def test_generic_long_duplicate_observations_use_median():
    frame = pd.DataFrame({'ts': [0, 1, 2, 3, 3], 'entity': ['host'] * 5,
                          'metric': ['cpu'] * 5, 'value': [1, 2, 3, 4, 8]})
    wide, info = normalize_metrics(frame)
    assert wide['host|cpu'].tolist() == [1, 2, 3, 6]
    assert info['duplicate_long_values'] == 'median'


def test_custom_time_column_is_sorted_without_mutating_input():
    frame = pd.DataFrame({'timestamp': [3, 2, 1, 0], 'host_cpu': [1, 2, 3, 4]})
    wide, _ = normalize_metrics(frame, time_col='timestamp')
    assert wide['time'].tolist() == [0, 1, 2, 3]
    assert 'time' not in frame


@pytest.mark.parametrize('frame', [
    pd.DataFrame({'time': [0, 1, 2], 'cpu': [1, 2, 3]}),
    pd.DataFrame({'time': [0, 1, 2, 2], 'cpu': [1, 2, 3, 4]}),
    pd.DataFrame({'time': [0, 1, 2, 3], 'cpu': [np.nan] * 4}),
    pd.DataFrame({'time': [0, 1, 2, 3], 'cpu': [1, 2, np.inf, 4]}),
    pd.DataFrame({'time': [0, 1, 2, 3], 'label': ['fault'] * 4}),
])
def test_invalid_metrics_fail_before_diagnosis(frame):
    with pytest.raises(InputError):
        normalize_metrics(frame)


def test_duplicate_csv_headers_are_not_silently_renamed(tmp_path):
    path = tmp_path / 'metric.csv'
    path.write_text('time,cpu,cpu\n0,1,2\n1,2,3\n2,3,4\n3,4,5\n')
    with pytest.raises(InputError, match='Duplicate'):
        normalize_metrics(path)


def test_empty_binary_and_invalid_encoding_logs(tmp_path):
    path = tmp_path / 'log.txt'
    for data in (b'', b'\x00binary', b'\xff\xfe'):
        path.write_bytes(data)
        with pytest.raises(InputError):
            DiagnosisInput(log=path).prepare()
