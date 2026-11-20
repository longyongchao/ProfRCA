"""Validate observable telemetry and convert supported metric schemas explicitly."""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .errors import InputError


@dataclass
class DiagnosisInput:
    log: object = None
    metrics: object = None
    metric_format: str = 'auto'
    time_col: str = 'time'
    entity: str = 'PostgreSQL'
    metadata: dict = field(default_factory=dict)

    def prepare(self):
        if self.log is None and self.metrics is None:
            raise InputError('Provide log and/or metrics')
        path = None
        if self.log is not None:
            path = Path(self.log)
            if not path.is_file():
                raise InputError('Log file does not exist')
            try:
                with path.open(encoding='utf-8-sig') as stream:
                    seen = False
                    for line in stream:
                        if '\x00' in line:
                            raise InputError('Log must be UTF-8 text, not binary')
                        seen = seen or bool(line.strip())
                if not seen:
                    raise InputError('Log file is empty')
            except UnicodeDecodeError as exc:
                raise InputError('Log must use UTF-8 encoding') from exc
        wide, info = normalize_metrics(self.metrics, self.metric_format, self.time_col, self.entity)
        return DiagnosisInput(path, wide, 'wide', 'time', self.entity, info)


def _seconds(values):
    numeric = pd.to_numeric(values, errors='coerce')
    if numeric.notna().all():
        result = numeric.astype(float)
    elif numeric.notna().any():
        raise InputError('Metric time mixes numeric and textual timestamps')
    else:
        try:
            stamps = pd.to_datetime(values, utc=True, errors='raise')
            if stamps.isna().any():
                raise ValueError('missing time')
            result = stamps.astype('int64') / 1e9
        except (ValueError, TypeError) as exc:
            raise InputError('Metric time must be numeric seconds or ISO-8601 timestamps') from exc
    if not np.isfinite(result.to_numpy(float)).all():
        raise InputError('Metric timestamps must be finite')
    return result


def normalize_metrics(metrics, metric_format='auto', time_col='time', entity='PostgreSQL'):
    if metrics is None:
        return None, {'metrics_format': None}
    if metric_format not in ('auto', 'wide', 'long', 'dbot_long'):
        raise InputError('metric_format must be auto, wide, long or dbot_long')
    if isinstance(metrics, pd.DataFrame):
        frame = metrics.copy()
    else:
        try:
            path = Path(metrics)
            # pandas otherwise silently renames duplicate CSV headers.
            with path.open(encoding='utf-8-sig', newline='') as stream:
                header = next(csv.reader(stream), [])
            if len(header) != len(set(header)):
                raise InputError('Duplicate metric column names')
            frame = pd.read_csv(path, encoding='utf-8-sig')
        except (OSError, UnicodeDecodeError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise InputError('Metrics must be a readable UTF-8 CSV') from exc
    if frame.empty or frame.columns.duplicated().any():
        raise InputError('Metrics are empty or contain duplicate column names')
    original_rows = len(frame)
    fmt = metric_format
    if fmt == 'auto':
        if {'timestamp', 'metric_name', 'metric_value'} <= set(frame.columns):
            fmt = 'dbot_long'
        elif {'ts', 'entity', 'metric', 'value'} <= set(frame.columns):
            fmt = 'long'
        else:
            fmt = 'wide'
    if fmt in ('long', 'dbot_long'):
        ts, name, val = ('timestamp', 'metric_name', 'metric_value') if fmt == 'dbot_long' else ('ts', 'metric', 'value')
        required = {ts, name, val} | ({'entity'} if fmt == 'long' else set())
        if not required <= set(frame.columns):
            raise InputError('Missing required long-table columns: ' + ', '.join(sorted(required - set(frame.columns))))
        entities = frame['entity'] if fmt == 'long' else pd.Series(entity, index=frame.index)
        if frame[name].isna().any() or entities.isna().any():
            raise InputError('Metric entity/name cannot be missing')
        data = pd.DataFrame({'time': _seconds(frame[ts]),
                             'column': entities.astype(str) + '|' + frame[name].astype(str),
                             'value': pd.to_numeric(frame[val], errors='raise')})
        # Only these observed fields are consumed: phase, labels and metadata are NOT model inputs.
        frame = data.pivot_table(index='time', columns='column', values='value', aggfunc='median').reset_index()
        frame.columns.name = None
    else:
        if time_col not in frame:
            raise InputError('Wide metrics require time column: ' + time_col)
        frame = frame.rename(columns={time_col: 'time'})
        frame['time'] = _seconds(frame['time'])
    if len(frame.columns) < 2 or len(frame) < 4 or frame['time'].nunique() < 4:
        raise InputError('Metrics require at least 4 distinct timestamps and one numeric metric')
    if frame['time'].duplicated().any():
        raise InputError('Duplicate timestamps in wide metrics; aggregate them explicitly first')
    try:
        values = frame.drop(columns='time').apply(pd.to_numeric, errors='raise')
    except (ValueError, TypeError) as exc:
        raise InputError('Wide metric columns must be numeric; remove labels/metadata before diagnosis') from exc
    array = values.to_numpy(float)
    if np.isinf(array).any() or not np.isfinite(array).any(axis=0).all():
        raise InputError('Metrics contain infinity or an entirely missing series')
    wide = pd.concat([frame[['time']], values], axis=1).sort_values('time', kind='stable').reset_index(drop=True)
    return wide, {'metrics_format': fmt, 'input_rows': original_rows, 'rows': len(wide),
                  'metric_columns': list(values.columns), 'time_unit': 'seconds',
                  'duplicate_long_values': 'median' if fmt != 'wide' else None}
