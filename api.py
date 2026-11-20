"""Stable single-case API shared by the application and external evaluators."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .errors import ConfigurationError, ExecutionUnavailable, InputError
from .inputs import DiagnosisInput
from .llm import LLMConfig, using_llm

SCHEMA_VERSION = 'profrca.diagnosis.v1'


@dataclass
class DiagnosisConfig:
    llm: Optional[LLMConfig] = None
    metric_format: str = 'auto'
    time_col: str = 'time'
    entity: str = 'PostgreSQL'
    include_details: bool = False
    # Backend-specific options are passed unchanged, not guessed by the facade.
    options: dict = field(default_factory=dict)


def _evidence(out):
    evidence = []
    for rank, root in enumerate(out.get('root_causes') or [], 1):
        for ref, lines in (root.get('evidence_lines') or {}).items():
            evidence.append({'kind': 'log_lines', 'candidate_rank': rank, 'ref': ref,
                             'source': 'log', 'lines': list(lines), 'verified': True})
        for ref in root.get('evidence_refs') or []:
            evidence.append({'kind': 'reference', 'candidate_rank': rank, 'ref': ref,
                             'source': 'metrics' if str(ref).startswith('M') else 'log', 'verified': False})
        if not root.get('evidence_refs'):
            for text in root.get('evidence_templates') or []:
                evidence.append({'kind': 'reported_text', 'candidate_rank': rank, 'text': str(text),
                                 'source': 'model_report', 'verified': False})
    return evidence


def _safe_error(exc, config):
    if isinstance(exc, (InputError, ConfigurationError, ExecutionUnavailable)):
        return str(exc)
    # Network exceptions can contain URLs or model-service response bodies.
    return type(exc).__name__ + ': diagnosis failed; inspect the local backend with credentials redacted'


def _protect_backend_errors(value):
    """Older backends may catch an HTTP exception themselves; redact those too."""
    safe = {'cancelled', 'no predictions', 'no root cause in final answer',
            'no expert reached a terminal Speak', 'format step returned no candidates',
            'no logs and no metrics', 'no root cause in final answer'}
    if isinstance(value, dict):
        for key, item in list(value.items()):
            if key == 'error' and item and str(item) not in safe:
                value[key] = 'Backend reported an execution or response error; inspect local instrumentation with credentials redacted'
            elif isinstance(item, (dict, list)):
                _protect_backend_errors(item)
    elif isinstance(value, list):
        for item in value:
            _protect_backend_errors(item)


def diagnose(log=None, metrics=None, *, method='profrca', config=None, **options):
    """Return a JSON-compatible result; legacy ProfRCA result keys remain at top level.

    Invalid method names/inputs/configuration raise ValueError subclasses. Runtime
    failures return status=error and a typed error_code. Labels and scoring are
    deliberately not accepted by this API.
    """
    from .factory import create_method
    if config is None:
        config = DiagnosisConfig()
    if not isinstance(config, DiagnosisConfig):
        raise ConfigurationError('config must be DiagnosisConfig')
    backend = create_method(method)
    merged = dict(config.options)
    merged.update(options)
    # Keep historical diagnose(..., time_col=...) calls working.
    time_col = merged.pop('time_col', config.time_col)
    prepared = DiagnosisInput(log, metrics, config.metric_format, time_col, config.entity).prepare()
    started = time.perf_counter()
    try:
        with using_llm(config.llm):
            out = backend.run(prepared, config, merged)
        _protect_backend_errors(out)
    except (InputError, ConfigurationError):
        raise
    except Exception as exc:
        out = {'root_causes': [], 'error': _safe_error(exc, config), 'error_code': type(exc).__name__,
               'route': backend.name, 'usage': []}
    roots = []
    for rank, root in enumerate(out.get('root_causes') or [], 1):
        roots.append({'rank': rank, 'type': '', 'mechanism_id': None, 'headline': '', 'description': '',
                      'location': None, 'evidence_templates': [], 'evidence_refs': [],
                      'evidence_lines': {}, **root})
    out['root_causes'] = roots
    status = 'cancelled' if out.get('error') == 'cancelled' else ('ok' if roots else ('error' if out.get('error') else 'no_prediction'))
    evidence = _evidence(out)
    usage = out.get('usage') or []
    out.update(schema_version=SCHEMA_VERSION, method=backend.name, status=status,
               root_causes=roots, evidence=evidence,
               evidence_status='available' if evidence else 'unavailable',
               evidence_note=None if evidence else (
                   'Metric classifier route has no log citations; log-evidence fusion is a separate extension.'
                   if out.get('route') == 'metric_tool' else 'This backend did not return source-linked evidence.'),
               seconds=round(time.perf_counter() - started, 4), input_summary=prepared.metadata)
    out.setdefault('error', None)
    out.setdefault('error_code', 'BackendError' if out.get('error') else None)
    out.setdefault('route', backend.name)
    out.setdefault('llm_calls', len(usage))
    out.setdefault('output_tokens', sum(int(u.get('completion_tokens') or 0) for u in usage))
    out.setdefault('prompt_tokens', sum(int(u.get('prompt_tokens') or 0) for u in usage))
    if not config.include_details:
        out.pop('trajectory', None)
        out.pop('final_answer', None)
    return out
