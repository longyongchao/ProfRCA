"""Small explicit factory: one registry, no import-time loading of optional engines."""
from __future__ import annotations

from .errors import ConfigurationError

_ALIASES = {'plain_llm': 'llm', 'd-bot': 'dbot', 'rca-agent': 'rca_agent'}
_METHODS = ('profrca', 'llm', 'dbot', 'rca_agent')


def list_methods():
    return list(_METHODS)


def create_method(name='profrca'):
    name = _ALIASES.get(str(name).lower(), str(name).lower())
    from .methods.adapters import ProfRCAMethod, PlainLLMMethod, DbotMethod, RCAAgentMethod
    classes = {'profrca': ProfRCAMethod, 'llm': PlainLLMMethod, 'dbot': DbotMethod, 'rca_agent': RCAAgentMethod}
    if name not in classes:
        raise ConfigurationError('Unknown method; choose one of: ' + ', '.join(_METHODS))
    return classes[name]()
