"""Per-call chat configuration without process-wide environment mutation."""
from __future__ import annotations

import os
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Callable, Optional

import requests

from .errors import ConfigurationError


@dataclass
class LLMConfig:
    base_url: Optional[str] = None
    api_key: Optional[str] = field(default=None, repr=False)
    model: Optional[str] = None
    timeout: float = 300
    trust_env: bool = True
    # Integration seam for a host's own client and deterministic transport tests.
    chat: Optional[Callable] = field(default=None, repr=False)

    def __post_init__(self):
        if self.timeout <= 0:
            raise ConfigurationError('LLM timeout must be positive')

    def resolved(self):
        base = (self.base_url or os.environ.get('DEEPSEEK_API_BASE', '')).rstrip('/')
        model = self.model or os.environ.get('FAST_RCA_MODEL') or os.environ.get('DEEPSEEK_MODEL', '').split(',')[0].strip()
        key = self.api_key or os.environ.get('DEEPSEEK_API_KEY', '')
        if not base or not model or not key:
            raise ConfigurationError('Configure DEEPSEEK_API_BASE, DEEPSEEK_MODEL, DEEPSEEK_API_KEY or pass LLMConfig')
        if not base.startswith(('https://', 'http://')):
            raise ConfigurationError('LLM base_url must be an http(s) URL')
        return base, model, key


_CONFIG = ContextVar('profrca_llm_config', default=None)


@contextmanager
def using_llm(config=None):
    token = _CONFIG.set(config)
    try:
        yield
    finally:
        _CONFIG.reset(token)


def current_config():
    return _CONFIG.get() or LLMConfig()


def chat(messages, max_tokens=800, temperature=0.0, timeout=None, thinking=False, config=None):
    config = config or current_config()
    if config.chat is not None:
        return config.chat(messages, max_tokens=max_tokens, temperature=temperature, timeout=timeout or config.timeout)
    base, model, key = config.resolved()
    body = {'model': model, 'messages': messages, 'temperature': temperature, 'max_tokens': max_tokens,
            'chat_template_kwargs': {'thinking': thinking, 'enable_thinking': thinking}}
    with requests.Session() as session:
        session.trust_env = config.trust_env
        response = session.post(base + '/chat/completions', json=body,
                                headers={'Authorization': 'Bearer ' + key}, timeout=timeout or config.timeout)
        # Do not include HTTP response bodies or credentials in public errors.
        if response.status_code in (400, 413) and any(
                word in response.text.lower() for word in ('context', '超出模型允许范围', 'maximum token')):
            raise OverflowError('Model context length exceeded')
        response.raise_for_status()
        data = response.json()
    return data['choices'][0]['message'].get('content') or '', data.get('usage') or {}
