"""Exercise real method control flow with scripted chat, not diagnosis replacements."""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import pytest

from profrca import DiagnosisConfig, LLMConfig, diagnose, list_methods
from profrca.errors import ConfigurationError, InputError

EXAMPLES = Path(__file__).resolve().parents[1] / 'examples'


class MemoryKernel:
    """Protocol fixture; real isolation/execution is tested separately with Docker."""
    def __init__(self, wd):
        self.wd = wd
        self.codes = []
        self.closed = False

    def run(self, code):
        assert not self.closed
        self.codes.append(code)
        return True, 'cpu_usage increased from 10 to 95', None

    def close(self):
        self.closed = True


class ScriptedChat:
    def __init__(self):
        self.calls = []
        self.controller_steps = 0

    def __call__(self, messages, **kwargs):
        self.calls.append(messages)
        system, last = messages[0]['content'], messages[-1]['content']
        if system.startswith('You are a role assigner'):
            text = 'CpuExpert'
        elif system.startswith('You are a CpuExpert'):
            text = ('Thought: CPU evidence supports saturation.\nAction: Speak\n'
                    'Action Input: {"diagnose":"CPU saturation", "solution":"reduce CPU load"}')
        elif 'mento-carlo-choice-GPT' in system:
            text = 'candidate_0'
        elif last.startswith('Below is the final root cause analysis report'):
            text = json.dumps({'root_causes': [{'category': 'cpu', 'mechanism': 'CPU saturation', 'location': 'host'}]})
        elif system.startswith('You are the Administrator of a DevOps Assistant'):
            if 'final answer of the root causes' in last or 'requested JSON format' in last:
                text = json.dumps({'1': {'root cause component': 'host', 'root cause reason': 'high CPU usage'}})
            else:
                self.controller_steps += 1
                text = json.dumps({'analysis': 'Check CPU data', 'completed': self.controller_steps > 1,
                                   'instruction': 'Read CPU metrics and compare the two windows'})
        elif 'Executor' in system or 'executor' in system or 'code' in system.lower() and 'python' in system.lower():
            text = ("```python\nimport pandas as pd\npd.read_csv('telemetry/metric.csv').describe()\n```"
                    if 'code' in last.lower() or 'Read CPU' in last or 'Response format' in last
                    else 'CPU usage increased during the observed window')
        elif '数据库' in system or '根因' in system:
            text = json.dumps({'root_causes': [{'category': 'cpu', 'mechanism': 'CPU saturation',
                                               'mechanism_id': 'cpu_saturation', 'location': 'host',
                                               'evidence': ['T0'], 'why': 'CPU increased'}]})
        else:
            text = 'CPU saturation caused request timeouts.'
        return text, {'prompt_tokens': 10, 'completion_tokens': 5}


@pytest.mark.parametrize('method', list_methods())
@pytest.mark.parametrize('mode', ['log', 'metrics', 'both'])
def test_all_methods_share_input_output_contract(method, mode):
    chat = ScriptedChat()
    config = DiagnosisConfig(llm=LLMConfig(chat=chat), include_details=True)
    options = {'kernel_factory': MemoryKernel} if method == 'rca_agent' else {}
    result = diagnose(log=EXAMPLES / 'log.txt' if mode != 'metrics' else None,
                      metrics=EXAMPLES / 'metric.csv' if mode != 'log' else None,
                      method=method, config=config, **options)
    assert result['status'] == 'ok', result
    assert result['method'] == method
    assert result['schema_version'] == 'profrca.diagnosis.v1'
    assert result['root_causes'] and result['root_causes'][0]['type'] == 'cpu'
    assert result['llm_calls'] > 0 and chat.calls
    assert result['output_tokens'] == result['llm_calls'] * 5
    assert 'evidence' in result and 'evidence_status' in result
    assert 'mechanism_id' in result['root_causes'][0]
    json.dumps(result)
    if method == 'rca_agent':
        assert result['cells'] >= 1 and result['trajectory']


def test_unknown_method_and_no_input():
    with pytest.raises(ConfigurationError, match='Unknown method'):
        diagnose(method='typo')
    with pytest.raises(InputError, match='Provide'):
        diagnose()


def test_unknown_option_is_configuration_error():
    with pytest.raises(ConfigurationError, match='Unsupported'):
        diagnose(metrics=EXAMPLES / 'metric.csv', gold={'cause': 'cpu'})


def test_missing_llm_configuration(monkeypatch):
    for key in ('DEEPSEEK_API_KEY', 'DEEPSEEK_API_BASE', 'DEEPSEEK_MODEL', 'FAST_RCA_MODEL'):
        monkeypatch.delenv(key, raising=False)
    with pytest.raises(ConfigurationError):
        diagnose(log=EXAMPLES / 'log.txt', method='llm')


def test_network_error_is_redacted():
    def fail(*args, **kwargs):
        raise RuntimeError('http://private-service/?key=sk-private-secret')
    result = diagnose(log=EXAMPLES / 'log.txt', method='llm', config=DiagnosisConfig(llm=LLMConfig(chat=fail)))
    assert result['status'] == 'error' and result['error_code'] == 'RuntimeError'
    assert 'private' not in json.dumps(result)


def test_llm_calls_do_not_share_per_request_configuration():
    def run(marker):
        def chat(*args, **kwargs):
            return json.dumps({'root_causes': [{'category': 'cpu', 'mechanism': marker}]}), {}
        return diagnose(log=EXAMPLES / 'log.txt', method='llm', config=DiagnosisConfig(llm=LLMConfig(chat=chat)))
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(run, ('first-model', 'second-model')))
    assert [r['root_causes'][0]['headline'] for r in results] == ['first-model', 'second-model']


def test_dbot_parallel_experts_keep_custom_client():
    chat = ScriptedChat()
    result = diagnose(log=EXAMPLES / 'log.txt', method='dbot', config=DiagnosisConfig(llm=LLMConfig(chat=chat)))
    assert result['status'] == 'ok' and result['experts'] == ['CpuExpert']
    assert any(m[0]['content'].startswith('You are a CpuExpert') for m in chat.calls)


def test_cli(monkeypatch, capsys):
    from profrca.__main__ import main
    from profrca import llm
    monkeypatch.setattr(llm, 'current_config', lambda: LLMConfig(chat=ScriptedChat()))
    assert main(['--list-methods']) == 0
    assert 'rca_agent' in capsys.readouterr().out
    assert main(['--method', 'llm', '--log', str(EXAMPLES / 'log.txt')]) == 0
    assert json.loads(capsys.readouterr().out)['status'] == 'ok'
