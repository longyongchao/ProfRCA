"""Opt-in real executor checks; no LLM API and no benchmark data needed."""
import os
from pathlib import Path

import pytest

from profrca.methods.kernel import DockerKernel

pytestmark = pytest.mark.skipif(os.environ.get('PROFRCA_RUN_DOCKER') != '1', reason='opt-in Docker integration')


def test_real_stateful_kernel_and_isolation(tmp_path):
    (tmp_path / 'telemetry').mkdir()
    (tmp_path / 'telemetry' / 'metric.csv').write_text('time,cpu\n0,10\n1,95\n')
    with DockerKernel(tmp_path, timeout=45) as kernel:
        assert kernel.run('a = 40\na + 2')[:2] == (True, '42')
        assert kernel.run('a + 3')[:2] == (True, '43')
        assert kernel.run("import pandas as pd\npd.read_csv('telemetry/metric.csv')['cpu'].max()")[:2] == (True, '95')
        assert kernel.run("open('telemetry/metric.csv', 'w').write('bad')")[0] is False
        assert kernel.run("open('/etc/blocked-write', 'w').write('bad')")[0] is False
        assert kernel.run("import socket\nsocket.create_connection(('1.1.1.1', 443), timeout=2)")[0] is False
        assert kernel.run("import os\n[k for k in os.environ if 'API_KEY' in k]")[:2] == (True, '[]')
        assert kernel.run("import os\nos.getuid()")[:2] == (True, '65534')
    assert kernel.closed and kernel.p.poll() is not None


def test_cell_timeout_closes_kernel(tmp_path):
    (tmp_path / 'telemetry').mkdir()
    with DockerKernel(tmp_path, timeout=30) as kernel:
        assert kernel.run('1 + 1')[:2] == (True, '2')
        kernel.timeout = 0.5
        ok, _, error = kernel.run('import time\ntime.sleep(3)')
        assert not ok and 'timed out' in error
        assert kernel.closed
