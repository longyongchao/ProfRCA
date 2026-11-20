"""Cross-platform RCA-Agent executor: only telemetry enters a disposable container."""
from __future__ import annotations

import json
import queue
import shutil
import subprocess
import threading
import uuid
from pathlib import Path

from ..errors import ExecutionUnavailable


class DockerKernel:
    """Stateful IPython JSON-lines worker, without host credentials or network.

    Uses an already-built image; never pulls/builds images implicitly. Only the
    telemetry directory and worker source are mounted, read-only. A timed-out
    cell terminates this kernel; stale responses cannot become the next reply.
    """
    def __init__(self, workdir, image='profrca-kernel:1', timeout=300):
        if timeout <= 0:
            raise ValueError('cell timeout must be positive')
        self.timeout = timeout
        self.name = 'profrca-kernel-' + uuid.uuid4().hex
        self.p = None
        self.closed = False
        self.started = False
        executable = shutil.which('docker')
        if not executable:
            raise ExecutionUnavailable('RCA-Agent requires Docker; see docs/UNIFIED_API.md for kernel setup')
        self.docker = executable
        try:
            subprocess.run([executable, 'image', 'inspect', image], check=True, capture_output=True, timeout=20)
        except (subprocess.SubprocessError, OSError) as exc:
            raise ExecutionUnavailable('Start Docker and build the RCA-Agent kernel image: docker build -t profrca-kernel:1 -f docker/Dockerfile.kernel .') from exc
        telemetry = Path(workdir).resolve() / 'telemetry'
        worker = Path(__file__).resolve().parent / 'engines' / 'kernel_worker.py'
        command = [executable, 'run', '--rm', '-i', '--name', self.name,
                   '--network', 'none', '--read-only', '--cap-drop', 'ALL',
                   '--security-opt', 'no-new-privileges', '--pids-limit', '64',
                   '--memory', '1g', '--cpus', '1', '--user', '65534:65534',
                   '--tmpfs', '/tmp:rw,nosuid,nodev,size=64m',
                   '--tmpfs', '/work:rw,nosuid,nodev,size=256m,mode=1777',
                   '--mount', f'type=bind,source={telemetry},target=/work/telemetry,readonly',
                   '--mount', f'type=bind,source={worker},target=/kernel_worker.py,readonly',
                   '-e', 'HOME=/tmp', '-e', 'IPYTHONDIR=/tmp/ipython',
                   '-e', 'OPENBLAS_NUM_THREADS=1', '-e', 'OMP_NUM_THREADS=1',
                   '--workdir', '/work', image, 'python', '-B', '/kernel_worker.py']
        self.messages = queue.Queue()
        self.p = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, encoding='utf-8', bufsize=1)
        self.started = True
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        try:
            for line in self.p.stdout:
                self.messages.put(line)
        finally:
            self.messages.put(None)

    def run(self, code):
        if self.closed:
            return False, '', 'Kernel is closed'
        try:
            self.p.stdin.write(json.dumps({'code': code}) + '\n')
            self.p.stdin.flush()
            line = self.messages.get(timeout=self.timeout)
            if line is None:
                self.close()
                return False, '', 'Docker kernel exited; check image and telemetry mount permissions'
            result = json.loads(line)
            return bool(result['success']), result.get('result', ''), result.get('error')
        except queue.Empty:
            self.close()
            return False, '', 'Cell timed out; kernel terminated and variables were discarded'
        except (OSError, ValueError, KeyError) as exc:
            self.close()
            return False, '', type(exc).__name__ + ': kernel protocol failed'

    def close(self):
        if self.closed:
            return
        self.closed = True
        # Exact task-generated container name only; no pruning or broad stop commands.
        if self.started:
            try:
                subprocess.run([self.docker, 'rm', '-f', self.name], capture_output=True, timeout=20)
            except (OSError, subprocess.SubprocessError):
                pass
        if self.p is not None:
            if self.p.poll() is None:
                self.p.kill()
            self.p.wait(timeout=10)
            self.p.stdin.close()
            self.p.stdout.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
