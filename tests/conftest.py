"""让 `pytest profrca/tests`（宿主根目录）和 `pytest tests`（子仓库目录）都能 import profrca。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
