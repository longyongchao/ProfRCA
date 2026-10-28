"""结构化日志条目的数据模型（取自宿主项目 LILAC 的 LogTemplate / GenericLogEntry，去掉了 SunDB 专用转换）。"""

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional


@dataclass
class LogTemplate:
    """日志模板"""
    template_id: str
    template_str: str
    tokens: List[str]
    token_count: int
    first_token: str
    hit_count: int = 0
    created_at: float = field(default_factory=time.time)
    last_hit_at: float = field(default_factory=time.time)
    source: Literal["llm", "drain3", "seed", "manual", "static"] = "llm"

    @staticmethod
    def generate_id(template_str: str) -> str:
        return hashlib.sha256(template_str.encode()).hexdigest()[:16]

    @classmethod
    def from_template_str(
        cls, template_str: str, source: str = "llm"
    ) -> "LogTemplate":
        tokens = template_str.split()
        return cls(
            template_id=cls.generate_id(template_str),
            template_str=template_str,
            tokens=tokens,
            token_count=len(tokens),
            first_token=tokens[0] if tokens else "",
            source=source,
        )


@dataclass
class GenericLogEntry:
    """通用日志条目"""
    timestamp: str = ""
    level: str = ""
    message: str = ""
    template: Optional[LogTemplate] = None
    parameters: List[str] = field(default_factory=list)
    source_file: str = ""
    line_number: int = 0
    raw_text: str = ""
    metadata: Dict[str, str] = field(default_factory=dict)
