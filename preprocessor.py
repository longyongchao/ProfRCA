"""日志预处理器（取自宿主项目 LILAC）：日志头剥离、正则掩码、Token化"""

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

PLACEHOLDER = "<*>"
LEVEL_WORD = r"DEBUG|INFO|INFORMATION|NOTICE|WARN(?:ING)?|ERROR|ERR|FATAL|CRITICAL|TRACE|SEVERE"
LEVEL_TOKEN_RE = re.compile(rf'(?<![A-Za-z0-9_])({LEVEL_WORD})(?![A-Za-z0-9_])')

# PostgreSQL jsonlog severity → 标准级别
PG_SEVERITY_MAP = {
    "DEBUG": "DEBUG",
    "LOG": "INFO",
    "INFO": "INFO",
    "NOTICE": "INFO",
    "WARNING": "WARNING",
    "ERROR": "ERROR",
    "FATAL": "FATAL",
    "PANIC": "FATAL",
}

# JSON 日志中不作为 metadata 展示的字段（已单独提取）
_JSON_DISPLAY_KEYS = frozenset({"timestamp", "error_severity", "message"})
_JSON_MESSAGE_EQUALS_RE = re.compile(
    r'(\b[A-Za-z_][\w.-]*\s*=\s*)("[^"]*"|\'[^\']*\'|[^\s,\]\)\};]+)'
)
_JSON_MESSAGE_COLON_VALUE_RE = re.compile(
    r'(\b[A-Za-z_][\w.-]*:\s*)("[^"]*"|\'[^\']*\'|0x[0-9a-fA-F]+|[0-9a-fA-F]+/[0-9a-fA-F]+|\d+(?:\.\d+)?)'
)
_JSON_MESSAGE_TIMESTAMP_RE = re.compile(
    r'\b\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:\s+[A-Z]{2,4})?\b'
)
_JSON_MESSAGE_IPV4_RE = re.compile(r'\b\d{1,3}(?:\.\d{1,3}){3}\b')
_JSON_MESSAGE_PATH_RE = re.compile(r'(?:/[\w.\-]+){2,}')
_JSON_MESSAGE_LSN_RE = re.compile(r'\b[0-9A-Fa-f]+/[0-9A-Fa-f]+\b')
_JSON_MESSAGE_NUMBER_RE = re.compile(r'\b\d+(?:\.\d+)?\b')

# ============================================================
# 日志头格式定义（按优先级排列）
# ============================================================

HEADER_PATTERNS = [
    # SunDB system.trc
    (
        "sundb_system",
        re.compile(
            r'^\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)'
            r'\s+INSTANCE\((\w+)\)'
            r'\s+THREAD\((\d+),(\d+)\)\]'
            r'\s+\[(\w+)\]'
        ),
        ["timestamp", "instance", "thread_pid", "thread_tid", "level"],
    ),
    # SunDB simple (listener/CDC/gmon)
    (
        "sundb_simple",
        re.compile(
            r'^\[(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d+)'
            r'\s+THREAD\((\d+),(\d+)\)\]'
        ),
        ["timestamp", "thread_pid", "thread_tid"],
    ),
    # OpenStack: 2017-05-16 00:00:00.008 25746 INFO nova.component [req-...] message
    (
        "openstack",
        re.compile(
            rf'^(\S+)\s+'
            rf'(\d{{4}}-\d{{2}}-\d{{2}}\s+\d{{2}}:\d{{2}}:\d{{2}}(?:[\.,]\d+)?)\s+'
            rf'(\d+)\s+({LEVEL_WORD})\s+(\S+)'
            rf'(?:\s+(\[[^\]]+\]))?\s*'
        ),
        ["logrecord", "timestamp", "pid", "level", "component", "context"],
    ),
    # Hadoop/log4j: 2015-10-18 18:01:47,978 INFO [main] org.xxx.Class: message
    (
        "log4j_component",
        re.compile(
            rf'^(\d{{4}}-\d{{2}}-\d{{2}}\s+\d{{2}}:\d{{2}}:\d{{2}}(?:[\.,]\d+)?)\s+'
            rf'({LEVEL_WORD})\s+'
            rf'(?:\[([^\]]+)\]\s+)?'
            rf'(\S+?):\s*'
        ),
        ["timestamp", "level", "process", "component"],
    ),
    # HDFS legacy: 081109 203615 148 INFO dfs.DataNode$PacketResponder: message
    (
        "hdfs_legacy",
        re.compile(
            rf'^(\d{{6}})\s+(\d{{6}})\s+(\d+)\s+({LEVEL_WORD})\s+(\S+?):\s*'
        ),
        ["date", "time", "pid", "level", "component"],
    ),
    # Syslog: Mar 15 10:23:01 hostname prog[pid]: message
    (
        "syslog",
        re.compile(
            r'^([A-Z][a-z]{2}\s+\d+\s+\d{2}:\d{2}:\d{2})'
            r'\s+(\S+)\s+(\S+?)(?:\[(\d+)\])?:\s*'
        ),
        ["timestamp", "hostname", "program", "pid"],
    ),
    # ISO8601 with level
    (
        "iso_level",
        re.compile(
            rf'^(\d{{4}}-\d{{2}}-\d{{2}}[\sT]\d{{2}}:\d{{2}}:\d{{2}}(?:[\.,]\d+)?)'
            rf'\s+({LEVEL_WORD})\s+'
        ),
        ["timestamp", "level"],
    ),
    # ISO8601 timestamp only
    (
        "iso_only",
        re.compile(
            r'^(\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}(?:[\.,]\d+)?)\s+'
        ),
        ["timestamp"],
    ),
]

TIMESTAMP_PATTERNS = [
    ("iso_timestamp", re.compile(r'\b\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2}(?:[\.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b')),
    ("compact_hdfs_timestamp", re.compile(r'\b\d{6}\s+\d{6}\b')),
    ("syslog_timestamp", re.compile(r'\b[A-Z][a-z]{2}\s+\d{1,2}\s+\d{2}:\d{2}:\d{2}\b')),
    ("epoch_timestamp", re.compile(r'\b\d{10}(?:\.\d+)?\b')),
]

# ============================================================
# 正则掩码规则（按顺序应用）
# ============================================================

MASK_PATTERNS = [
    # UUID
    (re.compile(r'[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}'), PLACEHOLDER),
    # IPv4
    (re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'), PLACEHOLDER),
    # Common Hadoop/HDFS runtime identifiers
    (re.compile(r'\bblk_-?\d+\b'), PLACEHOLDER),
    (re.compile(r'\bappattempt_\d+_\d+_\d+\b'), PLACEHOLDER),
    (re.compile(r'\bapplication_\d+_\d+\b'), PLACEHOLDER),
    (re.compile(r'\bcontainer_\d+_\d+_\d+_\d+\b'), PLACEHOLDER),
    (re.compile(r'\battempt_\d+(?:_\d+)*[A-Za-z0-9_]*\b'), PLACEHOLDER),
    # HTTP request line
    (re.compile(r'"(GET|POST|PUT|DELETE|PATCH|HEAD|OPTIONS)\s+[^"]+\s+HTTP/\d(?:\.\d+)?"'), r'"\1 <*>"'),
    # Key-value numeric patterns
    (re.compile(r'\b(status|len|size|time|duration|elapsed|latency|cost|port|id|attemptId|keyId|startIndex|maxEvents)\s*([:=])\s*-?\d+(?:\.\d+)?', re.IGNORECASE), r'\1\2 <*>'),
    # Context-specific short numbers
    (re.compile(r'\b(PacketResponder)\s+\d+\b'), r'\1 <*>'),
    # Hex values (0x...)
    (re.compile(r'\b0x[0-9a-fA-F]+\b'), PLACEHOLDER),
    # Floating point values
    (re.compile(r'(?<![\w.])-?\d+\.\d+(?![\w.])'), PLACEHOLDER),
    # File paths
    (re.compile(r'(?:/[\w.\-]+){2,}'), PLACEHOLDER),
    # Signed long numbers
    (re.compile(r'(?<![\w])-?\d{4,}\b'), PLACEHOLDER),
    # Long numbers (4+ digits)
    (re.compile(r'\b\d{4,}\b'), PLACEHOLDER),
    # Numeric values in key=value pairs (e.g. exec_time_seconds=32, num_lora=0)
    # This catches small numbers that the 4+-digit rule misses, preventing the
    # static shortcut from firing when the log body actually contains variables.
    (re.compile(r'(?<==)\d+(?:\.\d+)?(?=[\s,\]\)\};]|$)', re.MULTILINE), PLACEHOLDER),
]


@dataclass
class PreprocessedLine:
    """预处理后的日志行"""
    raw_text: str
    header_fields: Dict[str, str] = field(default_factory=dict)
    header_format: str = ""
    body: str = ""
    template_body: str = ""
    masked_body: str = ""
    tokens: List[str] = field(default_factory=list)
    token_count: int = 0
    first_token: str = ""


class LogPreprocessor:
    """日志预处理器：头部剥离 → 正则掩码 → Token化"""

    def __init__(
        self,
        header_patterns: Optional[List] = None,
        mask_patterns: Optional[List] = None,
    ):
        self._header_patterns = header_patterns or HEADER_PATTERNS
        self._mask_patterns = mask_patterns or MASK_PATTERNS

    def preprocess(self, raw_line: str) -> PreprocessedLine:
        raw_line = raw_line.rstrip("\n\r")
        if not raw_line.strip():
            return PreprocessedLine(raw_text=raw_line)

        json_result = self._try_parse_json_line(raw_line)
        if json_result is not None:
            header_fields, header_format, body, template_body = json_result
        else:
            header_fields, header_format, body = self._strip_header(raw_line)
            template_body = body

        masked_body = self._apply_masks(template_body)
        tokens = self._tokenize(masked_body)

        return PreprocessedLine(
            raw_text=raw_line,
            header_fields=header_fields,
            header_format=header_format,
            body=body,
            template_body=template_body,
            masked_body=masked_body,
            tokens=tokens,
            token_count=len(tokens),
            first_token=tokens[0] if tokens else "",
        )

    def _try_parse_json_line(
        self, line: str,
    ) -> Optional[Tuple[Dict[str, str], str, str, str]]:
        """解析 JSON 结构化日志行（如 PostgreSQL jsonlog）。"""
        stripped = line.strip()
        if not stripped.startswith("{"):
            return None
        try:
            obj = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        if not isinstance(obj, dict):
            return None

        header_fields: Dict[str, str] = {}
        for key, value in obj.items():
            if value is None or key in _JSON_DISPLAY_KEYS:
                continue
            if isinstance(value, (str, int, float, bool)):
                header_fields[key] = str(value)

        if "timestamp" in obj and obj["timestamp"] is not None:
            header_fields["timestamp"] = str(obj["timestamp"])

        severity = obj.get("error_severity") or obj.get("level") or ""
        if severity:
            header_fields["level"] = PG_SEVERITY_MAP.get(
                str(severity).upper(), str(severity).upper()
            )

        message = obj.get("message", "")
        if message is None:
            message = ""
        elif not isinstance(message, str):
            message = str(message)

        template_body = self._mask_json_message_values(message)
        return header_fields, "json_log", message, template_body

    @staticmethod
    def _mask_json_message_values(message: str) -> str:
        """JSON 日志用 message 提取模板，并预先掩码常见 key=value 动态值。"""
        if not message:
            return message
        message = _JSON_MESSAGE_EQUALS_RE.sub(rf"\1{PLACEHOLDER}", message)
        message = _JSON_MESSAGE_COLON_VALUE_RE.sub(rf"\1{PLACEHOLDER}", message)
        message = _JSON_MESSAGE_TIMESTAMP_RE.sub(PLACEHOLDER, message)
        message = _JSON_MESSAGE_IPV4_RE.sub(PLACEHOLDER, message)
        message = _JSON_MESSAGE_PATH_RE.sub(PLACEHOLDER, message)
        message = _JSON_MESSAGE_LSN_RE.sub(PLACEHOLDER, message)
        return _JSON_MESSAGE_NUMBER_RE.sub(PLACEHOLDER, message)

    def _strip_header(self, line: str) -> Tuple[Dict[str, str], str, str]:
        for fmt_name, pattern, field_names in self._header_patterns:
            m = pattern.match(line)
            if m:
                fields = {}
                for i, name in enumerate(field_names):
                    val = m.group(i + 1)
                    if val is not None:
                        fields[name] = val
                fields = self._normalize_header_fields(fields, fmt_name)
                body = self._clean_body(line[m.end():])
                return fields, fmt_name, body
        return self._infer_header(line)

    def _infer_header(self, line: str) -> Tuple[Dict[str, str], str, str]:
        """Field-level fallback for unseen log formats."""
        fields: Dict[str, str] = {}
        ts_match, ts_kind = self._find_timestamp(line)
        cursor = 0

        if ts_match:
            fields["timestamp"] = ts_match.group(0)
            cursor = ts_match.end()

        level_match = self._find_level(line, cursor, bool(ts_match))
        if level_match:
            fields["level"] = level_match.group(1).upper()
            cursor = level_match.end()

        if not fields:
            return {}, "unknown", line

        body = self._clean_body(line[cursor:])
        if not body:
            body = line

        fmt_name = f"generic_{ts_kind}" if ts_kind else "generic_header"
        fields = self._normalize_header_fields(fields, fmt_name)
        return fields, fmt_name, body

    def _find_timestamp(self, line: str):
        best = None
        best_kind = ""
        for kind, pattern in TIMESTAMP_PATTERNS:
            m = pattern.search(line[:160])
            if m and (best is None or m.start() < best.start()):
                best = m
                best_kind = kind
        return best, best_kind

    def _find_level(self, line: str, cursor: int, has_timestamp: bool):
        """Find log level near the beginning of the line after cursor."""
        search_region = line[cursor:cursor + 80]
        m = LEVEL_TOKEN_RE.search(search_region)
        if m and m.start() < 40:
            # Adjust match positions to full line
            class _Match:
                def __init__(self, val, s, e):
                    self._val = val
                    self._s = s
                    self._e = e
                def group(self, i=0):
                    return self._val if i <= 1 else None
                def start(self):
                    return self._s
                def end(self):
                    return self._e
            return _Match(m.group(1), cursor + m.start(), cursor + m.end())
        return None

    @staticmethod
    def _normalize_header_fields(fields: Dict[str, str], fmt_name: str) -> Dict[str, str]:
        """Normalize extracted header fields.

        Note: SunDB formats preserve original level names (e.g. INFORMATION)
        since they are meaningful in that context.
        """
        if "level" in fields and not fmt_name.startswith("sundb"):
            fields["level"] = fields["level"].upper()
            if fields["level"] == "INFORMATION":
                fields["level"] = "INFO"
        return fields

    @staticmethod
    def _clean_body(text: str) -> str:
        """Strip leading/trailing whitespace and common separators from body."""
        text = text.strip()
        if text.startswith(": "):
            text = text[2:]
        elif text.startswith(":"):
            text = text[1:].lstrip()
        return text

    def _apply_masks(self, text: str) -> str:
        for pattern, replacement in self._mask_patterns:
            text = pattern.sub(replacement, text)
        return text

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        if not text:
            return []
        return text.split()
