"""dsh-vision-dashscope —— 配置与环境变量加载。

所有配置均可通过环境变量覆盖；首次运行时自动从项目根目录的 .env
读取（纯手工解析，不引入 python-dotenv）。API Key 的查找顺序：
``DASH_VISION_API_KEY`` → ``OMNIMODAL_API_KEY`` → ``DASHSCOPE_API_KEY``。
"""

from __future__ import annotations

import os
from pathlib import Path

# ---- 常量 ----

# DashScope OpenAI 兼容 chat 端点。
DEFAULT_API_BASE = "https://dashscope.aliyuncs.com/compatible-mode/v1"
# DashScope 原生 API 端点（上传凭证 / ASR 任务）。
DEFAULT_NATIVE_BASE = "https://dashscope.aliyuncs.com"

DEFAULT_IMAGE_MODEL = "qwen3.7-flash"
DEFAULT_VIDEO_MODEL = "qwen3.7-flash"
DEFAULT_AUDIO_MODEL = "qwen3.5-omni-flash"
DEFAULT_ASR_MODEL = "fun-asr"
DEFAULT_LANGUAGE = "zh"

# DashScope 对 data-uri（base64 内嵌）的硬上限：20MB。
# 二进制文件 base64 膨胀 4/3，留安全余量取 14MB 二进制。
DASH_SCOPE_DATA_URI_BYTES = 20 * 1024 * 1024
DEFAULT_VIDEO_BASE64_MAX_BYTES = int(DASH_SCOPE_DATA_URI_BYTES * 3 / 4) - 512 * 1024
# 长音频转写分流阈值。
DEFAULT_AUDIO_OMNI_MAX_SEC = 300
# omni input_audio 的 base64 上限。
DEFAULT_AUDIO_BASE64_MAX_BYTES = 10 * 1024 * 1024

_ENV_FILE = Path(__file__).resolve().parent.parent.parent / ".env"


def _load_env_file() -> None:
    """加载项目根目录 .env（仅设置尚未定义的环境变量）。"""
    try:
        lines = _ENV_FILE.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_env_file()


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default).strip() or default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "").strip() or default)
    except ValueError:
        return default


def api_key() -> str:
    key = (
        os.environ.get("DASH_VISION_API_KEY")
        or os.environ.get("OMNIMODAL_API_KEY")
        or os.environ.get("DASHSCOPE_API_KEY")
        or ""
    ).strip()
    if not key:
        raise RuntimeError(
            "缺少 DashScope API Key：设置 DASH_VISION_API_KEY（或 OMNIMODAL_API_KEY / "
            "DASHSCOPE_API_KEY），或在插件根目录 .env 中配置。"
        )
    return key


def api_base() -> str:
    return _env("DASH_VISION_BASE_URL", DEFAULT_API_BASE)


def native_base() -> str:
    return _env("DASH_VISION_NATIVE_BASE_URL", DEFAULT_NATIVE_BASE)


def image_model() -> str:
    return _env("DASH_VISION_IMAGE_MODEL", DEFAULT_IMAGE_MODEL)


def video_model() -> str:
    return _env("DASH_VISION_VIDEO_MODEL", DEFAULT_VIDEO_MODEL)


def audio_model() -> str:
    return _env("DASH_VISION_AUDIO_MODEL", DEFAULT_AUDIO_MODEL)


def asr_model() -> str:
    return _env("DASH_VISION_ASR_MODEL", DEFAULT_ASR_MODEL)


def language() -> str:
    return _env("DASH_VISION_LANGUAGE", DEFAULT_LANGUAGE)


def video_base64_max_bytes() -> int:
    return _env_int("DASH_VISION_VIDEO_BASE64_MAX_MB", 0) * 1024 * 1024 or DEFAULT_VIDEO_BASE64_MAX_BYTES


def audio_omni_max_sec() -> int:
    return _env_int("DASH_VISION_AUDIO_OMNI_MAX_SEC", DEFAULT_AUDIO_OMNI_MAX_SEC)


def audio_base64_max_bytes() -> int:
    return _env_int("DASH_VISION_AUDIO_BASE64_MAX_MB", 0) * 1024 * 1024 or DEFAULT_AUDIO_BASE64_MAX_BYTES


def max_tokens() -> int:
    return _env_int("DASH_VISION_MAX_TOKENS", 4096)
