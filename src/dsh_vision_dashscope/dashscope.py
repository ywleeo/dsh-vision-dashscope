"""dashscope 客户端：识别图片/视频/音频。

核心设计（解决长视频问题）：
- 小视频（≤14MB 二进制）走 base64 data-uri —— DashScope data-uri 硬上限 20MB；
- 大视频/大文件走「临时 OSS 直传」：``GET /api/v1/uploads?action=getPolicy`` 拿上传
  凭证 → multipart 上传到 OSS → 得到 ``oss://`` 临时 URL（有效期 48h，上限 1GB，
  且 policy 会给出模型实际 ``max_file_size_mb``）→ chat 时在 header 加
  ``X-DashScope-OssResourceResolve: enable``，模型服务端拉取，**不压缩、无大小损失**；
- 音频：短音频（≤300s / ≤10MB）走 qwen3.5-omni 流式理解；长音频走 fun-asr
  异步转写（同样支持 oss:// 临时 URL）。
"""

from __future__ import annotations

import asyncio
import base64
import mimetypes
import time
from pathlib import Path

import httpx

from . import config

# ---- 基础工具 ----


def data_url(path: str) -> str:
    """把本地文件编码成 data-uri。"""
    mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
    raw = Path(path).read_bytes()
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def is_remote_url(value: str) -> bool:
    return value.lower().startswith(("http://", "https://"))


def resolve_local(path: str) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    return p.resolve()


def is_audio_path(path: str) -> bool:
    return Path(path).suffix.lower() in {".mp3", ".wav", ".ogg", ".oga", ".m4a", ".aac", ".flac", ".amr", ".wma"}


def is_video_path(path: str) -> bool:
    return Path(path).suffix.lower() in {".mp4", ".m4v", ".webm", ".mov", ".mkv", ".avi", ".ogv"}


def is_image_path(path: str) -> bool:
    return Path(path).suffix.lower() in {".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".avif"}


# ---- HTTP 客户端 ----


def _auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {config.api_key()}"}


async def upload_temp_oss(
    client: httpx.AsyncClient,
    model: str,
    path: str,
) -> str:
    """上传本地文件到百炼临时存储，返回 oss:// 临时 URL（模型绑定，48h 有效）。"""
    local = resolve_local(path)
    if not local.is_file():
        raise FileNotFoundError(f"找不到文件：{local}")
    policy = await _get_upload_policy(client, model)
    max_mb = policy.get("max_file_size_mb")
    if max_mb is not None and local.stat().st_size > max_mb * 1024 * 1024:
        raise RuntimeError(
            f"文件 {local.stat().st_size / 1024 / 1024:.0f}MB 超过模型 {model} 允许的 {max_mb}MB 上限"
        )
    file_name = local.name
    key = f"{policy['upload_dir']}/{file_name}"
    fields = {
        "OSSAccessKeyId": policy["oss_access_key_id"],
        "Signature": policy["signature"],
        "policy": policy["policy"],
        "x-oss-object-acl": policy["x_oss_object_acl"],
        "x-oss-forbid-overwrite": policy["x_oss_forbid_overwrite"],
        "key": key,
        "success_action_status": "200",
    }
    with local.open("rb") as handle:
        resp = await client.post(
            policy["upload_host"],
            data=fields,
            files={"file": (file_name, handle)},
            timeout=httpx.Timeout(600.0, connect=30.0),
        )
    if resp.status_code != 200:
        raise RuntimeError(f"OSS 临时上传失败（HTTP {resp.status_code}）：{resp.text[:200]}")
    return f"oss://{key}"


async def _get_upload_policy(client: httpx.AsyncClient, model: str) -> dict:
    resp = await client.get(
        f"{config.native_base()}/api/v1/uploads",
        params={"action": "getPolicy", "model": model},
        headers=_auth_headers(),
        timeout=httpx.Timeout(30.0),
    )
    if resp.status_code != 200:
        raise RuntimeError(f"获取上传凭证失败（HTTP {resp.status_code}）：{resp.text[:200]}")
    return resp.json()["data"]


# ---- Chat（图片 / 视频） ----


async def chat_media(
    client: httpx.AsyncClient,
    *,
    model: str,
    task: str,
    content_items: list[dict],
    oss_resolve: bool = False,
    fps: float | None = None,
) -> str:
    """调用 OpenAI 兼容 chat/completions（非流式）识别媒体。"""
    if fps is not None:
        content_items = [dict(item) for item in content_items]
        if 0 < fps <= 10:
            content_items[0].setdefault("fps", fps)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": task}, *content_items],
            }
        ],
        "max_tokens": config.max_tokens(),
    }
    headers = _auth_headers()
    if oss_resolve:
        headers["X-DashScope-OssResourceResolve"] = "enable"
    resp = await client.post(
        f"{config.api_base()}/chat/completions",
        headers=headers,
        json=payload,
        timeout=httpx.Timeout(600.0, connect=30.0),
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"识别失败（HTTP {resp.status_code}）：{resp.text[:300]}")
    body = resp.json()
    try:
        return body["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"识别返回异常：{str(body)[:300]}") from exc


# ---- 音频理解（qwen3.5-omni，流式） ----


async def chat_audio_omni(
    client: httpx.AsyncClient,
    *,
    model: str,
    task: str,
    audio_url: str,
) -> str:
    """用 qwen3.5-omni 流式理解音频内容。"""
    payload = {
        "model": model,
        "stream": True,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": task},
                    {"type": "input_audio", "input_audio": {"data": audio_url, "format": _audio_format(audio_url)}},
                ],
            }
        ],
        "max_tokens": config.max_tokens(),
    }
    collected: list[str] = []
    async with client.stream(
        "POST",
        f"{config.api_base()}/chat/completions",
        headers=_auth_headers(),
        json=payload,
        timeout=httpx.Timeout(600.0, connect=30.0),
    ) as resp:
        if resp.status_code >= 400:
            text = await resp.aread()
            raise RuntimeError(f"音频识别失败（HTTP {resp.status_code}）：{text[:300]}")
        async for line in resp.aiter_lines():
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if not data or data == "[DONE]":
                continue
            try:
                import json

                chunk = json.loads(data)
            except ValueError:
                continue
            delta = (chunk.get("choices") or [{}])[0].get("delta") or {}
            content = delta.get("content")
            if isinstance(content, str):
                collected.append(content)
    return "".join(collected).strip()


def _audio_format(url: str) -> str:
    suffix = Path(url.split("?")[0]).suffix.lower()
    return {
        ".mp3": "mp3",
        ".wav": "wav",
        ".ogg": "ogg",
        ".oga": "ogg",
        ".m4a": "mp4",
        ".aac": "aac",
        ".flac": "flac",
        ".amr": "amr",
    }.get(suffix, "mp3")


# ---- ASR（长音频转写，fun-asr 异步任务） ----


async def transcribe_asr(
    client: httpx.AsyncClient,
    *,
    audio_url: str,
    language: str = "zh",
) -> str:
    """通过 DashScope 异步 ASR 任务转写长音频；轮询直到完成。"""
    model = config.asr_model()
    payload = {
        "model": model,
        "input": {"file_urls": [audio_url]},
        "parameters": {"language_hints": [language]},
    }
    headers = _auth_headers()
    if audio_url.startswith("oss://"):
        headers["X-DashScope-OssResourceResolve"] = "enable"
    resp = await client.post(
        f"{config.native_base()}/api/v1/services/audio/asr/transcription",
        headers=headers,
        json=payload,
        timeout=httpx.Timeout(60.0, connect=30.0),
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"ASR 任务创建失败（HTTP {resp.status_code}）：{resp.text[:300]}")
    task_id = resp.json().get("output", {}).get("task_id")
    if not task_id:
        raise RuntimeError(f"ASR 任务创建异常：{resp.text[:300]}")

    deadline = time.monotonic() + 1800
    while time.monotonic() < deadline:
        await asyncio.sleep(5)
        status_resp = await client.get(
            f"{config.native_base()}/api/v1/tasks/{task_id}",
            headers=_auth_headers(),
            timeout=httpx.Timeout(30.0),
        )
        if status_resp.status_code >= 400:
            raise RuntimeError(f"ASR 任务查询失败（HTTP {status_resp.status_code}）：{status_resp.text[:200]}")
        output = status_resp.json().get("output", {})
        state = output.get("task_status")
        if state == "SUCCEEDED":
            results = output.get("results") or []
            texts = [item.get("transcription", "") for item in results if item.get("transcription")]
            return "\n".join(texts).strip()
        if state in {"FAILED", "CANCELED"}:
            raise RuntimeError(f"ASR 任务{state}：{str(output)[:300]}")
    raise TimeoutError("ASR 转写超时（1800s）")
