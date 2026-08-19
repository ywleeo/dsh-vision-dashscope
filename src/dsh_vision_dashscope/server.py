"""dsh-vision-dashscope MCP server：识别图片、视频、音频。

工具（FastMCP）：
- ``recognize_image`` —— 本地图片/图片 URL → 描述或按任务分析；
- ``recognize_video`` —— 本地视频/视频 URL → 描述或按任务分析（大文件临时 OSS 直传，
  无 base64 大小限制、不压缩画质；小文件走 base64 省一次上传）；
- ``recognize_audio`` —— 短音频 qwen3.5-omni 理解，长音频 fun-asr 自动转写；
- ``dashscope_status`` —— 当前模型/限制配置（不含 API Key）。

识别类工具会调用 DashScope 付费 API（按 token 计费，量小价低），标注为
openWorldHint=True，由客户端决定是否确认。
"""

from __future__ import annotations

from typing import Annotated

import httpx
from mcp.server.fastmcp import FastMCP
from mcp.types import ToolAnnotations
from pydantic import Field

from . import config, dashscope

DEFAULT_IMAGE_TASK = "详细描述这张图片的内容"
DEFAULT_VIDEO_TASK = "详细描述视频内容，按时间顺序说明关键画面、动作和字幕"
DEFAULT_AUDIO_TASK = "详细描述这段音频的内容"

EXTERNAL_SEND_ANNOTATIONS = ToolAnnotations(
    readOnlyHint=False,
    openWorldHint=True,
    destructiveHint=False,
)

mcp = FastMCP("dsh-vision-dashscope")


@mcp.tool(
    name="recognize_image",
    annotations=EXTERNAL_SEND_ANNOTATIONS,
)
async def recognize_image(
    image: Annotated[str, Field(description="本地图片绝对路径，或 http(s) 图片 URL。")],
    task: Annotated[
        str,
        Field(description="本次要从图片中提取或分析的具体内容；未传时默认详细描述图片内容。"),
    ] = DEFAULT_IMAGE_TASK,
    mode: Annotated[
        str,
        Field(description="识别档位：quick/standard/full/quick_analysis/balanced_analysis/deep_analysis，默认 standard。"),
    ] = "standard",
) -> str:
    """识别本地图片或图片 URL，按 task 调用千问多模态模型。"""
    del mode  # 保留参数位：后续可接 thinking/分辨率策略
    async with httpx.AsyncClient() as client:
        if dashscope.is_remote_url(image):
            item = {"type": "image_url", "image_url": {"url": image}}
            oss = False
        else:
            local = dashscope.resolve_local(image)
            if not local.is_file():
                raise FileNotFoundError(f"找不到图片：{local}")
            item = {"type": "image_url", "image_url": {"url": dashscope.data_url(str(local))}}
            oss = False
        return await dashscope.chat_media(
            client,
            model=config.image_model(),
            task=task.strip() or DEFAULT_IMAGE_TASK,
            content_items=[item],
            oss_resolve=oss,
        )


@mcp.tool(
    name="recognize_video",
    annotations=EXTERNAL_SEND_ANNOTATIONS,
)
async def recognize_video(
    video: Annotated[str, Field(description="本地视频绝对路径，或 http(s) 视频 URL。")],
    task: Annotated[
        str,
        Field(description="本次要从视频中提取或分析的具体内容；未传时默认详细描述视频内容。"),
    ] = DEFAULT_VIDEO_TASK,
    mode: Annotated[
        str,
        Field(description="识别档位：quick/standard/full/quick_analysis/balanced_analysis/deep_analysis，默认 standard。"),
    ] = "standard",
    fps: Annotated[
        float | None,
        Field(description="抽帧频率（每秒帧数，0.1~10，默认 2.0）；画面运动快可调高。"),
    ] = None,
) -> str:
    """识别本地视频或视频 URL。

    大文件（>14MB）自动走百炼临时 OSS 直传（上限 1GB、不压缩画质），小文件走
    base64；支持任意浏览器不可播放的容器（mkv/avi 等），由模型服务端解码抽帧。
    """
    del mode  # 保留参数位
    async with httpx.AsyncClient() as client:
        if dashscope.is_remote_url(video):
            item = {"type": "video_url", "video_url": {"url": video}}
            oss = False
        else:
            local = dashscope.resolve_local(video)
            if not local.is_file():
                raise FileNotFoundError(f"找不到视频：{local}")
            if local.stat().st_size > config.video_base64_max_bytes():
                oss_url = await dashscope.upload_temp_oss(client, config.video_model(), str(local))
                item = {"type": "video_url", "video_url": {"url": oss_url}}
                oss = True
            else:
                item = {"type": "video_url", "video_url": {"url": dashscope.data_url(str(local))}}
                oss = False
        return await dashscope.chat_media(
            client,
            model=config.video_model(),
            task=task.strip() or DEFAULT_VIDEO_TASK,
            content_items=[item],
            oss_resolve=oss,
            fps=fps,
        )


@mcp.tool(
    name="recognize_audio",
    annotations=EXTERNAL_SEND_ANNOTATIONS,
)
async def recognize_audio(
    audio: Annotated[str, Field(description="本地音频绝对路径，或 http(s) 音频 URL。")],
    task: Annotated[
        str,
        Field(description="本次要从音频中提取或分析的具体内容；未传时默认详细描述音频内容。"),
    ] = DEFAULT_AUDIO_TASK,
    language: Annotated[
        str,
        Field(description="长音频转写时的语言提示（zh/en/ja/yue/ko/de/fr/ru），默认 zh。"),
    ] = "zh",
) -> str:
    """识别本地音频或音频 URL。

    短音频（≤300s 且 ≤10MB）走 qwen3.5-omni 直接理解内容；长音频自动走
    fun-asr 异步转写（本地文件先上传到临时 OSS），返回纯文本转录。
    """
    async with httpx.AsyncClient() as client:
        if dashscope.is_remote_url(audio):
            url = audio
            local = None
        else:
            local = dashscope.resolve_local(audio)
            if not local.is_file():
                raise FileNotFoundError(f"找不到音频：{local}")
            url = str(local)

        # 长音频或超 base64 上限 → 临时 OSS + fun-asr 转写。
        size = local.stat().st_size if local is not None else 0
        should_transcribe = size > config.audio_base64_max_bytes() or (
            local is not None and _audio_too_long(local)
        )
        if should_transcribe:
            if local is not None:
                audio_url = await dashscope.upload_temp_oss(client, config.asr_model(), str(local))
            else:
                audio_url = url
            return await dashscope.transcribe_asr(client, audio_url=audio_url, language=language)

        # 短音频 → qwen3.5-omni 理解。
        if local is not None:
            data = dashscope.data_url(str(local))
        else:
            data = url
        return await dashscope.chat_audio_omni(
            client,
            model=config.audio_model(),
            task=task.strip() or DEFAULT_AUDIO_TASK,
            audio_url=data,
        )


def _audio_too_long(local) -> bool:
    """粗略判断音频时长是否超过 omni 理解阈值（用文件大小近似 + 常见码率）。"""
    try:
        import subprocess

        probe = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", "-of", "csv=p=0", str(local)],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if probe.returncode == 0 and probe.stdout.strip():
            return float(probe.stdout.strip()) > config.audio_omni_max_sec()
    except Exception:
        pass
    # 无 ffprobe 时按码率近似：~16KB/s（128kbps）估算。
    return local.stat().st_size > config.audio_omni_max_sec() * 16 * 1024


@mcp.tool(name="dashscope_status")
async def dashscope_status() -> str:
    """查看当前模型与限制配置（不含 API Key）。"""
    return (
        "dsh-vision-dashscope 配置：\n"
        f"- 图片模型：{config.image_model()}\n"
        f"- 视频模型：{config.video_model()}（base64 阈值 {config.video_base64_max_bytes() / 1024 / 1024:.0f}MB，"
        f"超出走临时 OSS 直传）\n"
        f"- 音频理解模型：{config.audio_model()}（≤{config.audio_omni_max_sec()}s）\n"
        f"- 长音频 ASR 模型：{config.asr_model()}\n"
        f"- API 端点：{config.api_base()}"
    )
