"""server 基础测试：工具注册与配置（不调用外部 API）。"""

from __future__ import annotations

import pytest


@pytest.fixture(scope="module")
def tools():
    from dsh_vision_dashscope.server import mcp

    manager = getattr(mcp, "_tool_manager", None)
    if manager is None:
        pytest.skip("FastMCP 版本不暴露 _tool_manager")
    items = manager.list_tools()
    return sorted(item.name for item in items)


def test_tools_registered(tools):
    assert "recognize_image" in tools
    assert "recognize_video" in tools
    assert "recognize_audio" in tools
    assert "dashscope_status" in tools


def test_config_defaults():
    from dsh_vision_dashscope import config

    assert config.image_model() == "qwen3.7-flash"
    assert config.video_model() == "qwen3.7-flash"
    assert config.audio_model() == "qwen3.5-omni-flash"
    assert config.asr_model() == "fun-asr"
    # 视频 base64 阈值应低于 DashScope 20MB data-uri 上限。
    assert config.video_base64_max_bytes() < 20 * 1024 * 1024


def test_media_helpers():
    from dsh_vision_dashscope import dashscope

    assert dashscope.is_remote_url("https://a.com/b.mp4")
    assert not dashscope.is_remote_url("/Users/x/a.mp4")
    assert dashscope.is_video_path("/x/a.MKV")
    assert dashscope.is_image_path("/x/a.png")
    assert dashscope.is_audio_path("/x/a.mp3")
