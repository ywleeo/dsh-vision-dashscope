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
    assert "generate_image" in tools
    assert "generate_video" in tools
    assert "generate_video_from_image" in tools


def test_generation_config_defaults():
    from dsh_vision_dashscope import config

    assert config.image_generation_model() == "qwen-image-3.0"
    assert config.image_generation_model("pro") == "wan2.7-image-pro"
    assert config.video_generation_model() == "wan2.7-t2v"
    assert config.video_generation_model("standard", "i2v") == "wan2.7-i2v"
    assert config.video_generation_model("max") == "happyhorse-1.1-t2v"
    assert config.max_video_duration() >= 5


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


def test_make_jpeg_preview(tmp_path):
    from pathlib import Path

    from PIL import Image

    from dsh_vision_dashscope import dashscope

    src = tmp_path / "big.png"
    Image.new("RGB", (2400, 1800), (200, 100, 50)).save(src)
    preview = dashscope.make_jpeg_preview(src, tmp_path, max_dim=1200, quality=80)
    assert preview is not None and preview.exists()
    assert preview.suffix == ".jpg"
    assert preview.stat().st_size < 5 * 1024 * 1024
    with Image.open(preview) as im:
        assert max(im.size) <= 1200
    # 原图保留
    assert src.exists()
    # 输入已是 JPEG 时不再重复转换
    assert dashscope.make_jpeg_preview(preview, tmp_path) is None


def test_generation_config_defaults():
    from dsh_vision_dashscope import config

    assert config.image_generation_model() == "qwen-image-3.0"
    assert config.image_generation_model("pro") == "wan2.7-image-pro"
    assert config.video_generation_model() == "wan2.7-t2v"
    assert config.video_generation_model("standard", "i2v") == "wan2.7-i2v"
    assert config.video_generation_model("max") == "happyhorse-1.1-t2v"
    assert config.max_video_duration() >= 5
    assert config.image_preview_jpeg_enabled() is True
