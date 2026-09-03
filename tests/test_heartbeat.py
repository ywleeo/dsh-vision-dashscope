"""_heartbeat_t 单元测试：验证它并发执行主协程、周期上报进度、完成时取消心跳。

用假的 ctx（一个记录 report_progress 调用的对象）即可，不依赖真实传输层——
传输层何时收到 notifications/progress 由 FastMCP 的 ctx.report_progress 负责
（已单独确认），本处只验证我们包装逻辑正确。
"""

from __future__ import annotations

import asyncio

import pytest

from dsh_vision_dashscope.server import _heartbeat_t


class FakeCtx:
    """记录 report_progress 调用（异步签名与 FastMCP Context 一致）。"""

    def __init__(self) -> None:
        self.calls: list[tuple[float, float | None, str | None]] = []

    async def report_progress(self, progress: float, total: float | None = None, message: str | None = None) -> None:
        self.calls.append((progress, total, message))


async def test_reports_progress_halfway_and_returns_result() -> None:
    ctx = FakeCtx()

    async def work() -> str:
        await asyncio.sleep(0.3)
        return "done"

    # 心跳间隔 0.05s，0.3s 的主任务应至少捕获 ≥1 次进度。
    result = await _heartbeat_t(ctx, work(), heartbeat_interval=0.05, label="测试")
    assert result == "done"
    assert len(ctx.calls) >= 1
    # 每次上报 progress=0, total=100，消息为给定 label。
    assert all(c[0] == 0 and c[1] == 100 for c in ctx.calls)
    assert all(c[2] == "测试" for c in ctx.calls)


async def test_heartbeat_cancelled_after_completion() -> None:
    ctx = FakeCtx()

    async def work() -> str:
        await asyncio.sleep(0.1)
        return "ok"

    await _heartbeat_t(ctx, work(), heartbeat_interval=0.02)
    # 完成后心跳任务应被取消，不再有新的 report_progress 调用。
    before = len(ctx.calls)
    await asyncio.sleep(0.05)
    assert len(ctx.calls) == before


async def test_passthrough_without_ctx() -> None:
    # 当未传 ctx（模拟客户端没带 progressToken / 直接快捷入口）时应正常返回。
    async def work() -> int:
        return 7

    assert await _heartbeat_t(None, work(), heartbeat_interval=0.01) == 7  # type: ignore[arg-type]
