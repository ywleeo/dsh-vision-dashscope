"""控制台入口：直接运行 MCP server（stdio）。"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    del argv
    from .server import mcp

    try:
        mcp.run(transport="stdio")
    except KeyboardInterrupt:
        return 0
    except Exception as exc:  # pragma: no cover - 启动期异常
        print(f"dsh-vision-dashscope 启动失败：{exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
