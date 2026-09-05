#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
独立生图脚本（不走 MCP，不受 60s 客户端死线限制）。

用法：
  python3 gen_image.py "一只戴礼帽的橘猫，暖色调插画"
  python3 gen_image.py "prompt" --size 1024*1024 --tier standard --out /path/to/dir

说明：
  - 同步调用 DashScope 文生图（multimodal-generation / qwen-image-3.0），
    自己等完成后保存，超时给足（默认 180s），不存在 MCP 那层超时。
  - API Key 从插件根目录 .env 读取（DASH_VISION_API_KEY）。
  - 成功打印保存文件的绝对路径，供对话内联预览（配合 dsh-image-preview）。

依赖：httpx（pip install httpx，或 pip install -e . 装到插件环境即可）。
"""

import argparse
import os
import pathlib
import time

import httpx

ENDPOINT = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"

MODELS = {
    "standard": "qwen-image-3.0",
    "pro": "wan2.7-image-pro",
    "max": "qwen-image-3.0-pro",
}

ENV_FILE = pathlib.Path(__file__).resolve().parent / ".env"
DEFAULT_OUT = pathlib.Path.home() / "Downloads" / "dsh-vision-dashscope"


def _load_env() -> None:
    if not ENV_FILE.is_file():
        return
    for line in ENV_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def api_key() -> str:
    key = (os.environ.get("DASH_VISION_API_KEY") or
           os.environ.get("OMNIMODAL_API_KEY") or
           os.environ.get("DASHSCOPE_API_KEY") or "").strip()
    if not key:
        raise RuntimeError("缺少 DashScope API Key：请在插件根目录 .env 设置 DASH_VISION_API_KEY。")
    return key


def sanitize_filename(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "-_.") else "_" for c in name).strip("._") or "output"


def main() -> None:
    _load_env()
    parser = argparse.ArgumentParser(description="同步文生图（image-3.0，不受 MCP 超时限制）")
    parser.add_argument("prompt", help="图片内容描述")
    parser.add_argument("--size", default=None, help="尺寸，如 1024*1024 或档位 1K/2K/4K")
    parser.add_argument("--tier", default="standard", choices=list(MODELS), help="档位：standard/pro/max")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="输出目录（默认 ~/Downloads/dsh-vision-dashscope）")
    parser.add_argument("--timeout", type=float, default=180.0, help="等待超时秒数（默认 180）")
    args = parser.parse_args()

    model = MODELS[args.tier]
    payload = {
        "model": model,
        "input": {"messages": [{"role": "user", "content": [{"text": args.prompt}]}]},
        "parameters": {"n": 1, "watermark": False},
    }
    if args.size:
        payload["parameters"]["size"] = args.size

    print(f"[gen_image] 提交 {model}（tier={args.tier}），等待不超过 {args.timeout:.0f}s……", flush=True)
    start = time.time()
    resp = httpx.post(
        ENDPOINT,
        headers={"Authorization": f"Bearer {api_key()}"},
        json=payload,
        timeout=httpx.Timeout(args.timeout, connect=30.0),
    )
    if resp.status_code >= 400:
        raise RuntimeError(f"文生图失败（HTTP {resp.status_code}）：{resp.text[:300]}")

    urls = []
    for choice in resp.json().get("output", {}).get("choices", []):
        for item in (choice.get("message") or {}).get("content") or []:
            image = item.get("image")
            if isinstance(image, str) and image:
                urls.append(image)
    if not urls:
        raise RuntimeError(f"文生图成功但响应中没有图片：{resp.text[:300]}")

    out_dir = pathlib.Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    saved = []
    for idx, url in enumerate(urls):
        fname = sanitize_filename(f"img-{args.tier}-{int(time.time())}-{idx}.png")
        target = out_dir / fname
        dl = httpx.get(url, timeout=httpx.Timeout(120.0), follow_redirects=True)
        if dl.status_code >= 400:
            raise RuntimeError(f"下载结果失败（HTTP {dl.status_code}）：{url[:120]}")
        target.write_bytes(dl.content)
        saved.append(target)

    for p in saved:
        print(f"[gen_image] 已保存：{p}")
    print(f"[gen_image] 耗时 {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
