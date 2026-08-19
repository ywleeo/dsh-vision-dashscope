# dsh-vision-dashscope

通过 **DashScope / 千问（Qwen）** 识别**并生成**图片、视频、音频的 DSH 插件（MCP server）。专治 Deepseek-omnimodal 等插件「视频大了就不行」的问题：

- **长视频/大视频直接识别**：走百炼**临时 OSS 直传**（`oss://` 临时 URL，单文件上限 1GB，按模型的 `max_file_size_mb` 校验），**不压缩、不损画质**；小视频（≤14MB）走 base64 省一次上传。
- **mkv / avi 等浏览器不可播格式**也能识别（模型服务端解码抽帧）。
- 音频：短音频 qwen3.5-omni 直接理解，长音频自动 fun-asr 转写。

## 工具

| 工具 | 说明 |
|---|---|
| `recognize_image` | 本地图片 / 图片 URL → 描述或按任务分析（qwen3.7-flash） |
| `recognize_video` | 本地视频 / 视频 URL → 描述或按任务分析（qwen3.7-flash；大文件 OSS 直传） |
| `recognize_audio` | 短音频 qwen3.5-omni 理解；长音频 fun-asr 转写 |
| `generate_image` | 文生图（qwen-image-3.0 等；约 0.18 元/张起） |
| `generate_video` | 文生视频（wan2.7-t2v / happyhorse-1.1-t2v；约 0.6 元/秒起） |
| `generate_video_from_image` | 图生视频（wan2.7-i2v 等，首帧图片自动上传临时 OSS） |
| `dashscope_status` | 查看当前模型与限制配置 |

- 识别工具会调用 DashScope 付费 API（按 token 计费，量小价低）。
- **生成工具必须 `confirm=true` 才会实际调用付费接口**；`false` 时只返回预计费用（如"预计 3.00 元（5 秒 × 0.6 元/秒）"）。
- 生成结果自动下载到本地输出目录（默认 `~/Downloads/dsh-vision-dashscope`，可用 `DASH_VISION_OUTPUT_DIR` 修改），返回本地路径——配合 dsh-image-preview 可直接在对话里内联预览。

## 安装

需要 Python 3.12+、uv。复制 `.env.example` 为 `.env` 并填写 API Key：

```bash
git clone https://github.com/ywleeo/dsh-vision-dashscope.git
cd dsh-vision-dashscope
cp .env.example .env    # 填入 DASH_VISION_API_KEY
uv sync
```

API Key 也可用环境变量：`DASH_VISION_API_KEY`（缺省回退 `OMNIMODAL_API_KEY` / `DASHSCOPE_API_KEY`）。

## DSH 接入

在 `~/.dsh/profiles/web/cordis.patch.yml`（用户 patch 层）追加：

```yaml
- insert:
    - id: mcp-vision-dashscope
      name: '@deepseek-ai/dsh-mcp-client'
      config:
        serverName: vision-dashscope
        transport: stdio
        command: /Users/leeo/.local/bin/uv
        args:
          - run
          - --directory
          - /path/to/dsh-vision-dashscope
          - dsh-vision-dashscope
        failOnStartupError: false
```

重启 `dsh web`。连接后工具以 `mcp__vision-dashscope__recognize_video` 等名字出现。

## 大文件上传原理（长视频关键）

1. `GET /api/v1/uploads?action=getPolicy&model=<model>` 获取上传凭证（含 `upload_host`、签名、`max_file_size_mb`）；
2. multipart 上传文件到 OSS 临时存储，得到 `oss://<key>` 临时 URL（有效期 48h，免费）；
3. chat 请求 `video_url: {"url": "oss://..."}` 并在 header 加 `X-DashScope-OssResourceResolve: enable`，模型服务端拉取解码抽帧。

参考：[上传文件获取临时 URL（阿里云百炼文档）](https://help.aliyun.com/zh/model-studio/get-temporary-file-url)、[视觉理解（视频）](https://www.alibabacloud.com/help/en/model-studio/vision)。

## 配置（环境变量 / .env）

| 变量 | 默认 | 说明 |
|---|---|---|
| `DASH_VISION_API_KEY` | — | API Key（回退 OMNIMODAL_API_KEY / DASHSCOPE_API_KEY） |
| `DASH_VISION_IMAGE_MODEL` | `qwen3.7-flash` | 图片识别模型 |
| `DASH_VISION_VIDEO_MODEL` | `qwen3.7-flash` | 视频识别模型 |
| `DASH_VISION_AUDIO_MODEL` | `qwen3.5-omni-flash` | 短音频理解模型 |
| `DASH_VISION_ASR_MODEL` | `fun-asr` | 长音频转写模型 |
| `DASH_VISION_VIDEO_BASE64_MAX_MB` | `14` | 视频 base64 阈值，超出走 OSS 直传 |
| `DASH_VISION_AUDIO_OMNI_MAX_SEC` | `300` | 长音频转写阈值（秒） |
| `DASH_VISION_IMAGE_GEN_MODEL_STANDARD` | `qwen-image-3.0` | 文生图模型（PRO/MAX 档同理） |
| `DASH_VISION_VIDEO_GEN_MODEL_STANDARD_T2V` | `wan2.7-t2v` | 文生视频模型（MAX 档 happyhorse-1.1-t2v；I2V 同理） |
| `DASH_VISION_OUTPUT_DIR` | `~/Downloads/dsh-vision-dashscope` | 生成结果保存目录 |
| `DASH_VISION_MAX_VIDEO_DURATION` | `10` | 视频生成时长上限（秒） |
| `DASH_VISION_MAX_TOKENS` | `4096` | 输出上限 |

## 测试

```bash
uv run pytest -q
```

## License

MIT
