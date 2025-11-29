# NSL Itasha Generator

一个基于 Python 和 Google Gemini 模型的 Switch Lite 痛贴生成器。

## 项目简介

该项目旨在生成 Nintendo Switch Lite 的“痛贴”设计模版。工作流如下：
1.  **视觉理解**：使用 `gemini-2.5-flash` 模型分析用户上传的图片，识别元素、风格和配色。
2.  **提示词生成**：使用 `gemini-3-pro-preview` 模型根据视觉分析结果，生成适用于生图模型的高质量提示词。
3.  **图像生成**：使用 `gemini-3-pro-image-preview` 模型生成最终的痛贴纹理。
4.  **模版合成**：将生成的纹理应用到 Switch Lite 模版上（目前为裁剪示意）。

## 环境设置

本项目使用 [uv](https://github.com/astral-sh/uv) 进行依赖管理。

1.  **安装依赖**：
    ```bash
    uv sync
    ```

2.  **配置环境变量**：
    在项目根目录创建一个 `.env` 文件，并填入你的 Google API Key：
    ```env
    GOOGLE_API_KEY=your_api_key_here
    ```

## 使用方法

1.  将参考图片放入 `assets/input/` 目录（例如 `assets/input/character.jpg`）。
2.  运行生成脚本：
    ```bash
    uv run main.py assets/input/character.jpg assets/input/style_ref.png
    ```
    或者：
```bash
    source .venv/bin/activate
    python main.py assets/input/character.jpg
```

3.  生成结果将保存在 `assets/output/` 目录下。

## 项目结构

- `src/services/vision.py`: 视觉分析服务 (Gemini 2.5 Flash)
- `src/services/generation.py`: 提示词与图像生成服务 (Gemini 3 Pro)
- `src/utils/image_ops.py`: 图像处理工具
- `main.py`: 入口文件
