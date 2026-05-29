# PaddleOCR GPU 一键部署

GPU 加速的 PaddleOCR Web 工具，支持图片/单张PDF/批量识别，导出 TXT + Markdown。

## 系统要求

- Windows 10/11
- Python 3.10（推荐用 conda 安装）
- NVIDIA GPU（显存 ≥ 6GB）
- CUDA 12.x

## 安装

1. 双击 `安装PaddleOCR.bat`
2. 等待自动下载并安装依赖（首次约 5-10 分钟）

## 启动

双击 `一键启动PaddleOCR.bat`，浏览器会自动打开 `http://localhost:7860`。

## 功能

- **图片 OCR**：上传图片，识别文字 + 检测框，保存为图片/TXT/Markdown
- **PDF OCR**：支持 500MB+ 大文件，每页 200 DPI 渲染，优先用原生文本，没有则用 OCR
- **批量 OCR**：一次上传多张图片/多个 PDF，批量处理
- **随时停止**：三个标签页都有红色 Stop 按钮，点击取消当前任务

## 输出目录

结果自动保存到 `output/` 文件夹，带时间戳。

## 缓存

PaddleX 模型缓存在 `.paddlex/`，首次运行自动下载（约 200MB），不占 C 盘。
