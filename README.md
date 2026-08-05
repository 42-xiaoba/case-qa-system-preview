# 智能案例问答系统

基于智谱 GLM-4.7-Flash（文本）与 GLM-4.6v-flash（视觉）双模型的案例智能问答 Web 应用。左侧 AI 对话区支持流式问答与图片识别，右侧原生渲染案例报告 PDF 并支持缩放，两侧独立滚动浏览。

## 简介

本系统将案例文本（`case.txt`）完整作为系统提示词的一部分传入大模型，结合三层提示词配置（系统人设、核心业务规则、输出格式），实现对案例的深度问答。前端采用左右分栏布局，左侧为暗色主题的 AI 对话区域，右侧为 PDF 案例报告查看器，各自独立滚动，互不干扰。

新增视觉能力后，用户可上传案例相关图片（图表、流程图、截图等），由 GLM-4.6v-flash 视觉模型识别图片内容并结合案例知识给出回答。视觉与文本模型共享同一套系统提示词（含案例全文），保证回答一致性。

### 技术栈

| 组件 | 技术 |
|------|------|
| 文本大模型 | 智谱 GLM-4.7-Flash |
| 视觉大模型 | 智谱 GLM-4.6v-flash（多模态，可选） |
| 后端 | FastAPI（本地开发，可选） |
| 前端 | Streamlit |
| 流式输出 | SSE 流式响应 + 后台线程动画渲染 |
| PDF 渲染 | pypdfium2（图片化缩放预览） |

### 运行模式

- **API 模式**：前端通过 FastAPI 后端调用模型，支持 PDF 预览（本地开发推荐）
- **直连模式**：前端直接调用智谱 API，无需后端（适合云端部署）

两种模式下视觉功能均可用，由前端 `vision_llm_client` 直接调用智谱视觉 API（不经过 FastAPI 中转，简化部署）。

## 项目结构

```
├── config.yaml           # 三层提示词配置 + 模型参数 + 视觉模型配置
├── case.txt              # 案例文本（作为系统提示词传入模型）
├── case_report.pdf       # 案例报告 PDF（右栏展示）
├── .env                  # 环境变量（GLM_API_KEY、GLM_V_API_KEY，不上传 Git）
├── .streamlit/
│   └── config.toml       # Streamlit 暗色主题配置
├── app.py                # FastAPI 后端（流式接口 + 视觉流式接口 + PDF 服务）
├── ui.py                 # Streamlit 前端主入口
├── core/
│   ├── __init__.py
│   ├── config.py         # 配置加载（读取 config.yaml + .env，含视觉密钥）
│   ├── llm_client.py     # 智谱 API 调用封装（文本 + 视觉，含流式接口）
│   └── prompt_manager.py # 三层提示词组装 + 视觉多模态消息构建
├── requirements.txt      # 依赖清单
└── README.md
```

## 使用方法

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置密钥

在 `.env` 文件中设置 API Key：

```
# 文本模型密钥（必需）
GLM_API_KEY=your_api_key_here

# 视觉模型密钥（可选，未配置则禁用图片识别功能）
GLM_V_API_KEY=your_vision_api_key_here
```

> 视觉密钥为可选配置。未配置 `GLM_V_API_KEY` 时，系统自动禁用图片上传功能，仅保留文本问答，不影响其他功能。

### 3. 启动服务

#### 本地开发（推荐，PDF 预览完整可用）

开两个终端：

```bash
# 终端1：启动 FastAPI 后端
python app.py

# 终端2：启动 Streamlit 前端
streamlit run ui.py
```

浏览器访问 `http://localhost:8501`。

#### 直连模式（仅前端，无需后端）

```bash
streamlit run ui.py
```

前端检测到后端未运行时自动切换为直连模式，直接调用智谱 API。

## 功能说明

### 文本问答

在底部输入框输入问题，模型基于案例文本给出回答。支持流式输出，等待期间显示动态省略号动画（`.` → `..` → `...` 循环），首个 token 到达后切换为流式回复。

### 图片识别问答

1. 点击左侧栏「🖼️ 添加图片」上传图片（支持 PNG/JPG/WebP/GIF，每次最多1张）
2. 在底部输入框输入关于图片的问题
3. 发送后自动路由到视觉模型（GLM-4.6v-flash）进行识别回答

视觉模型使用与文本模型相同的系统提示词（含案例全文），可结合图片内容和案例知识给出回答。上传新图片会自动替换旧图片；点击「移除图片」可清空当前图片。

### PDF 案例报告预览

右侧栏展示案例报告 PDF，支持以下操作：

| 操作 | 效果 |
|------|------|
| 鼠标滚轮 | 上下翻阅 PDF 内容 |
| 点击 ➕ / ➖ 按钮 | 放大或缩小 PDF（0.5x ~ 3.0x，步进 0.2x） |
| Shift + 鼠标滚轮 | 横向滚动放大后的 PDF |

### 对话管理

- **清空对话**：左侧栏「🗑️ 清空对话」按钮，清空当前所有对话记录并重置图片上传状态
- **连接状态**：左侧栏底部显示当前与后端服务的连接情况，✅ 表示正常，❌ 表示异常（会自动切换至直连模式）

### 初始问候语

首次进入或清空对话后，助手会显示问候语及功能详细介绍，涵盖文本问答、图片识别、PDF 预览、对话管理等各项功能的使用方法。

### 4. 修改提示词

编辑 `config.yaml` 即可调整三层提示词，无需改动代码，重启服务生效：

| 层级 | 名称 | 说明 |
|------|------|------|
| 第一层 | 系统人设与角色 | 定义 AI 助手身份定位 |
| 第二层 | 核心业务规则 | 问答规则与案例知识 |
| 第三层 | 输出格式 | 回答格式规范 |

视觉模型默认复用与文本模型相同的系统提示词。如需为视觉模型单独配置提示词，可在 `config.yaml` 的 `vision_model` 段设置。

### 5. 部署到 Streamlit Cloud

1. 推送代码到 GitHub（`.env` 已被 `.gitignore` 排除）
2. 登录 [Streamlit Cloud](https://streamlit.io/cloud)，点击 New app
3. 设置 Main file path 为 `ui.py`
4. 在 Secrets 页面添加：
   ```
   GLM_API_KEY = "your_api_key_here"
   GLM_V_API_KEY = "your_vision_api_key_here"
   ```
5. 点击 Deploy

## 扩展预留

- **RAG 检索增强**：`core/prompt_manager.py` 预留 `build_messages_with_rag()` 接口
- **视觉模型独立提示词**：`config.yaml` 的 `vision_model` 段支持独立配置系统提示词
- **主题定制**：`.streamlit/config.toml` 可调整全局配色
