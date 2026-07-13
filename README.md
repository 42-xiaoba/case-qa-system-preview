# 智能案例问答系统

基于智谱 GLM-4.7-Flash 模型的案例智能问答 Web 应用。左侧 AI 对话区支持流式问答，右侧原生渲染案例报告 PDF，两侧独立滚动浏览。

## 简介

本系统将案例文本（`case.txt`）完整作为系统提示词的一部分传入大模型，结合三层提示词配置（系统人设、核心业务规则、输出格式），实现对案例的深度问答。前端采用左右分栏布局，左侧为暗色主题的 AI 对话区域，右侧为 PDF 案例报告查看器，各自独立滚动，互不干扰。

### 技术栈

| 组件 | 技术 |
|------|------|
| 大模型 | 智谱 GLM-4.7-Flash |
| 后端 | FastAPI（本地开发，可选） |
| 前端 | Streamlit |
| 流式输出 | SSE 流式响应 + `st.write_stream` 实时渲染 |

### 运行模式

- **API 模式**：前端通过 FastAPI 后端调用模型，支持 PDF 预览（本地开发推荐）
- **直连模式**：前端直接调用智谱 API，无需后端（适合云端部署）

## 项目结构

```
├── config.yaml           # 三层提示词配置 + 模型参数
├── case.txt              # 案例文本（作为系统提示词传入模型）
├── case_report.pdf       # 案例报告 PDF（右栏展示）
├── .env                  # 环境变量（GLM_API_KEY，不上传 Git）
├── .streamlit/
│   └── config.toml       # Streamlit 暗色主题配置
├── app.py                # FastAPI 后端（流式接口 + PDF 服务）
├── ui.py                 # Streamlit 前端主入口
├── core/
│   ├── __init__.py
│   ├── config.py         # 配置加载（读取 config.yaml + .env）
│   ├── llm_client.py     # 智谱 API 调用封装（含流式接口）
│   └── prompt_manager.py # 三层提示词组装
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
GLM_API_KEY=your_api_key_here
```

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

### 4. 修改提示词

编辑 `config.yaml` 即可调整三层提示词，无需改动代码，重启服务生效：

| 层级 | 名称 | 说明 |
|------|------|------|
| 第一层 | 系统人设与角色 | 定义 AI 助手身份定位 |
| 第二层 | 核心业务规则 | 问答规则与案例知识 |
| 第三层 | 输出格式 | 回答格式规范 |

### 5. 部署到 Streamlit Cloud

1. 推送代码到 GitHub（`.env` 已被 `.gitignore` 排除）
2. 登录 [Streamlit Cloud](https://streamlit.io/cloud)，点击 New app
3. 设置 Main file path 为 `ui.py`
4. 在 Secrets 页面添加：
   ```
   GLM_API_KEY = "your_api_key_here"
   ```
5. 点击 Deploy

## 扩展预留

- **RAG 检索增强**：`core/prompt_manager.py` 预留 `build_messages_with_rag()` 接口
- **侧边栏功能扩展**：`ui.py` 侧边栏预留功能入口
- **主题定制**：`.streamlit/config.toml` 可调整全局配色
