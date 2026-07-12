# 智能案例问答系统

基于智谱 GLM-4.7-Flash 模型的案例智能问答 Web 应用，支持对 `case.txt` 中的完整案例内容进行深度问答分析。

## 项目结构

```
├── config.yaml          # 配置文件（三层提示词 + 模型参数）
├── case.txt             # 案例文本
├── case_report.pdf      # 案例报告 PDF（可选，用于右栏预览）
├── .env                 # 环境变量（GLM_API_KEY，仅本地开发，不上传 Git）
├── .gitignore           # Git 忽略规则
├── app.py               # FastAPI 后端服务（本地开发用）
├── ui.py                # Streamlit 前端界面（主入口）
├── core/
│   ├── __init__.py      # 包初始化
│   ├── config.py        # 配置加载模块（支持 st.secrets）
│   ├── llm_client.py    # API 调用封装
│   └── prompt_manager.py# 提示词管理模块
├── requirements.txt     # 依赖清单
└── README.md            # 本文件
```

## 启动步骤

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置密钥

**本地开发**：在 `.env` 文件中设置：

```
GLM_API_KEY=your_api_key_here
```

**Streamlit Cloud 部署**：在 Secrets 管理页面添加 `GLM_API_KEY`，无需创建 `.env` 文件。

### 3. 启动服务

#### 方式一：本地开发（推荐，支持 PDF 预览）

开两个终端：

```bash
# 终端1：启动 FastAPI 后端
python app.py

# 终端2：启动 Streamlit 前端
streamlit run ui.py
```

前端页面：`http://localhost:8501`

#### 方式二：仅启动前端（直连模式，适合测试）

```bash
streamlit run ui.py
```

此时前端会直接调用 GLM API，无需 FastAPI 后端，但 PDF 预览不可用（提供下载按钮代替）。

## 配置说明

`config.yaml` 中的提示词分为三层：

| 层级 | 名称 | 说明 |
|------|------|------|
| 第一层 | 系统人设与角色 | 定义 AI 助手的身份定位 |
| 第二层 | 核心业务规则/案例知识 | 问答规则和案例知识摘要 |
| 第三层 | 输出格式要求 | 回答的格式规范 |

修改 `config.yaml` 后无需改动代码，重启服务即可生效。

## 部署到 Streamlit Cloud

1. **推送代码到 GitHub**（确保 `.env` 已被 `.gitignore` 排除）
2. **登录 [Streamlit Cloud](https://streamlit.io/cloud)**，点击 "New app"
3. 选择你的仓库，设置：
   - **Main file path**: `ui.py`
   - **Python version**: 3.11+
4. 在 **Secrets** 页面添加：
   ```
   GLM_API_KEY = "your_api_key_here"
   ```
5. 点击 **Deploy**
6. 部署完成后，应用会在 `https://{your-app}.streamlit.app` 运行

> **注意**：Streamlit Cloud 上 UI 以直连模式运行，PDF 预览会显示下载按钮。

## 扩展预留

- **RAG 检索增强**：`core/prompt_manager.py` 中预留了 `build_messages_with_rag()` 接口
- **CSS 主题美化**：`ui.py` 中预留了 `CUSTOM_CSS` 变量和 `load_custom_css()` 函数
- **侧边栏扩展**：已预留文件上传、高级检索等功能入口
- **流式输出**：`core/llm_client.py` 中预留了 `chat_stream()` 接口