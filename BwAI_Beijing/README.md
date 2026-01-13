# Google GenAI Notebooks Collection

这是一个关于 Google GenAI SDK (`google-genai`) 使用范例的笔记本集合，涵盖了从文本翻译到最新的图像生成模型（Imagen 4, Gemini 2.5, Gemini 3）的应用。

---

## 📂 档案介绍

本项目包含以下 Jupyter Notebooks，分别演示了不同的功能：

| 档案名称 | 说明 |
|----------|------|
| `gen_translate_cn.ipynb` | **文本翻译**<br>使用 Gemini API 进行多语言翻译的范例，展示如何处理文件上传并进行翻译。 |
| `imagen4_image_generation_ipynb_cn.ipynb` | **Imagen 4 图像生成**<br>演示如何使用 Google 最新的 Imagen 4 模型进行高质量图像生成，包含 Imagen 4 Fast 和 Ultra 的比较。 |
| `intro_gemini_2_5_image_gen_ipynb_cn.ipynb` | **Gemini 2.5 Flash Image**<br>介绍 Gemini 2.5 Flash 模型在图像生成与编辑上的应用（Nano Banana）。 |
| `intro_gemini_3_image_gen_ipynb_cn.ipynb` | **Gemini 3 Pro Image**<br>展示更强大的 Gemini 3 Pro 模型在图像生成、多轮对话编辑与思考过程可视化的能力（Nano Banana Pro）。 |

---

## 🛠️ google.genai 用法

本项目使用最新的 `google-genai` SDK。根据您的环境与需求，主要有两种初始化客户端 (`Client`) 的方式。

### 1. Vertex AI (企业级/Google Cloud)

适用于在 Google Cloud Platform (GCP) 上运行，需要配置 Project ID 和 Location。这是企业级开发最常用的方式，支持完整的 Vertex AI 功能。

**完整范例：**

```python
from google import genai
import os

# 设置您的 Google Cloud Project ID
# 建议通过环境变量获取，或者直接填入字符串
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "your-project-id")
LOCATION = "us-central1"  # 或其他支持的区域

# 初始化 Vertex AI Client
client = genai.Client(
    vertexai=True,
    project=PROJECT_ID,
    location=LOCATION
)

# 测试调用 (例如生成文本)
response = client.models.generate_content(
    model="gemini-2.0-flash-001",
    contents="你好，Vertex AI!"
)

print(response.text)
```

### 2. Gemini API Key (开发者/快速原型)

适用于快速开发或个人项目，直接使用 API Key 认证。这通常用于 Google AI Studio 的 Gemini API，或者 Vertex AI 的 Express Mode。

**完整范例：**

```python
from google import genai
import os

# 设置您的 API Key
# 可以在 Google AI Studio (aistudio.google.com) 获取
API_KEY = os.environ.get("GOOGLE_API_KEY", "your-api-key-here")

# 初始化 Client (使用 API Key)
client = genai.Client(
    api_key=API_KEY
)

# 测试调用
response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents="你好，Gemini API!"
)

print(response.text)
```

---

## 🚀 快速开始

1. **安装 SDK**:
   ```bash
   pip install -U google-genai
   ```

2. **配置环境**:
   - 如果使用 **Vertex AI**，请确保已安装 `gcloud` CLI 并完成认证 (`gcloud auth login`, `gcloud auth application-default login`)，且项目已启用 Vertex AI API。
   - 如果使用 **API Key**，请确保您拥有有效的 Key。

3. **运行笔记本**:
   使用 Jupyter Lab 或 VS Code 打开上述 `.ipynb` 文件即可开始实验。

---

## 📝 License

[MIT](LICENSE)