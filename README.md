<div align="center">
  <img src="images/logo.jpg" alt="llama_flow Logo" width="200" height="200">
</div>

# 🦙✨ llama_flow

<p align="center">
  <a href="https://github.com/sabeeralikp/llama_flow/stargazers">
    <img alt="GitHub stars" src="https://img.shields.io/github/stars/sabeeralikp/llama_flow?style=social">
  </a>
  <a href="https://github.com/sabeeralikp/llama_flow/network/members">
    <img alt="GitHub forks" src="https://img.shields.io/github/forks/sabeeralikp/llama_flow?style=social">
  </a>
  <a href="https://github.com/sabeeralikp/llama_flow/issues">
    <img alt="GitHub issues" src="https://img.shields.io/github/issues/sabeeralikp/llama_flow">
  </a>
  <a href="https://github.com/sabeeralikp/llama_flow/pulls">
    <img alt="GitHub pull requests" src="https://img.shields.io/github/issues-pr/sabeeralikp/llama_flow">
  </a>
  <a href="https://github.com/sabeeralikp/llama_flow/releases">
    <img alt="GitHub release downloads" src="https://img.shields.io/github/downloads/sabeeralikp/llama_flow/total">
  </a>
</p>

**llama_flow** is an open-source application designed to develop and customize Retrieval-Augmented Generation (RAG) workflows **without code**. Easily run it locally using a variety of open-source and closed-source large language models, vector databases, embedding models, and chunking strategies.

## 🚀 Features

### 🖥️ Frontend
- Built with **[Flutter](https://flutter.dev/)**
- Available for **[Linux](https://github.com/sabeeralikp/llama_flow/releases/download/Desktop-1.0.1/llama_flow-1.0.1+2-linux.deb)** and **[Windows](https://github.com/sabeeralikp/llama_flow/releases/download/Desktop-1.0.1/llama_flow-1.0.1+2-windows-setup.exe)**
- Source Code: [llama_flow_desktop](https://github.com/sabeeralikp/llama_flow_desktop)

### 🛠️ Backend
- Dockerized for local hosting
- Utilizes **[FastAPI](https://fastapi.tiangolo.com/)** and **[llama_index](https://github.com/run-llama/llama_index)**

### Supported Components

#### Basic RAG Workflow
| Workflow                      | Status                    |
|-------------------------------|---------------------------|
| Default with [Huggingface](https://huggingface.co/)      | ✅                        |
| Support for [llamacpp](https://github.com/ggerganov/llama.cpp) and [ollama](https://ollama.com/) | ✅                        |

#### Vector DB
| Vector DB                     | Status                    |
|-------------------------------|---------------------------|
| [chromadb](https://github.com/chroma-core/chroma)                      | ✅                        |
| [waviate](https://github.com/semi-technologies/weaviate)                      | ⏳                        |
| [faiss](https://github.com/facebookresearch/faiss)                         | ⏳                        |
| [qdrant](https://github.com/qdrant/qdrant)                        | ⏳                        |

#### Embed Model Provider
| Embed Model Provider          | Status                    |
|-------------------------------|---------------------------|
| [Huggingface](https://huggingface.co/)                   | ✅                        |
| [Ollama](https://ollama.com/)                        | ⏳                        |
| [OpenAI](https://www.openai.com/)                        | ⏳                        |
| [Cohere](https://cohere.ai/)                        | ⏳                        |

#### Embed Models (Huggingface)
| Embed Model (Huggingface)      | Status                    |
|-------------------------------|---------------------------|
| [Snowflake/snowflake-arctic-embed-l](https://huggingface.co/Snowflake/snowflake-arctic-embed-l)  | ✅                 |
| [Alibaba-NLP/gte-large-en-v1.5](https://huggingface.co/Alibaba-NLP/gte-large-en-v1.5)        | ✅                 |
| [Snowflake/snowflake-arctic-embed-m](https://huggingface.co/Snowflake/snowflake-arctic-embed-m)  | ✅                 |
| [Snowflake/snowflake-arctic-embed-m-long](https://huggingface.co/Snowflake/snowflake-arctic-embed-m-long) | ✅             |
| [WhereIsAI/UAE-Large-V1](https://huggingface.co/WhereIsAI/UAE-Large-V1)               | ✅                 |
| [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)               | ✅                 |
| [mixedbread-ai/mxbai-embed-large-v1](https://huggingface.co/mixedbread-ai/mxbai-embed-large-v1)  | ✅                 |
| [BAAI/bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5)               | ✅                 |

#### LLM Providers
| LLM Provider                  | Status                    |
|-------------------------------|---------------------------|
| [Huggingface](https://huggingface.co/models)                   | ✅                        |
| [llamacpp](https://github.com/ggerganov/llama.cpp)                      | ✅                        |
| [ollama](https://ollama.com/)                        | ✅                        |
| Huggingface API               | ⏳                        |
| [OpenAI](https://www.openai.com/)                        | ⏳                        |
| [Cohere](https://cohere.ai/)                        | ⏳                        |

#### Huggingface LLMs
| Huggingface LLM               | Status                    |
|-------------------------------|---------------------------|
| [microsoft/Phi-3-mini-128k-instruct](https://huggingface.co/microsoft/Phi-3-mini-128k-instruct) | ✅                    |
| [upstage/SOLAR-10.7B-Instruct-v1.0](https://huggingface.co/upstage/SOLAR-10.7B-Instruct-v1.0) | ✅                    |
| [Intel/neural-chat-7b-v3-3](https://huggingface.co/Intel/neural-chat-7b-v3-3)        | ✅                     |
| [Nexusflow/Starling-LM-7B-beta](https://huggingface.co/Nexusflow/Starling-LM-7B-beta)    | ✅                     |
| [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) | ✅                  |
| [mistralai/Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) | ✅                   |
| [meta-llama/CodeLlama-7b-hf](https://huggingface.co/meta-llama/CodeLlama-7b-hf)       | ✅                     |
| [google/gemma-1.1-7b-it](https://huggingface.co/google/gemma-1.1-7b-it)           | ✅                     |
| [google/gemma-1.1-2b-it](https://huggingface.co/google/gemma-1.1-2b-it)           | ✅                     |

#### llamacpp LLMs
| llamacpp LLM                  | Status                    |
|-------------------------------|---------------------------|
| llama2-7b                     | ✅                        |
| llama2-13b                    | ✅                        |
| llama3-8b                     | ✅                        |

#### ollama LLMs
| ollama LLM                    | Status                    |
|-------------------------------|---------------------------|
| llama3                        | ✅                        |
| phi3                          | ✅                        |
| mistral                       | ✅                        |
| neural-chat                   | ✅                        |
| starling-lm                   | ✅                        |
| codellama                     | ✅                        |
| gemma:2b                      | ✅                        |
| gemma:7b                      | ✅                        |
| solar                         | ✅                        |

#### Chunking Strategy
| Chunking Strategy             | Status                    |
|-------------------------------|---------------------------|
| semantic-splitting            | ✅                        |
| simple-node-parser            | ⏳                        |
| sentence-splitting            | ⏳                        |
| sentence-window               | ⏳                        |
| token-splitting               | ⏳                        |
| hierarchical-splitting        | ⏳                        |

### Planned Features
- Advanced RAG workflow (See illustration below)
- Custom workflow with drag-and-drop functionality

<div align="center">
  <img src="https://miro.medium.com/v2/resize:fit:2000/format:webp/0*Gr_JqzdpHu7enWG9.png" alt="Advanced RAG Workflow" width="500">
</div>

## 🛠️ Installation

### 1. Frontend UI Application
Download the desktop application for your OS:
- [Linux](https://github.com/sabeeralikp/llama_flow/releases/download/linux-desktop-1.0.1/llama_flow-1.0.1+2-linux.deb)
- [Windows](https://github.com/sabeeralikp/llama_flow/releases/download/linux-desktop-1.0.1/llama_flow-1.0.1+2-windows-setup.exe)

### 2. Backend Configuration

#### Option 1: Dockerized Backend
1. **Clone the repo**
   ```sh
   git clone https://github.com/sabeeralikp/llama_flow.git
   cd llama_flow
   ```
2. **Build and Run Docker Container**
   ```sh
   docker build -t llama_flow_image . 
   docker run -d --name llama_flow_container -p 8000:8000 llama_flow_image
   ```

#### Option 2: Manual Setup with FastAPI
1. **Clone the repo**
   ```sh
   git clone https://github.com/sabeeralikp/llama_flow.git
   cd llama_flow
   ```
2. **Create a virtual environment**
   ```sh
   python -m venv env
   source env/bin/activate   # On Windows use `env\Scripts\activate`
   ```
3. **Install necessary packages**
   ```sh
   pip install -r requirements.txt
   ```
4. **Run the backend with Uvicorn**
   ```sh
   uvicorn main:app --workers 4
   ```

#### Option 3: Colab/Kaggle Notebooks
- **llama_flow with Huggingface**: [Colab Notebook](https://github.com/sabeeralikp/llama_flow/blob/main/notebooks/llama_flow_huggingface_backend_colab.ipynb)
- **llama_flow with llamacpp**: [Colab Notebook](https://github.com/sabeeralikp/llama_flow/blob/main/notebooks/llama_flow_llamacpp_backend_colab.ipynb)
- **llama_flow with Ollama**: [Colab Notebook](https://github.com/sabeeralikp/llama_flow/blob/main/notebooks/llama_flow_ollama_backend_colab.ipynb)

**Instructions:**
1. Create a [Ngrok](https://ngrok.com/) account.
2. Go to the Ngrok dashboard, create an auth tunnel, copy the auth token, and paste it into Colab.
3. Open the notebook file in Colab and run all cells.
4. Copy the external link from the cell output.
5. Open the desktop application, go to settings, change the backend to remote, and paste the copied external domain link in the base URL text field.

### Additional Backend Configurations

#### Using llamacpp
```sh
pip install -q llama-cpp-python \
  --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122  # For CUDA version 12.2
pip install -q llama_index llama-index-llms-llama-cpp
```

#### Using Ollama
Follow the [official Ollama installation guide](https://github.com/ollama/ollama/?tab=readme-ov-file#ollama) for different devices.
```sh
apt install lshw
curl -fsSL https://ollama.com/install.sh | sh
pip install -q llama-index-llms-ollama
```

## 🤝 Contribution Guidelines
We welcome all contributions to improve llama_flow. Please follow these steps:
1. **Fork the repository**
2. **Create a new branch**
   ```sh
   git checkout -b feature-branch
   ```
3. **Commit your changes**
   ```sh
   git commit -m 'Add some feature'
   ```
4. **Push to the branch**
   ```sh
   git push origin feature-branch
   ```
5. **Create a new Pull Request**

## 📜 License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/sabeeralikp/llama_flow/blob/main/LICENSE) file for details.

---

> **Stay tuned for exciting updates!** 🚀✨
