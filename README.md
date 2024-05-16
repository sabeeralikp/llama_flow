
# <img src="images/logo.jpg" alt="llama_flow Logo" width="64" height="64"> llama_flow

llama_flow is a No-Code custom RAG (Retrieval-Augmented Generation) chatbot system designed for seamless integration and easy setup. The project is available on GitHub in two versions: [llama_flow](https://github.com/sabeeralikp/llama_flow) for the backend written in FastAPI, and [llama_flow_desktop](https://github.com/sabeeralikp/llama_flow_desktop) for the desktop application built with Flutter. This system is in its early development stage.

## Setup Instructions

### Desktop Version
1. Download the desktop version from the releases section on GitHub.
   - Note: Currently, the code only supports [Linux](https://github.com/sabeeralikp/llama_flow/releases/tag/linux-desktop) and Windows. The MacOS build is yet to be done.

### Backend Setup

#### Option 1: Setup in Google Colab
1. Create a Ngrok account.
2. Head over to the Ngrok dashboard, create an auth tunnel, copy the auth token, and paste it into Colab.
3. Open the [`llama_flow_backend_colab.ipynb`](https://github.com/sabeeralikp/llama_flow/blob/main/notebooks/llama_flow_backend_colab.ipynb) file in Colab and run all cells.
4. Copy the external link from the cell output.
5. Open the desktop application, go to settings, change the backend to remote, and paste the copied external domain link in the base URL text field.

#### Option 2: Self-Host or Local Setup
1. Clone the repo
2. Create a virtual environment using pip, Miniconda, Conda, or virtualenv.
3. Install the necessary packages using the [`requirements.txt`](https://github.com/sabeeralikp/llama_flow/blob/main/requirements.txt) file.
   ```sh
   pip install -r requirements.txt
   ```
4. Run the backend with Uvicorn using workers.
   ```sh
   uvicorn main:app --workers 4
   ```

### Create a RAG Conversational Agent
Follow the instructions provided in the desktop application to create and configure your RAG conversational agent.

## Contribution Guidelines
We welcome contributions to improve llama_flow. Please follow these guidelines:
- Fork the repository.
- Create a new branch (`git checkout -b feature-branch`).
- Commit your changes (`git commit -m 'Add some feature'`).
- Push to the branch (`git push origin feature-branch`).
- Create a new Pull Request.

## License
This project is licensed under the MIT License. See the [LICENSE](https://github.com/sabeeralikp/llama_flow/blob/main/LICENSE) file for details.
