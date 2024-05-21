import asyncio
import chromadb
from basic_rag_workflow.llamacpp_model_url import llamacpp_model_url
import torch
from typing import List
from multiprocessing import cpu_count
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext,
    VectorStoreIndex,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
)
from llama_index.llms.huggingface import HuggingFaceLLM
from fastapi import HTTPException, status, Response, UploadFile

from schema import BaseRAGModel

try:
    from llama_index.llms.llama_cpp import LlamaCPP
    from llama_index.llms.llama_cpp.llama_utils import (
        messages_to_prompt,
        completion_to_prompt,
    )
    from llama_index.core import set_global_tokenizer
    from transformers import AutoTokenizer
    from urllib.request import urlretrieve
    import os
except ImportError:
    print("llama_cpp not installed, using HuggingFace LLM instead")

try:
    from llama_index.llms.ollama import Ollama
except ImportError:
    print("ollama not installed, using HuggingFace LLM instead")


class BasicRagWorkflow:
    """
    BasicRagWorkflow class handles the setup and operation of a basic 
    Retrieval-Augmented Generation (RAG) workflow using various LLMs 
    (Large Language Models) and vector databases.
    """
    
    def __init__(self):
        """
        Initialize the BasicRagWorkflow with default settings.
        Sets up vector database, embedding models, LLMs, and other necessary components.
        """
        self.vector_db = chromadb.PersistentClient("chromadb")

        self.embed_model = HuggingFaceEmbedding(
            model_name="Snowflake/snowflake-arctic-embed-l",
            trust_remote_code=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        self.system_prompt = "You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided. the answer should be only based on the given context, if no acceptable context respond as not found on given context"

        self.llm = HuggingFaceLLM(
            context_window=4096,
            max_new_tokens=1048,
            generate_kwargs={"temperature": 0, "do_sample": False},
            system_prompt=self.system_prompt,
            # query_wrapper_prompt=qa_prompt_tmpl,
            tokenizer_name="microsoft/Phi-3-mini-128k-instruct",
            model_name="microsoft/Phi-3-mini-128k-instruct",
            device_map="auto" if torch.cuda.is_available() else "cpu",
            tokenizer_kwargs={
                "max_length": 4096,
                "trust_remote_code": True,
            },
            # uncomment this if using CUDA to reduce memory usage
            model_kwargs=(
                {
                    "torch_dtype": torch.float16,
                    "llm_int8_enable_fp32_cpu_offload": True,
                    "bnb_4bit_quant_type": "nf4",
                    "bnb_4bit_use_double_quant": True,
                    "bnb_4bit_compute_dtype": torch.bfloat16,
                    "load_in_4bit": True,
                    "trust_remote_code": True,
                }
                if torch.cuda.is_available()
                else {
                    "trust_remote_code": True,
                }
            ),
        )

        self.splitter = SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=self.embed_model,
        )

        self.service_context = ServiceContext.from_defaults(
            embed_model=self.embed_model, node_parser=self.splitter, llm=self.llm
        )

        self.retriver_top_k = 5

        if "default" in [
            collection_name.name
            for collection_name in self.vector_db.list_collections()
        ]:
            self.chroma_collection = self.vector_db.get_or_create_collection("default")

            self.vector_store = ChromaVectorStore(self.chroma_collection)
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            self.vector_store_index = VectorStoreIndex.from_vector_store(
                vector_store=self.vector_store,
                service_context=self.service_context,
                storage_context=self.storage_context,
            )
            self.query_engine = self.vector_store_index.as_query_engine(
                similarity_top_k=self.retriver_top_k
            )
        else:
            self.chroma_collection = self.vector_db.get_or_create_collection("default")

            self.vector_store = ChromaVectorStore(self.chroma_collection)
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )

    def get_db_collections(self, vector_db_name: str):
        """
        Retrieve the list of collections in the specified vector database.

        Args:
            vector_db_name (str): Name of the vector database.

        Returns:
            List of collections or HTTPException if the database is not found.
        """
        if vector_db_name == "chromadb":
            return self.vector_db.list_collections()
        else:
            return HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="DB Not Found",
            )

    def get_basic_settings(self):
        """
        Get the basic settings for the RAG workflow.

        Returns:
            dict: Basic settings including vector DB, embed model, LLM provider, and more.
        """
        return {
            "vector_db": ["chromadb", "waviate", "faiss", "qdrant"],
            "vector_db_collection": "default",
            "embed_model_provider": ["huggingface", "openai", "cohere"],
            "embed_model": [
                "Snowflake/snowflake-arctic-embed-l",
                "Alibaba-NLP/gte-large-en-v1.5",
                "Snowflake/snowflake-arctic-embed-m",
                "Snowflake/snowflake-arctic-embed-m-long",
                "WhereIsAI/UAE-Large-V1",
                "BAAI/bge-small-en-v1.5",
                "mixedbread-ai/mxbai-embed-large-v1",
                "BAAI/bge-large-en-v1.5",
            ],
            "llm_provider": ["huggingface", "llamacpp", "ollama"],
            "huggingface_llm": [
                "microsoft/Phi-3-mini-128k-instruct",
                "upstage/SOLAR-10.7B-Instruct-v1.0",
                "Intel/neural-chat-7b-v3-3",
                "Nexusflow/Starling-LM-7B-beta",
                "meta-llama/Meta-Llama-3-8B-Instruct",
                "mistralai/Mistral-7B-Instruct-v0.2",
                "meta-llama/CodeLlama-7b-hf",
                "google/gemma-1.1-7b-it",
                "google/gemma-1.1-2b-it",
            ],
            "llama_cpp": [
                "llama2-7b",
                "llama2-13b",
                "llama3-8b",
            ],
            "ollama": [
                "llama3",
                "phi3",
                "mistral",
                "neural-chat",
                "starling-lm",
                "codellama",
                "gemma:2b",
                "gemma:7b",
                "solar",
            ],
            "load_in_4bit": True,
            "load_in_8bit": False,
            "chunking_strategy": [
                "semantic-splitting",
                "simple-node-parser",
                "sentence-splitting",
                "sentence-window",
                "token-splitting",
                "heirarchical-splitting",
            ],
            "semantic-splitting": {
                "buffer_size": 1,
                "breakpoint_percentile_threshold": 95,
            },
            "retriver": {
                "top-k": 5,
            },
        }

    def update_basic_settings(self, basic_settings: BaseRAGModel):
        """
        Update the basic settings for the RAG workflow.

        Args:
            basic_settings (BaseRAGModel): The updated settings.

        Returns:
            HTTPException if any invalid settings are provided.
        """

        setting_changed = False

        # TODO: Plug and Play different vectorDBs
        if basic_settings.vector_db != "chromadb":
            return HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid vector_db",
            )
        if basic_settings.vector_db_collection != "default":
            setting_changed = True
            self.chroma_collection = self.vector_db.get_or_create_collection(
                basic_settings.vector_db_collection
            )

        # TODO: Plug and Play different embed model Provider and Embedding Models
        if basic_settings.embed_model_provider != "huggingface":
            return HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid vector_db",
            )
        else:
            if basic_settings.embed_model != "Snowflake/snowflake-arctic-embed-l":
                setting_changed = True
                self.embed_model = HuggingFaceEmbedding(
                    model_name=basic_settings.embed_model,
                    trust_remote_code=True,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                )

        # TODO: Plug and Play different LLM Provider and its Settings and Embedding Models
        if basic_settings.llm_provider != "huggingface":
            setting_changed = True
            if basic_settings.llm_provider == "llamacpp":
                self.llm = LlamaCPP(
                    # You can pass in the URL to a GGML model to download it automatically
                    model_url=llamacpp_model_url(basic_settings.llm),
                    # optionally, you can set the path to a pre-downloaded model instead of model_url
                    # model_path=f"llama_models/{model_url.split('/')[-1]}",
                    temperature=0.0,
                    max_new_tokens=1048,
                    # llama2 has a context window of 4096 tokens, but we set it lower to allow for some wiggle room
                    context_window=4096,
                    # kwargs to pass to __call__()
                    generate_kwargs={},
                    # kwargs to pass to __init__()
                    # set to at least 1 to use GPU
                    model_kwargs=(
                        {"n_gpu_layers": -1} if torch.cuda.is_available() else {}
                    ),
                    # transform inputs into Llama2 format
                    messages_to_prompt=messages_to_prompt,
                    completion_to_prompt=completion_to_prompt,
                    verbose=False,
                )
                set_global_tokenizer(
                    AutoTokenizer.from_pretrained(
                        "NousResearch/Llama-2-7b-chat-hf"
                    ).encode
                )
            if basic_settings.llm_provider == "ollama":
                self.llm = Ollama(model=basic_settings.llm, request_timeout=300.0)
        else:
            if basic_settings.llm != "microsoft/Phi-3-mini-128k-instruct":
                setting_changed = True
                self.llm = HuggingFaceLLM(
                    context_window=4096,
                    max_new_tokens=1048,
                    generate_kwargs={"temperature": 0, "do_sample": False},
                    system_prompt=self.system_prompt,
                    # query_wrapper_prompt=qa_prompt_tmpl,
                    tokenizer_name=basic_settings.llm,
                    model_name=basic_settings.llm,
                    device_map="auto" if torch.cuda.is_available() else "cpu",
                    tokenizer_kwargs={
                        "max_length": 4096,
                        "trust_remote_code": True,
                    },
                    # uncomment this if using CUDA to reduce memory usage
                    model_kwargs=(
                        {
                            "torch_dtype": torch.float16,
                            "llm_int8_enable_fp32_cpu_offload": True,
                            "bnb_4bit_quant_type": "nf4",
                            "bnb_4bit_use_double_quant": True,
                            "bnb_4bit_compute_dtype": torch.bfloat16,
                            "load_in_4bit": True,
                            "trust_remote_code": True,
                        }
                        if torch.cuda.is_available() and basic_settings.load_in_4bit
                        else {
                            "trust_remote_code": True,
                        }
                    ),
                )
            # if basic_settings.load_in_4bit != True:

        # TODO: Add more Chunking Strategy
        if basic_settings.chunking_strategy != "semantic-splitting":
            return HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid chunking_strategy",
            )

        if (
            basic_settings.semantic_splitting_buffer_size != 1
            or basic_settings.semantic_splitting_breakpoint_percentile_threshold != 95
        ):
            setting_changed = True
            self.splitter = SemanticSplitterNodeParser(
                buffer_size=basic_settings.semantic_splitting_buffer_size,
                breakpoint_percentile_threshold=basic_settings.semantic_splitting_breakpoint_percentile_threshold,
                embed_model=self.embed_model,
            )

        if basic_settings.retriver_top_k != 5:
            if self.query_engine is None:
                return HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Invalid retriver_top_k",
                )
            self.retriver_top_k = basic_settings.retriver_top_k

        if setting_changed:
            self.vector_store = ChromaVectorStore(self.chroma_collection)
            self.storage_context = StorageContext.from_defaults(
                vector_store=self.vector_store
            )
            self.service_context = ServiceContext.from_defaults(
                embed_model=self.embed_model, node_parser=self.splitter, llm=self.llm
            )

    def document_indexing(self, file_paths: List[str], num_workers: int = cpu_count()):
        """
        Index documents from the specified file paths.

        Args:
            file_paths (List[str]): List of file paths to be indexed.
            num_workers (int): Number of worker processes to use for indexing.

        Returns:
            Response: A success response upon completion of indexing.
        """
        loader = SimpleDirectoryReader(input_files=file_paths)
        docs = loader.load_data(num_workers=num_workers)

        self.vector_store_index = VectorStoreIndex.from_documents(
            docs,
            storage_context=self.storage_context,
            service_context=self.service_context,
        )

        self.query_engine = self.vector_store_index.as_query_engine(
            streaming=False, similarity_top_k=self.retriver_top_k
        )

        return Response(
            content="Indexing Successfully Completed",
            status_code=status.HTTP_201_CREATED,
        )

    def document_querying(self, query_str: str):
        """
        Query the indexed documents.

        Args:
            query_str (str): The query string.

        Returns:
            The query results.
        """
        return self.query_engine.query(query_str)
