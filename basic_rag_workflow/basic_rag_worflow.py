import asyncio
import chromadb
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
from fastapi import HTTPException, status, Response

from ..models import BaseRAGModel


class BasicRagWorkflow:

    def __init__(self):
        self.vector_db = chromadb.PersistentClient("chromadb")

        self.system_prompt = "You are a Q&A assistant. Your goal is to answer questions as accurately as possible based on the instructions and context provided."

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
        if vector_db_name == "chromadb":
            return self.vector_db.list_collections()
        else:
            return HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="DB Not Found",
            )

    def get_basic_settings(self):
        return Response(
            {
                "vector_db": ["chromadb"],
                "vector_db_collection": "default",
                "embed_model_provider": ["huggingface"],
                "embed_model": [
                    "Snowflake/snowflake-arctic-embed-l",
                    "BAAI/bge-small-en-v1.5",
                    "bge-large-en-v1.5",
                ],
                "llm_provider": ["huggingface"],
                "llm": ["microsoft/Phi-3-mini-128k-instruct"],
                "load_in_4bit": True,
                "chunking_strategy": ["semantic-splitting"],
                "semantic-splitting": {
                    "buffer_size": 4,
                    "breakpoint_percentile_threshold": 98,
                },
                "retriver": {
                    "top-k": 5,
                },
            },
            status_code=status.HTTP_200_OK,
        )

    def update_basic_settings(self, basic_settings: BaseRAGModel):

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
        if basic_settings.embed_model != "Snowflake/snowflake-arctic-embed-l":
            return HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid embed_model",
            )

        # TODO: Plug and Play different LLM Provider and its Settings and Embedding Models
        if basic_settings.llm != "microsoft/Phi-3-mini-128k-instruct":
            return HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid llm"
            )
        if basic_settings.llm_provider != "huggingface":
            return HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid embed_model_provider",
            )
        if basic_settings.load_in_4bit != True:
            return HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid load_in_4bit",
            )

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
        return self.query_engine.query(query_str)
