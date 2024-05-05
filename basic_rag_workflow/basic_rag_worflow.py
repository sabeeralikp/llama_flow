import chromadb
import torch
from typing import List
from multiprocessing import cpu_count
from llama_index.core import (
    SimpleDirectoryReader,
    StorageContext,
    ServiceContext,
    VectorStoreIndex,
    SummaryIndex,
)
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.node_parser import (
    SemanticSplitterNodeParser,
)
from llama_index.llms.huggingface import HuggingFaceLLM
from fastapi import HTTPException, status, Response


class BasicRagWorkflow:

    def __init__(self):
        self.vector_db = chromadb.PersistentClient("chromadb")

        self.embed_model = HuggingFaceEmbedding(
            model_name="Snowflake/snowflake-arctic-embed-l",
            trust_remote_code=True,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        self.llm = HuggingFaceLLM(
            context_window=4096,
            max_new_tokens=256,
            generate_kwargs={"temperature": 0, "do_sample": False},
            # system_prompt=system_prompt,
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
            buffer_size=4,
            breakpoint_percentile_threshold=98,
            embed_model=self.embed_model,
        )

        self.service_context = ServiceContext.from_defaults(
            embed_model=self.embed_model, node_parser=self.splitter, llm=self.llm
        )

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
            self.query_engine = self.vector_store_index.as_query_engine()
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

    def document_indexing(
        self,
        file_paths: List[str],
        num_workers: int = cpu_count(),
    ):
        loader = SimpleDirectoryReader(input_files=file_paths)
        docs = loader.load_data(num_workers=num_workers)

        self.vector_store_index = VectorStoreIndex.from_documents(
            docs,
            storage_context=self.storage_context,
            service_context=self.service_context,
        )

        self.query_engine = self.vector_store_index.as_query_engine()

        return Response(
            content="Indexing Successfully Completed",
            status_code=status.HTTP_201_CREATED,
        )

    def document_querying(self, query_str: str):
        return self.query_engine.query(query_str)
