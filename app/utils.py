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

DB = chromadb.PersistentClient("chromadb")

EMBED_MODEL = HuggingFaceEmbedding(
    model_name="Snowflake/snowflake-arctic-embed-l",
    trust_remote_code=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
)

LLM = HuggingFaceLLM(
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


SPLITTER = SemanticSplitterNodeParser(
    buffer_size=4, breakpoint_percentile_threshold=98, embed_model=EMBED_MODEL
)

SERVICE_CONTEXT = ServiceContext.from_defaults(
    embed_model=EMBED_MODEL, node_parser=SPLITTER, llm=LLM
)

if "default" in [collection_name.name for collection_name in DB.list_collections()]:
    CHROMA_COLLECTION = DB.get_or_create_collection("default")

    VECTOR_STORE = ChromaVectorStore(CHROMA_COLLECTION)
    STORAGE_CONTEXT = StorageContext.from_defaults(vector_store=VECTOR_STORE)
    VECTORE_STORE_INDEX = VectorStoreIndex.from_vector_store(
        vector_store=VECTOR_STORE,
        service_context=SERVICE_CONTEXT,
        storage_context=STORAGE_CONTEXT,
    )
    QUERY_ENGINE = VECTORE_STORE_INDEX.as_query_engine()
else:
    CHROMA_COLLECTION = DB.get_or_create_collection("default")

    VECTOR_STORE = ChromaVectorStore(CHROMA_COLLECTION)
    STORAGE_CONTEXT = StorageContext.from_defaults(vector_store=VECTOR_STORE)


def get_db_collections(vector_db: str):
    if vector_db == "chromadb":
        return DB.list_collections()
    else:
        return HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="DB Not Found",
        )


def document_indexing(
    file_paths: List[str],
    num_workers: int = cpu_count(),
):
    loader = SimpleDirectoryReader(input_files=file_paths)
    docs = loader.load_data(num_workers=num_workers)

    VECTORE_STORE_INDEX = VectorStoreIndex.from_documents(
        docs,
        storage_context=STORAGE_CONTEXT,
        service_context=SERVICE_CONTEXT,
    )

    return Response(
        content="Indexing Successfully Completed",
        status_code=status.HTTP_201_CREATED,
    )


def document_querying(query_str: str):
    return QUERY_ENGINE.query(query_str)
