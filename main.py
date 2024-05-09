from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from basic_rag_workflow.basic_rag_worflow import BasicRagWorkflow
from typing import List

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

fastapi_app_version = "/api/v1"

basic = BasicRagWorkflow()


@app.get(f"{fastapi_app_version}/get-collections/")
async def get_collections(vector_db_name: str = "chromadb"):
    return basic.get_db_collections(vector_db_name=vector_db_name)


@app.post(f"{fastapi_app_version}/document-index/")
async def index_files(file_paths: List[str]):
    return basic.document_indexing(file_paths=file_paths)


@app.get(f"{fastapi_app_version}/document-query/")
async def get_collections(query: str):
    StreamingResponse(basic.document_querying(query_str=query))
