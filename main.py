from fastapi import FastAPI, UploadFile
from index_utils import get_db_collections, decument_indexing, document_querying
from typing import List

app = FastAPI()

fastapi_app_version = "/api/v1"


@app.get(f"{fastapi_app_version}/get-collections/")
async def get_collections(vector_db: str = "chromadb"):
    return get_db_collections(vector_db=vector_db)


@app.post(f"{fastapi_app_version}/document-index/")
async def index_files(file_paths: List[str]):
    return decument_indexing(file_paths=file_paths)


@app.get(f"{fastapi_app_version}/document-query/")
async def get_collections(query: str):
    return document_querying(query_str=query)
