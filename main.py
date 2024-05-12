from fastapi import FastAPI, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from basic_rag_workflow.basic_rag_worflow import BasicRagWorkflow
from typing import List
from models import BaseRAGModel
import aiofiles

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


@app.get(f"{fastapi_app_version}/get-basic-settings/")
async def get_basic_settings():
    return basic.get_basic_settings()


@app.post(f"{fastapi_app_version}/update-basic-settings/")
async def update_basic_settings(basic_settings: BaseRAGModel):
    basic.update_basic_settings(basic_settings=basic_settings)


@app.post(f"{fastapi_app_version}/document-index/")
async def index_files(files: List[UploadFile]):
    file_paths = []
    for file in files:
        contents = await file.read()
        async with aiofiles.open(f"data/{file.filename}", "wb") as f:
            await f.write(contents)
        file_paths.append(f"data/{file.filename}")

    return basic.document_indexing(file_paths=file_paths)


@app.get(f"{fastapi_app_version}/document-query/")
async def get_collections(query: str):
    Response(basic.document_querying(query_str=query))
