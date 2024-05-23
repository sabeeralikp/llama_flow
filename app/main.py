from fastapi import FastAPI, Response, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from basic_rag_workflow.basic_rag_worflow import BasicRagWorkflow
from typing import List
from schema import BaseChatBotBaseModel, BaseRAGModel
import aiofiles
import crud
from database.database import SessionLocal, engine
from sqlalchemy.orm import Session
from model import chatbot

# Create all tables in the database
chatbot.Base.metadata.create_all(bind=engine)

app = FastAPI()

# Middleware to handle CORS (Cross-Origin Resource Sharing)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API version prefix
fastapi_app_version = "/api/v1"

# Initialize the BasicRagWorkflow
basic = BasicRagWorkflow()


# Dependency to get the database session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Function to set initial settings for the application
def setSettings():
    print(default_backend)
    base_dict = basic.get_basic_settings()
    base_rag_model = BaseRAGModel(
        vector_db=base_dict["vector_db"][0],
        vector_db_collection=base_dict["vector_db_collection"],
        embed_model_provider=(
            "ollama" if default_backend == "ollama" else "huggingface"
        ),
        embed_model=(
            base_dict["ollama_embed_models"][0]
            if default_backend == "ollama"
            else base_dict["embed_model"][0]
        ),
        llm_provider=default_backend,
        llm=(
            base_dict["ollama"][0]
            if default_backend == "ollama"
            else (
                base_dict["llama_cpp"][0]
                if default_backend == "llamacpp"
                else base_dict["huggingface_llm"][0]
            )
        ),
        load_in_4bit=base_dict["load_in_4bit"],
        chunking_strategy=base_dict["chunking_strategy"][0],
        semantic_splitting_buffer_size=base_dict["semantic-splitting"]["buffer_size"],
        semantic_splitting_breakpoint_percentile_threshold=base_dict[
            "semantic-splitting"
        ]["breakpoint_percentile_threshold"],
        retriver_top_k=base_dict["retriver"]["top-k"],
    )
    db = SessionLocal()
    try:
        crud.create_base_model_settings(db=db, base_rag_settingsModel=base_rag_model)
    finally:
        db.close()


# Set initial settings when the application starts
setSettings()


@app.get(f"{fastapi_app_version}/get-collections/")
async def get_collections(vector_db_name: str = "chromadb"):
    """
    Get the list of collections from the vector database.

    Args:
        vector_db_name (str): The name of the vector database. Default is "chromadb".

    Returns:
        List of collections in the specified vector database.
    """
    return basic.get_db_collections(vector_db_name=vector_db_name)


@app.get(f"{fastapi_app_version}/get-basic-settings/")
async def get_basic_settings():
    """
    Get the basic settings for the application.

    Returns:
        Basic settings as a dictionary.
    """
    return basic.get_basic_settings()


@app.get(f"{fastapi_app_version}/get-current-basic-settings/")
async def get_current_basic_settings(db: Session = Depends(get_db)):
    """
    Get the current basic settings from the database.

    Args:
        db (Session): The database session.

    Returns:
        Current basic settings from the database.
    """
    return crud.get_base_model_settings(db=db)


@app.post(f"{fastapi_app_version}/update-basic-settings/")
async def update_basic_settings(
    basic_settings: BaseRAGModel, db: Session = Depends(get_db)
):
    """
    Update the basic settings and save them to the database.

    Args:
        basic_settings (BaseRAGModel): The new basic settings.
        db (Session): The database session.

    Returns:
        Updated basic settings saved in the database.
    """
    basic.update_basic_settings(basic_settings=basic_settings)
    return crud.create_base_model_settings(db=db, base_rag_settingsModel=basic_settings)


@app.post(f"{fastapi_app_version}/document-index/")
async def index_files(files: List[UploadFile]):
    """
    Index uploaded files.

    Args:
        files (List[UploadFile]): List of files to be indexed.

    Returns:
        Response indicating the status of indexing.
    """
    file_paths = []
    for file in files:
        contents = await file.read()
        async with aiofiles.open(f"data/{file.filename}", "wb") as f:
            await f.write(contents)
        file_paths.append(f"data/{file.filename}")

    return basic.document_indexing(file_paths=file_paths)


@app.get(f"{fastapi_app_version}/document-query/")
async def get_collections(query: str):
    """
    Query the indexed documents.

    Args:
        query (str): The query string.

    Returns:
        Query results from the indexed documents.
    """
    return basic.document_querying(query_str=query)


@app.get(f"{fastapi_app_version}/get-chatbots/")
async def get_chatbots(db: Session = Depends(get_db)):
    """
    Get the list of all chatbots from the database.

    Args:
        db (Session): The database session.

    Returns:
        List of chatbots in the database.
    """
    return crud.get_chatbots(db=db)


@app.post(f"{fastapi_app_version}/create-chatbots/")
async def create_chatbots(chatbot: BaseChatBotBaseModel, db: Session = Depends(get_db)):
    """
    Create a new chatbot in the database.

    Args:
        chatbot (BaseChatBotBaseModel): The chatbot data to be created.
        db (Session): The database session.

    Returns:
        The created chatbot instance.
    """
    return crud.create_chatbot(db=db, chatbotModel=chatbot)
