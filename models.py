from pydantic import BaseModel


class BaseRAGModel(BaseModel):
    vector_db: str
    vector_db_collection: str
    embed_model_provider: str
    embed_model: str
    llm_provider: str
    llm: str
    load_in_4bit: bool
    chunking_strategy: str
    semantic_splitting_buffer_size: int
    semantic_splitting_breakpoint_percentile_threshold: int
    retriver_top_k: int
