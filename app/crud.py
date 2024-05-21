from sqlalchemy.orm import Session

from model import chatbot, base_rag_settings
import schema


def get_chatbots(db: Session):
    """
    Retrieve all chatbots from the database.

    Args:
        db (Session): The database session.

    Returns:
        List[chatbot.BaseChatBot]: A list of all chatbots in the database.
    """
    return db.query(chatbot.BaseChatBot).all()


def create_chatbot(db: Session, chatbotModel: schema.BaseChatBotCreate):
    """
    Create a new chatbot entry in the database.

    Args:
        db (Session): The database session.
        chatbotModel (schema.BaseChatBotCreate): The schema representing the chatbot to be created.

    Returns:
        chatbot.BaseChatBot: The created chatbot instance.
    """
    db_chatbot = chatbot.BaseChatBot(
        name=chatbotModel.name,
        description=chatbotModel.description,
        vector_db=chatbotModel.vector_db,
        vector_db_collection=chatbotModel.vector_db_collection,
        embed_model_provider=chatbotModel.embed_model_provider,
        embed_model=chatbotModel.embed_model,
        llm_provider=chatbotModel.llm_provider,
        llm=chatbotModel.llm,
        load_in_4bit=chatbotModel.load_in_4bit,
        chunking_strategy=chatbotModel.chunking_strategy,
        semantic_splitting_buffer_size=chatbotModel.semantic_splitting_buffer_size,
        semantic_splitting_breakpoint_percentile_threshold=chatbotModel.semantic_splitting_breakpoint_percentile_threshold,
        retriver_top_k=chatbotModel.retriver_top_k,
    )
    db.add(db_chatbot)
    db.commit()
    db.refresh(db_chatbot)
    return db_chatbot


def get_base_model_settings(db: Session):
    """
    Retrieve the most recent base model settings from the database.

    Args:
        db (Session): The database session.

    Returns:
        base_rag_settings.BasicRAGSettingsModel: The most recent base model settings.
    """
    return (
        db.query(base_rag_settings.BasicRAGSettingsModel)
        .order_by(base_rag_settings.BasicRAGSettingsModel.id.desc())
        .first()
    )


def create_base_model_settings(
    db: Session, base_rag_settingsModel: schema.BaseRAGModel
):
    """
    Create a new base model settings entry in the database.

    Args:
        db (Session): The database session.
        base_rag_settingsModel (schema.BaseRAGModel): The schema representing the base model settings to be created.

    Returns:
        base_rag_settings.BasicRAGSettingsModel: The created base model settings instance.
    """
    db_base_rag_settings = base_rag_settings.BasicRAGSettingsModel(
        vector_db=base_rag_settingsModel.vector_db,
        vector_db_collection=base_rag_settingsModel.vector_db_collection,
        embed_model_provider=base_rag_settingsModel.embed_model_provider,
        embed_model=base_rag_settingsModel.embed_model,
        llm_provider=base_rag_settingsModel.llm_provider,
        llm=base_rag_settingsModel.llm,
        load_in_4bit=base_rag_settingsModel.load_in_4bit,
        chunking_strategy=base_rag_settingsModel.chunking_strategy,
        semantic_splitting_buffer_size=base_rag_settingsModel.semantic_splitting_buffer_size,
        semantic_splitting_breakpoint_percentile_threshold=base_rag_settingsModel.semantic_splitting_breakpoint_percentile_threshold,
        retriver_top_k=base_rag_settingsModel.retriver_top_k,
    )
    db.add(db_base_rag_settings)
    db.commit()
    db.refresh(db_base_rag_settings)
    return db_base_rag_settings
