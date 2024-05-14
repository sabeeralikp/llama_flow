from sqlalchemy.orm import Session

from model import chatbot
import schema


def get_chatbots(db: Session):
    return db.query(chatbot.BaseChatBot).all()


def create_chatbot(db: Session, chatbotModel: schema.BaseChatBotCreate):
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
