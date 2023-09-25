from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import os
from fastapi.middleware.cors import CORSMiddleware
import openai
from langchain.prompts import PromptTemplate
import logging
from dotenv import find_dotenv, load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
)

ROLE_CLASS_MAP = {"assistant": AIMessage, "user": HumanMessage, "system": SystemMessage}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
try:
    load_dotenv(find_dotenv())
    openai.api_key = os.getenv("OPENAI_API_KEY")
    connection_string = os.getenv("CONNECTION_STRING")
    collection_name = os.getenv("COLLECTION_NAME")

    logger.info("Environment variables loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load environment variables: {e}")


class Message(BaseModel):
    role: str
    content: str


class Conversation(BaseModel):
    conversation: List[Message]


# Initialize data models, embeddings, and vector store
try:
    embeddings = OpenAIEmbeddings()
    chat = ChatOpenAI(temperature=0)
    store = PGVector(
        collection_name=collection_name,
        connection_string=connection_string,
        embedding_function=embeddings,
    )
    retriever = store.as_retriever()

    logger.info("Successfully initialized data models, embeddings, and vector store.")
except Exception as e:
    logger.error(f"Initialization failed: {e}")

# Initialize FastAPI and middleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("FastAPI and middleware initialized.")


# ... (Other code definitions like create_messages and format_docs)
prompt_template = """As a FAQ Bot for our restaurant, you have the following information about our restaurant:

{context}

Please provide the most suitable response for the users question.
Answer:"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)


def create_messages(conversation):
    return [
        ROLE_CLASS_MAP[message.role](content=message.content)
        for message in conversation
    ]


def format_docs(docs):
    formatted_docs = []
    for doc in docs:
        formatted_doc = "Source: " + doc.metadata["source"]
        formatted_docs.append(formatted_doc)
    return "\n".join(formatted_docs)


# POST Endpoint for generating an assistant's reply
@app.post("/service3/{conversation_id}")
async def service3(conversation_id: str, conversation: Conversation):
    logger.info(f"Received POST request for conversation_id: {conversation_id}")
    try:
        # Retrieve relevant documents
        query = conversation.conversation[-1].content
        docs = retriever.get_relevant_documents(query=query)
        logger.info("Successfully retrieved relevant documents.")

        # Format documents and prepare prompt
        docs = format_docs(docs=docs)
        prompt = system_message_prompt.format(context=docs)
        messages = [prompt] + create_messages(conversation=conversation.conversation)

        # Generate assistant reply using chat
        result = chat(messages)
        logger.info("Successfully generated assistant's reply.")

        return {"id": conversation_id, "reply": result.content}
    except Exception as e:
        logger.error(f"An error occurred while processing POST request: {e}")
        return {"error": "Internal Server Error"}
