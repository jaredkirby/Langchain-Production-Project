from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import redis
import requests
import json
import logging

# Initialize logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Redis
try:
    r = redis.Redis(host="redis", port=6379, db=0)
    logger.info("Successfully connected to Redis.")
except Exception as e:
    logger.error(f"Failed to connect to Redis: {e}")

# Initialize FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
logger.info("FastAPI and middleware initialized.")


# Data models
class Message(BaseModel):
    role: str
    content: str


class Conversation(BaseModel):
    conversation: List[Message]


logger.info("Data models initialized.")


# GET Endpoint to fetch conversation
@app.get("/service2/{conversation_id}")
async def get_conversation(conversation_id: str):
    logger.info(f"Received GET request for conversation_id: {conversation_id}")
    try:
        existing_conversation_json = r.get(conversation_id)
        if existing_conversation_json:
            existing_conversation = json.loads(existing_conversation_json)
            logger.info(f"Conversation found for ID {conversation_id}.")
            return existing_conversation
        else:
            logger.warning(f"No conversation found for ID {conversation_id}.")
            return {"error": "Conversation not found"}
    except Exception as e:
        logger.error(f"An error occurred while processing GET request: {e}")
        return {"error": "Internal Server Error"}


# POST Endpoint to update conversation and get reply
@app.post("/service2/{conversation_id}")
async def service2(conversation_id: str, conversation: Conversation):
    logger.info(f"Received POST request for conversation_id: {conversation_id}")
    try:
        existing_conversation_json = r.get(conversation_id)
        if existing_conversation_json:
            existing_conversation = json.loads(existing_conversation_json)
        else:
            existing_conversation = {
                "conversation": [
                    {"role": "system", "content": "You are a helpful assistant."}
                ]
            }

        existing_conversation["conversation"].append(
            conversation.model_dump()["conversation"][-1]
        )

        logger.info(f"Sending updated conversation to service3.")
        response = requests.post(
            f"http://service3:80/service3/{conversation_id}", json=existing_conversation
        )
        response.raise_for_status()

        assistant_message = response.json()["reply"]
        existing_conversation["conversation"].append(
            {"role": "assistant", "content": assistant_message}
        )

        r.set(conversation_id, json.dumps(existing_conversation))
        logger.info(f"Updated conversation saved to Redis.")

        return existing_conversation
    except Exception as e:
        logger.error(f"An error occurred while processing POST request: {e}")
        return {"error": "Internal Server Error"}
