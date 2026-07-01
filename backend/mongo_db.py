"""
mongo_db.py — MongoDB + GridFS integration for CocoaGPT
==========================================================
Replaces Firestore for chat persistence. Adds GridFS for image storage,
which Firestore couldn't handle.

Collections
-----------
  users     — one doc per user (mirrors Firebase Auth uid)
  chats     — one doc per chat session
  messages  — one doc per message (text + recipe JSON + image refs)
  fs.files / fs.chunks  — GridFS, created automatically on first image upload

Design principle
-----------------
  Messages NEVER store raw image bytes. They store GridFS ObjectIds in
  `image_refs`. Images are fetched separately via a streaming endpoint
  (added in app.py) so chat history loads fast and stays lightweight.

Install
-------
    pip install pymongo

Usage in app.py
----------------
    from mongo_db import (
        create_chat, get_user_chats, delete_chat,
        save_message, get_chat_messages,
        save_image, get_image, delete_chat_images,
    )
"""

import os
import io
from datetime import datetime, timezone
from typing import Optional

from bson import ObjectId
from bson.errors import InvalidId
from pymongo import MongoClient, ASCENDING
from pymongo.errors import PyMongoError
import gridfs

from dotenv import load_dotenv
load_dotenv()

# =====================================================
# CONNECTION
# =====================================================

MONGO_URI     = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
MONGO_DB_NAME = os.getenv("MONGO_DB_NAME", "cocoagpt_db")

_client = MongoClient(MONGO_URI)
db      = _client[MONGO_DB_NAME]
fs      = gridfs.GridFS(db)   # GridFS bound to the same database

# Collections — match what you created in Compass
users_col    = db["users"]
chats_col    = db["chats"]
messages_col = db["messages"]

# Verify connection on import — fail fast if MongoDB isn't reachable
try:
    _client.admin.command("ping")
    print(f"[MongoDB] Connected to '{MONGO_DB_NAME}' at {MONGO_URI}")
except PyMongoError as e:
    print(f"[MongoDB] Connection FAILED: {e}")
    raise


# =====================================================
# HELPERS
# =====================================================

def _now() -> datetime:
    """UTC timestamp — always store timestamps in UTC, convert on display."""
    return datetime.now(timezone.utc)


def _oid(id_str: str) -> ObjectId:
    """
    Safely converts a string to ObjectId.
    Raises ValueError with a clear message if invalid — easier to debug
    than pymongo's raw InvalidId exception in FastAPI error responses.
    """
    try:
        return ObjectId(id_str)
    except (InvalidId, TypeError):
        raise ValueError(f"Invalid ObjectId: {id_str}")


def _serialize(doc: dict) -> dict:
    """
    Converts MongoDB document for JSON responses:
      - ObjectId  -> str
      - datetime  -> ISO 8601 string
    Recursively handles nested dicts/lists (e.g. image_refs list of ObjectIds).
    """
    if doc is None:
        return None

    out = {}
    for key, val in doc.items():
        if isinstance(val, ObjectId):
            out[key] = str(val)
        elif isinstance(val, datetime):
            out[key] = val.isoformat()
        elif isinstance(val, list):
            out[key] = [
                str(v) if isinstance(v, ObjectId) else
                v.isoformat() if isinstance(v, datetime) else v
                for v in val
            ]
        elif isinstance(val, dict):
            out[key] = _serialize(val)
        else:
            out[key] = val
    return out


# =====================================================
# USERS
# =====================================================

def ensure_user(uid: str, email: str = "", display_name: str = "") -> dict:
    """
    Creates the user document if it doesn't exist yet (upsert).
    Call this once after Firebase Auth login succeeds on the frontend,
    or lazily on first chat creation.
    """
    users_col.update_one(
        {"_id": uid},                      # use Firebase uid directly as _id
        {
            "$setOnInsert": {
                "_id":          uid,
                "email":        email,
                "display_name": display_name,
                "created_at":   _now(),
            }
        },
        upsert=True,
    )
    return _serialize(users_col.find_one({"_id": uid}))


# =====================================================
# CHATS
# =====================================================

def create_chat(user_id: str, title: str = "New Chat") -> dict:
    """Creates a new chat session. Returns the created chat doc."""
    doc = {
        "user_id":    user_id,
        "title":      title,
        "created_at": _now(),
        "updated_at": _now(),
    }
    result = chats_col.insert_one(doc)
    doc["_id"] = result.inserted_id
    return _serialize(doc)


def get_user_chats(user_id: str) -> list[dict]:
    """
    Returns all chats for a user, most recent first.
    Mirrors the Firestore query: orderBy("createdAt", "desc")
    """
    cursor = chats_col.find({"user_id": user_id}).sort("created_at", -1)
    return [_serialize(doc) for doc in cursor]


def get_chat(chat_id: str) -> Optional[dict]:
    """Fetches a single chat document by id."""
    doc = chats_col.find_one({"_id": _oid(chat_id)})
    return _serialize(doc)


def update_chat_title(chat_id: str, title: str) -> bool:
    """
    Updates a chat's title (e.g. first message becomes the title).
    Returns True if a document was actually modified.
    """
    result = chats_col.update_one(
        {"_id": _oid(chat_id)},
        {"$set": {"title": title, "updated_at": _now()}},
    )
    return result.modified_count > 0


def touch_chat(chat_id: str) -> None:
    """Updates updated_at — call after every new message so chat list can sort by recency."""
    chats_col.update_one(
        {"_id": _oid(chat_id)},
        {"$set": {"updated_at": _now()}},
    )


def delete_chat(chat_id: str) -> dict:
    """
    Deletes a chat AND cascades:
      - all its messages
      - all GridFS images referenced by those messages
    Returns a summary of what was deleted.
    """
    oid = _oid(chat_id)

    # 1. Find all messages in this chat (need their image_refs before deleting)
    messages = list(messages_col.find({"chat_id": chat_id}))

    # 2. Delete every GridFS image referenced by these messages
    deleted_images = 0
    for msg in messages:
        for img_id in msg.get("image_refs", []):
            try:
                fs.delete(img_id if isinstance(img_id, ObjectId) else ObjectId(img_id))
                deleted_images += 1
            except gridfs.NoFile:
                pass   # already gone — not an error

    # 3. Delete all messages
    msg_result = messages_col.delete_many({"chat_id": chat_id})

    # 4. Delete the chat itself
    chat_result = chats_col.delete_one({"_id": oid})

    return {
        "chat_deleted":    chat_result.deleted_count > 0,
        "messages_deleted": msg_result.deleted_count,
        "images_deleted":   deleted_images,
    }


# =====================================================
# MESSAGES
# =====================================================

def save_message(
    chat_id:     str,
    user_id:     str,
    role:        str,                       # "user" | "assistant"
    text:        str = "",
    recipe:      Optional[dict] = None,
    image_refs:  Optional[list[str]] = None,  # list of GridFS id strings
    image_urls:  Optional[list[str]] = None,  # external URLs (SerpAPI etc.) — NOT GridFS
    is_file:     bool = False,
    file_name:   str = "",
    detected_cake_type: Optional[str] = None,
) -> dict:
    """
    Saves a single chat message. Mirrors the shape your chat.jsx already
    builds for Firestore — same fields, different backend.

    image_refs  : GridFS ObjectIds for images WE stored (e.g. user uploads)
    image_urls  : external image URLs from SerpAPI — passed straight through,
                  never stored in GridFS since we don't own that data
    """
    doc = {
        "chat_id":     chat_id,
        "user_id":     user_id,
        "role":        role,
        "text":        text,
        "recipe":      recipe,
        "image_refs":  [
            oid if isinstance(oid, ObjectId) else _oid(oid)
            for oid in (image_refs or [])
        ],
        "image_urls":  image_urls or [],
        "is_file":     is_file,
        "file_name":   file_name,
        "detected_cake_type": detected_cake_type,
        "created_at":  _now(),
    }
    result = messages_col.insert_one(doc)
    doc["_id"] = result.inserted_id

    touch_chat(chat_id)   # bump chat's updated_at for recency sorting

    return _serialize(doc)


def get_chat_messages(chat_id: str) -> list[dict]:
    """
    Returns all messages for a chat, oldest first (natural conversation order).
    Mirrors Firestore: orderBy("createdAt")  [ascending]
    """
    cursor = messages_col.find({"chat_id": chat_id}).sort("created_at", ASCENDING)
    return [_serialize(doc) for doc in cursor]


def get_last_recipe_message(chat_id: str) -> Optional[dict]:
    """
    Finds the most recent assistant message that carried a structured recipe.
    Mirrors getLastRecipe() in chat.jsx — used to ground follow-up questions.
    """
    doc = messages_col.find_one(
        {"chat_id": chat_id, "role": "assistant", "recipe": {"$ne": None}},
        sort=[("created_at", -1)],
    )
    return _serialize(doc)["recipe"] if doc else None


# =====================================================
# GRIDFS — IMAGE STORAGE
# =====================================================

def save_image(
    image_bytes: bytes,
    filename:    str,
    content_type: str = "image/jpeg",
    user_id:     str = "",
    chat_id:     str = "",
) -> str:
    """
    Stores an image in GridFS. Returns the file's ObjectId as a string —
    this is what you put into a message's `image_refs` list.

    metadata (user_id, chat_id) lets us:
      - query "all images for this chat" without scanning messages
      - cascade-delete images when a chat is deleted
    """
    file_id = fs.put(
        image_bytes,
        filename=filename,
        content_type=content_type,
        metadata={
            "user_id": user_id,
            "chat_id": chat_id,
            "uploaded_at": _now(),
        },
    )
    return str(file_id)


def get_image(file_id: str) -> Optional[tuple[bytes, str]]:
    """
    Retrieves an image from GridFS.
    Returns (image_bytes, content_type) or None if not found.

    Used by the streaming endpoint in app.py:
        GET /images/{file_id}
    """
    try:
        grid_out = fs.get(_oid(file_id))
        data         = grid_out.read()
        content_type = grid_out.content_type or "image/jpeg"
        return data, content_type
    except (gridfs.NoFile, ValueError):
        return None


def delete_chat_images(chat_id: str) -> int:
    """
    Deletes all GridFS images whose metadata.chat_id matches.
    Useful as a standalone cleanup call (delete_chat() already does this
    internally, but this is handy if you ever need to run it separately).
    Returns number of images deleted.
    """
    count = 0
    for grid_file in fs.find({"metadata.chat_id": chat_id}):
        fs.delete(grid_file._id)
        count += 1
    return count


def get_chat_image_ids(chat_id: str) -> list[str]:
    """Returns all GridFS file ids belonging to a chat (debugging/admin use)."""
    return [str(f._id) for f in fs.find({"metadata.chat_id": chat_id})]


# =====================================================
# QUICK SELF-TEST  (run directly: python mongo_db.py)
# =====================================================

if __name__ == "__main__":
    print("\n--- MongoDB self-test ---")

    # Users
    user = ensure_user("test_uid_123", email="test@example.com", display_name="Test User")
    print(f"User: {user}")

    # Chats
    chat = create_chat("test_uid_123", title="Test Chat")
    print(f"Chat created: {chat['_id']}")

    chats = get_user_chats("test_uid_123")
    print(f"User has {len(chats)} chat(s)")

    # Messages
    msg1 = save_message(
        chat_id=chat["_id"], user_id="test_uid_123",
        role="user", text="Give me a banana cake recipe",
    )
    print(f"Message saved: {msg1['_id']}")

    msg2 = save_message(
        chat_id=chat["_id"], user_id="test_uid_123",
        role="assistant", text="",
        recipe={"recipe_name": "Banana Cake", "ingredients": ["bananas", "flour"]},
    )
    print(f"Recipe message saved: {msg2['_id']}")

    messages = get_chat_messages(chat["_id"])
    print(f"Chat has {len(messages)} message(s)")

    last_recipe = get_last_recipe_message(chat["_id"])
    print(f"Last recipe: {last_recipe}")

    # GridFS image test
    fake_image_bytes = b"\xff\xd8\xff\xe0fake_jpeg_data_for_testing"
    file_id = save_image(
        fake_image_bytes, filename="test.jpg",
        user_id="test_uid_123", chat_id=chat["_id"],
    )
    print(f"Image saved to GridFS: {file_id}")

    retrieved = get_image(file_id)
    if retrieved:
        data, ctype = retrieved
        print(f"Image retrieved: {len(data)} bytes, type={ctype}")

    # Cleanup
    result = delete_chat(chat["_id"])
    print(f"Cleanup: {result}")

    users_col.delete_one({"_id": "test_uid_123"})
    print("\n--- Self-test PASSED ---")