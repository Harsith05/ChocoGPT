from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from google import genai
import os
from dotenv import load_dotenv
import json


load_dotenv()

# =====================================================
# CONFIG
# =====================================================

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "recipes"

# =====================================================
# FASTAPI
# =====================================================

app = FastAPI(title="Recipe RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # React localhost
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# GEMINI
# =====================================================

client = genai.Client(
    api_key=GEMINI_API_KEY
)

# =====================================================
# EMBEDDING MODEL
# MUST MATCH THE MODEL USED DURING INDEXING
# =====================================================

embedder = SentenceTransformer(
    "BAAI/bge-m3"
)

# =====================================================
# QDRANT
# =====================================================

qdrant = QdrantClient(
    url=QDRANT_URL
)

# =====================================================
# REQUEST MODEL
# =====================================================

class ChatRequest(BaseModel):
    query: str

# =====================================================
# RETRIEVAL
# =====================================================

def retrieve_recipe(query: str):

    query_vector = embedder.encode(query).tolist()

    result = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=3
    )

    hits = result.points

    if not hits:
        return None

    return hits[0].payload

# =====================================================
# BUILD CONTEXT
# =====================================================

def build_context(recipe):

    ingredients = recipe.get("ingredients", [])
    steps = recipe.get("steps", [])

    if isinstance(ingredients, list):
        ingredients_text = "\n".join(
            f"- {item}" for item in ingredients
        )
    else:
        ingredients_text = str(ingredients)

    if isinstance(steps, list):
        steps_text = "\n".join(
            f"{i+1}. {step}"
            for i, step in enumerate(steps)
        )
    else:
        steps_text = str(steps)
    context = f"""
      Recipe Name:
       {recipe.get('recipe_name', '')}

      Recipe Content:
        {recipe.get('recipe_text', '')}
       """

    return context

# =====================================================
# GEMINI RESPONSE
# =====================================================

def generate_answer(user_query, context):

    prompt = f"""
Extract the recipe from the content below.

{context}

Return ONLY valid JSON:

{{
  "recipe_name": "",
  "prep_time": "",
  "servings": "",
  "ingredients": [],
  "instructions": [],
  "notes": "",
  "image_description": ""
}}
"""

    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config={
        "response_mime_type": "application/json"
    }
)
    

    return json.loads(response.text)


# =====================================================
# CHAT ENDPOINT
# =====================================================

@app.post("/chat")
async def chat(request: ChatRequest):

    recipe = retrieve_recipe(
        request.query
    )

    if recipe is None:
        return {
            "answer": "No matching recipe found.",
            "image_url": None
        }

    context = build_context(recipe)

    print("\n===== RETRIEVED RECIPE =====")
    print(recipe)
    print("\n===== CONTEXT SENT TO GEMINI =====")
    print(context)
    print("=============================\n")

    recipe_json = generate_answer(
    request.query,
    context
)

    image_paths = recipe.get("image_paths", [])

    image_urls = [
    f"http://localhost:8000/images/{os.path.basename(img)}"
    for img in image_paths
]

    return {
     "recipe": recipe_json,
    "image_urls": image_urls
}

# =====================================================
# HEALTH CHECK
# =====================================================
@app.get("/debug")
def debug():

    result = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        limit=3,
        with_payload=True
    )

    points = result[0]

    return [
        p.payload
        for p in points
    ]

@app.get("/")
def root():
    return {
        "status": "running",
        "service": "Recipe RAG API"
    }