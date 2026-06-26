import os
import json
import httpx

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchText,
    ScoredPoint,
)

from sentence_transformers import SentenceTransformer, CrossEncoder

from google import genai
from google.genai import types

from dotenv import load_dotenv

load_dotenv()

# =====================================================
# CONFIG
# =====================================================

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY")
SERPAPI_KEY     = os.getenv("SERPAPI_KEY")

QDRANT_URL      = "http://localhost:6333"
COLLECTION_NAME = "recipes"

DENSE_TOP_K     = 10   # candidates from vector search
SPARSE_TOP_K    = 10   # candidates from keyword search
RERANK_TOP_N    = 3    # keep after reranking

# =====================================================
# FASTAPI
# =====================================================

app = FastAPI(title="Recipe RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =====================================================
# MODELS
# =====================================================

print("Loading BGE-M3 embedder...")
embedder = SentenceTransformer("BAAI/bge-m3")

print("Loading reranker (BGE-reranker-v2-m3)...")
reranker = CrossEncoder("BAAI/bge-reranker-v2-m3")

# =====================================================
# CLIENTS
# =====================================================

gemini_client = genai.Client(api_key=GEMINI_API_KEY)

qdrant = QdrantClient(url=QDRANT_URL)

# =====================================================
# REQUEST MODEL
# =====================================================

class ChatRequest(BaseModel):
    query: str

# =====================================================
# HYBRID RETRIEVAL  (dense + keyword filter + rerank)
# =====================================================

def hybrid_retrieve(query: str) -> dict | None:
    """
    1. Dense vector search  (BGE-M3 embeddings)
    2. Keyword filter search using MatchText on recipe_name + recipe_text
    3. Merge candidate pools  (deduplicate by id)
    4. Rerank with BGE-reranker-v2-m3
    5. Return the top-1 payload
    """

    # --------------------------------------------------
    # 1. DENSE SEARCH
    # --------------------------------------------------
    query_vector = embedder.encode(
        query,
        normalize_embeddings=True
    ).tolist()

    dense_result = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=DENSE_TOP_K,
        with_payload=True
    )

    dense_hits: list[ScoredPoint] = dense_result.points

    # --------------------------------------------------
    # 2. KEYWORD SEARCH
    # FIX: qdrant-client 1.18.0 has no TextQuery.
    # Use scroll() with a MatchText payload filter instead —
    # one filter per field, then union the results.
    # --------------------------------------------------
    keyword_hits_raw = []

    for field in ["recipe_name", "recipe_text"]:
        try:
            scroll_result = qdrant.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[
                        FieldCondition(
                            key=field,
                            match=MatchText(text=query)
                        )
                    ]
                ),
                limit=SPARSE_TOP_K,
                with_payload=True,
                with_vectors=False,
            )
            keyword_hits_raw.extend(scroll_result[0])
        except Exception as e:
            print(f"[Keyword Search] field={field} error: {e}")

    # --------------------------------------------------
    # 3. MERGE  (deduplicate, dense hits take priority)
    # --------------------------------------------------
    seen: dict[str, object] = {}

    for hit in dense_hits:
        seen[str(hit.id)] = hit

    for point in keyword_hits_raw:
        if str(point.id) not in seen:
            seen[str(point.id)] = point

    candidates = list(seen.values())

    if not candidates:
        return None

    # --------------------------------------------------
    # 4. RERANK
    # --------------------------------------------------
    pairs = [
        (query, c.payload.get("recipe_text", ""))
        for c in candidates
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, candidates),
        key=lambda x: x[0],
        reverse=True
    )

    top_hit = ranked[0][1]

    print(f"\n[Hybrid Retrieve] candidates={len(candidates)}")
    print(f"[Hybrid Retrieve] top recipe = {top_hit.payload.get('recipe_name')}")

    return top_hit.payload


# =====================================================
# SERPAPI  IMAGE SEARCH
# =====================================================

def search_recipe_image(recipe_name: str) -> list[str]:
    """
    Calls SerpAPI Google Images endpoint.
    Returns up to 3 image URLs.
    """

    # Read fresh each call — module-level var captures None if .env
    # wasn't loaded yet at import time
    api_key = os.getenv("SERPAPI_KEY")

    print(f"[SerpAPI] Key present: {bool(api_key)}")

    if not api_key:
        print("[SerpAPI] SERPAPI_KEY not set, skipping image search.")
        return []

    params = {
        "engine":  "google_images",
        "q":       f"{recipe_name} recipe",
        "api_key": api_key,
        "num":     3,
        "safe":    "active",
    }

    try:
        print(f"[SerpAPI] Searching for: {recipe_name}")
        response = httpx.get(
            "https://serpapi.com/search",
            params=params,
            timeout=15.0
        )
        print(f"[SerpAPI] Status: {response.status_code}")
        response.raise_for_status()
        data = response.json()

        images_results = data.get("images_results", [])
        print(f"[SerpAPI] Raw results count: {len(images_results)}")

        urls = [
            img["original"]
            for img in images_results[:3]
            if img.get("original")
        ]

        print(f"[SerpAPI] Found {len(urls)} images for '{recipe_name}'")
        return urls

    except Exception as e:
        print(f"[SerpAPI] Error type: {type(e).__name__}")
        print(f"[SerpAPI] Error detail: {e}")
        return []


# =====================================================
# GEMINI TOOL  DEFINITION
# =====================================================

IMAGE_SEARCH_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_recipe_image",
            description=(
                "Search the web for food/recipe images using SerpAPI. "
                "Call this whenever the user asks for a recipe so a "
                "relevant dish photo can be returned alongside the recipe."
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "recipe_name": types.Schema(
                        type=types.Type.STRING,
                        description="The exact name of the recipe to search images for."
                    )
                },
                required=["recipe_name"]
            )
        )
    ]
)

# =====================================================
# TOOL DISPATCHER
# =====================================================

def dispatch_tool(tool_call) -> str:
    """Execute the tool Gemini requested and return a JSON string result."""

    name = tool_call.name
    args = tool_call.args or {}

    if name == "search_recipe_image":
        urls = search_recipe_image(args.get("recipe_name", ""))
        return json.dumps({"image_urls": urls})

    return json.dumps({"error": f"Unknown tool: {name}"})


# =====================================================
# GEMINI AGENTIC RESPONSE  (tool-use loop)
# =====================================================

def generate_answer(
    user_query: str,
    context: str
) -> tuple[dict, list[str]]:
    """
    Runs a Gemini tool-use loop:
      Turn 1  → model extracts recipe JSON + may call search_recipe_image
      Turn 2  → we send tool result back
      Turn 3  → model returns final JSON with image_urls injected

    Returns (recipe_dict, image_urls).
    """

    system_prompt = """
You are a helpful recipe assistant.
When given recipe context, you MUST:
1. Call the search_recipe_image tool with the recipe name.
2. After receiving image results, return ONLY valid JSON in this exact schema:

{
  "recipe_name": "",
  "prep_time": "",
  "servings": "",
  "ingredients": [],
  "instructions": [],
  "notes": "",
  "image_urls": []
}

Put the image URLs returned by the tool into the image_urls field.
Do NOT include markdown fences or any text outside the JSON.
"""

    user_message = f"""
Extract the recipe from the context below and fetch an image for it.

{context}

User query: {user_query}
"""

    messages = [
        types.Content(
            role="user",
            parts=[types.Part(text=user_message)]
        )
    ]

    image_urls: list[str] = []

    # --------------------------------------------------
    # AGENTIC LOOP  (max 3 turns to avoid infinite loops)
    # --------------------------------------------------
    for turn in range(3):

        # Use tools config (no response_mime_type — Gemini rejects both together).
        # The system prompt already enforces JSON output on the final turn.
        has_pending_tools = any(
            p.function_call is not None
            for m in messages
            for p in (m.parts if hasattr(m, "parts") else [])
        )

        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=messages,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                tools=[IMAGE_SEARCH_TOOL],
            )
        )

        candidate = response.candidates[0]
        parts      = candidate.content.parts

        # Check if Gemini wants to call a tool
        tool_calls = [
            p.function_call
            for p in parts
            if p.function_call is not None
        ]

        if tool_calls:
            messages.append(
                types.Content(
                    role="model",
                    parts=parts
                )
            )

            tool_result_parts = []

            for tc in tool_calls:
                result_str = dispatch_tool(tc)

                try:
                    result_data = json.loads(result_str)
                    image_urls.extend(result_data.get("image_urls", []))
                except Exception:
                    pass

                tool_result_parts.append(
                    types.Part(
                        function_response=types.FunctionResponse(
                            name=tc.name,
                            response={"result": result_str}
                        )
                    )
                )

            messages.append(
                types.Content(
                    role="user",
                    parts=tool_result_parts
                )
            )

            continue

        # No tool call → parse the final text response
        text_parts = [p.text for p in parts if p.text]
        raw_text = "\n".join(text_parts).strip()

        raw_text = raw_text.replace("```json", "").replace("```", "").strip()

        try:
            recipe_dict = json.loads(raw_text)
        except json.JSONDecodeError:
            recipe_dict = {
                "recipe_name": "",
                "prep_time": "",
                "servings": "",
                "ingredients": [],
                "instructions": [],
                "notes": raw_text,
                "image_urls": []
            }

        existing = recipe_dict.get("image_urls", [])
        recipe_dict["image_urls"] = list(dict.fromkeys(existing + image_urls))

        return recipe_dict, recipe_dict["image_urls"]

    # Fallback if loop exhausted
    return {
        "recipe_name": "",
        "prep_time": "",
        "servings": "",
        "ingredients": [],
        "instructions": [],
        "notes": "Could not generate recipe.",
        "image_urls": image_urls
    }, image_urls


# =====================================================
# BUILD CONTEXT
# =====================================================

def build_context(recipe: dict) -> str:
    return (
        f"Recipe Name:\n  {recipe.get('recipe_name', '')}\n\n"
        f"Recipe Content:\n  {recipe.get('recipe_text', '')}"
    )


# =====================================================
# CHAT ENDPOINT
# =====================================================

@app.post("/chat")
async def chat(request: ChatRequest):

    # 1. Hybrid retrieve
    recipe = hybrid_retrieve(request.query)

    if recipe is None:
        return {
            "answer": "No matching recipe found.",
            "image_urls": []
        }

    context = build_context(recipe)

    print("\n===== RETRIEVED RECIPE =====")
    print(recipe.get("recipe_name"))
    print("\n===== CONTEXT SENT TO GEMINI =====")
    print(context[:500], "...")
    print("=================================\n")

    # 2. Generate answer + fetch images via tool calling
    recipe_json, image_urls = generate_answer(
        request.query,
        context
    )

    return {
        "recipe":     recipe_json,
        "image_urls": image_urls
    }


# =====================================================
# DEBUG / HEALTH
# =====================================================

@app.get("/debug")
def debug():
    result = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        limit=3,
        with_payload=True
    )
    return [p.payload for p in result[0]]


@app.get("/")
def root():
    return {
        "status":  "running",
        "service": "Recipe RAG API"
    }