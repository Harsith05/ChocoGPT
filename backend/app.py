import os
import re
import json
import httpx

from typing import Optional, TypedDict

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from pydantic import BaseModel

from mongo_db import (
    ensure_user,
    create_chat,
    get_user_chats,
    get_chat,
    update_chat_title,
    delete_chat,
    save_message,
    get_chat_messages,
    get_last_recipe_message,
    save_image,
    get_image,
)

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

from langgraph.graph import StateGraph, START, END

# LangSmith
from langsmith import Client as LangSmithClient
from langsmith.run_helpers import traceable

from dotenv import load_dotenv

load_dotenv()

# =====================================================
# CONFIG
# =====================================================

GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY")
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")

QDRANT_URL      = "http://localhost:6333"
COLLECTION_NAME = "recipes"

DENSE_TOP_K  = 10
SPARSE_TOP_K = 10

# Max turns of conversation history sent to Gemini per request
HISTORY_WINDOW = 10

# =====================================================
# LANGSMITH SETUP
# =====================================================

os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"]    = LANGSMITH_API_KEY or ""
os.environ["LANGCHAIN_PROJECT"]    = "CocoaGPT-RAG"

langsmith_client = LangSmithClient(api_key=LANGSMITH_API_KEY) if LANGSMITH_API_KEY else None

# =====================================================
# FASTAPI
# =====================================================

app = FastAPI(title="CocoaGPT RAG API")

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
qdrant        = QdrantClient(url=QDRANT_URL)

# =====================================================
# REQUEST MODELS
# =====================================================

class HistoryMessage(BaseModel):
    role: str          # "user" | "assistant"
    text: str          # plain text summary for context window

class ChatRequest(BaseModel):
    query:       str
    history:     list[HistoryMessage] = []   # last N turns from frontend
    last_recipe: Optional[dict]       = None  # structured recipe from previous turn, if any


# ---- MongoDB persistence request models ----

class EnsureUserRequest(BaseModel):
    uid:          str
    email:        str = ""
    display_name: str = ""


class CreateChatRequest(BaseModel):
    user_id: str
    title:   str = "New Chat"


class UpdateChatTitleRequest(BaseModel):
    title: str


class SaveMessageRequest(BaseModel):
    user_id:            str
    role:                str                  # "user" | "assistant"
    text:                str          = ""
    recipe:              Optional[dict] = None
    image_urls:          list[str]    = []    # external URLs (SerpAPI etc.)
    is_file:             bool         = False
    file_name:           str          = ""
    detected_cake_type:  Optional[str] = None

# =====================================================
# DOMAIN CLASSIFIER  (pure keyword/regex — no LLM call)
# =====================================================

# Anything related to cake / chocolate / baking lives here.
# Kept broad on purpose: better to let a borderline query through to RAG
# (which will just fail to find a match) than to wrongly block it.
DOMAIN_KEYWORDS = [
    "cake", "cakes", "chocolate", "cocoa", "choco", "brownie", "brownies",
    "cupcake", "cupcakes", "frosting", "icing", "ganache", "batter",
    "sponge", "fondant", "bake", "baking", "baker", "recipe", "recipes",
    "ingredient", "ingredients", "dessert", "desserts", "pastry", "pastries",
    "cheesecake", "muffin", "muffins", "cookie", "cookies", "pie",
    "vanilla", "red velvet", "black forest", "tart", "macaron", "macarons",
    "custard", "caramel", "truffle", "truffles", "glaze",
]

DOMAIN_RE = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in DOMAIN_KEYWORDS) + r")\b",
    re.IGNORECASE,
)

def is_domain_query(text: str) -> bool:
    return bool(DOMAIN_RE.search(text or ""))


# =====================================================
# FOLLOW-UP DETECTION  (pure keyword/regex — no LLM call)
# =====================================================

# Words/phrases that strongly suggest the user is asking about a NEW dish
# rather than continuing the conversation about the previously retrieved one.
NEW_DISH_SIGNAL_RE = re.compile(
    r"\b(recipe for|how (do|to) (i|you) make|give me|show me|"
    r"i want|can you (make|give|find)|search for|another|different|"
    r"instead)\b",
    re.IGNORECASE,
)

def is_recipe_followup_query(query: str, history: list, last_recipe: Optional[dict]) -> bool:
    """
    Recipe-grounded follow-up: there's a structured recipe already in
    play, AND the query doesn't signal a request for a brand new dish.
    e.g. "only 2 eggs enough for this cake", "what about the icing".
    """
    if not history or not last_recipe:
        return False

    if NEW_DISH_SIGNAL_RE.search(query):
        return False

    return True


def is_general_followup_query(query: str, history: list) -> bool:
    """
    General conversational follow-up: NOT a domain (cake/chocolate) query
    on its own, but there IS prior conversation history that might give
    it meaning — e.g. "I'm allergic to nuts" earlier, then later "does
    that apply to what I asked before". Without this check, any message
    that doesn't contain a domain keyword and hasn't yet produced a
    structured recipe falls through to off_topic with zero memory of
    the conversation, which is the gap this fixes.
    """
    if not history:
        return False

    if is_domain_query(query):
        return False  # handled by the domain/recipe routing instead

    if NEW_DISH_SIGNAL_RE.search(query):
        return False  # still signals a fresh request, not a continuation

    return True


# =====================================================
# QUANTITY DETECTION HELPERS
# =====================================================

QUANTITY_RE = re.compile(
    r'(\d+(?:\.\d+)?)\s*'
    r'(kg|g|gram|grams|pound|pounds|lb|lbs|serving|servings|portion|portions|people|person)',
    re.IGNORECASE
)

def detect_requested_quantity(query: str) -> Optional[dict]:
    match = QUANTITY_RE.search(query)
    if not match:
        return None
    value = float(match.group(1))
    unit  = match.group(2).lower().rstrip("s")
    return {"value": value, "unit": unit}


def estimate_recipe_quantity(recipe_text: str) -> Optional[dict]:
    match = QUANTITY_RE.search(recipe_text)
    if not match:
        return None
    value = float(match.group(1))
    unit  = match.group(2).lower().rstrip("s")
    return {"value": value, "unit": unit}

# =====================================================
# DISH NAME EXTRACTION  (for parallel image search keyword)
# =====================================================

# Strips common request scaffolding so the image-search node has a clean
# dish keyword to search with WITHOUT waiting on RAG to return a name.
STRIP_PHRASES_RE = re.compile(
    r"\b(recipe for|recipe of|how (do|to) (i|you) make|give me|show me|"
    r"i want|can you (make|give|find)|search for|please|the|a|an|for)\b",
    re.IGNORECASE,
)

def extract_dish_keyword(query: str) -> str:
    cleaned = STRIP_PHRASES_RE.sub(" ", query)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned if cleaned else query.strip()

# =====================================================
# VISION  (Gemini multimodal cake detection)
# =====================================================

def detect_cake_type(image_bytes: bytes, mime_type: str = "image/jpeg") -> str:
    prompt = (
        "Look at this image of a cake or dessert. "
        "Identify the specific type/name of the cake (e.g. 'Chocolate "
        "Fudge Cake', 'Red Velvet Cake', 'Carrot Cake', 'Cheesecake'). "
        "Reply with ONLY the cake name, nothing else. No punctuation, "
        "no explanation."
    )
    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[
            prompt,
            types.Part.from_bytes(data=image_bytes, mime_type=mime_type),
        ],
    )
    output_text = (response.text or "").strip()
    print(f"[Gemini Vision] Detected cake type: {output_text}")
    return output_text

# =====================================================
# RAG RETRIEVAL  (hybrid: dense + keyword, then rerank)
# =====================================================

@traceable(name="hybrid_retrieve")
def hybrid_retrieve(query: str) -> dict | None:
    query_vector = embedder.encode(query, normalize_embeddings=True).tolist()

    dense_result = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=DENSE_TOP_K,
        with_payload=True
    )
    dense_hits: list[ScoredPoint] = dense_result.points

    keyword_hits_raw = []
    for field in ["recipe_name", "recipe_text"]:
        try:
            scroll_result = qdrant.scroll(
                collection_name=COLLECTION_NAME,
                scroll_filter=Filter(
                    must=[FieldCondition(key=field, match=MatchText(text=query))]
                ),
                limit=SPARSE_TOP_K,
                with_payload=True,
                with_vectors=False,
            )
            keyword_hits_raw.extend(scroll_result[0])
        except Exception as e:
            print(f"[Keyword Search] field={field} error: {e}")

    seen: dict[str, object] = {}
    for hit in dense_hits:
        seen[str(hit.id)] = hit
    for point in keyword_hits_raw:
        if str(point.id) not in seen:
            seen[str(point.id)] = point

    candidates = list(seen.values())
    if not candidates:
        return None

    pairs  = [(query, c.payload.get("recipe_text", "")) for c in candidates]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    top_hit = ranked[0][1]

    print(f"[Hybrid Retrieve] candidates={len(candidates)}")
    print(f"[Hybrid Retrieve] top recipe = {top_hit.payload.get('recipe_name')}")

    return top_hit.payload

# =====================================================
# SERPAPI IMAGE SEARCH
# =====================================================

@traceable(name="search_recipe_image")
def search_recipe_image(recipe_name: str, max_attempts: int = 2) -> list[str]:
    api_key = os.getenv("SERPAPI_KEY")
    print(f"[SerpAPI] Key present: {bool(api_key)}")
    if not api_key or not recipe_name.strip():
        return []

    params = {
        "engine":  "google_images",
        "q":       f"{recipe_name} recipe",
        "api_key": api_key,
        "num":     3,
        "safe":    "active",
    }

    # SerpAPI's google_images engine occasionally takes a while to render
    # results; a single long timeout means one slow call blocks the whole
    # generate_recipe join step. A couple of shorter attempts recovers
    # from transient slowness/hiccups faster than one long wait, since
    # ReadTimeout here usually means "still rendering," not "down."
    timeout = httpx.Timeout(connect=5.0, read=8.0, write=5.0, pool=5.0)

    for attempt in range(1, max_attempts + 1):
        try:
            print(f"[SerpAPI] Searching for: {recipe_name} (attempt {attempt}/{max_attempts})")
            response = httpx.get("https://serpapi.com/search", params=params, timeout=timeout)
            print(f"[SerpAPI] Status: {response.status_code}")
            response.raise_for_status()
            data = response.json()
            images_results = data.get("images_results", [])
            urls = [img["original"] for img in images_results[:3] if img.get("original")]
            print(f"[SerpAPI] Found {len(urls)} images for '{recipe_name}'")
            return urls
        except httpx.TimeoutException as e:
            print(f"[SerpAPI] Timeout on attempt {attempt}/{max_attempts}: {e}")
            if attempt == max_attempts:
                print("[SerpAPI] All attempts timed out — returning no images.")
                return []
            continue
        except Exception as e:
            print(f"[SerpAPI] Error type: {type(e).__name__}")
            print(f"[SerpAPI] Error detail: {e}")
            return []  # non-timeout errors (4xx/5xx, bad key, etc.) aren't worth retrying

    return []

# =====================================================
# QUANTITY SCALING  (pure python helper)
# =====================================================

NUM_RE = re.compile(r'(\d+(?:\.\d+)?)')

def scale_ingredients(ingredients: list[str], original_qty: float, target_qty: float) -> list[str]:
    if not original_qty:
        return ingredients

    ratio = target_qty / original_qty

    def replace_num(m):
        val = float(m.group(1))
        new_val = val * ratio
        if new_val == int(new_val):
            return str(int(new_val))
        return f"{new_val:.1f}"

    return [NUM_RE.sub(replace_num, ing, count=1) for ing in ingredients]

# =====================================================
# BUILD CONVERSATION HISTORY FOR GEMINI
# =====================================================

# =====================================================
# BUILD CONVERSATION HISTORY FOR GEMINI  (with summarization)
# =====================================================

# Simple in-process cache so the same "older half" of a long conversation
# isn't re-summarized on every single turn — keyed by a hash of the exact
# message slice being summarized. Bounded size to avoid unbounded growth
# across a long-running server process.
_SUMMARY_CACHE: dict[str, str] = {}
_SUMMARY_CACHE_MAX_ENTRIES = 256


def _history_slice_key(messages: list[HistoryMessage]) -> str:
    import hashlib
    raw = "\n".join(f"{m.role}:{m.text}" for m in messages)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _summarize_older_history(messages: list[HistoryMessage]) -> str:
    """
    Collapses an older slice of conversation into a short running summary
    via one Gemini call. Cached by content hash so re-summarizing the same
    older slice on every subsequent turn doesn't cost extra tokens/latency
    — only the newly-aged-out messages ever need a fresh summarization.
    """
    if not messages:
        return ""

    cache_key = _history_slice_key(messages)
    if cache_key in _SUMMARY_CACHE:
        return _SUMMARY_CACHE[cache_key]

    transcript = "\n".join(f"{m.role}: {m.text}" for m in messages)
    prompt = (
        "Summarize the following conversation between a user and CocoaGPT "
        "(a cake & chocolate recipe assistant) into a short, dense paragraph "
        "that preserves anything that might matter for answering FUTURE "
        "questions: dietary restrictions/allergies mentioned, preferences "
        "stated, recipes discussed by name, and any unresolved questions. "
        "Omit pleasantries and filler. Keep it under 150 words.\n\n"
        f"Conversation:\n{transcript}"
    )

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt],
        )
        summary = (response.text or "").strip()
    except Exception as e:
        print(f"[summarize_history] Gemini error: {e}")
        # Fail safe: fall back to a truncated raw transcript rather than
        # losing the older context entirely.
        summary = transcript[:1200]

    if len(_SUMMARY_CACHE) >= _SUMMARY_CACHE_MAX_ENTRIES:
        _SUMMARY_CACHE.clear()
    _SUMMARY_CACHE[cache_key] = summary
    return summary


def build_gemini_history(history: list[HistoryMessage]) -> list[types.Content]:
    """
    Recent turns (up to HISTORY_WINDOW*2 messages) are sent verbatim for
    accuracy. Anything older is collapsed into one summarized turn so
    long conversations stay within a bounded token budget instead of
    either being silently truncated (losing context) or growing the
    prompt unboundedly as the chat gets longer.
    """
    recent_cutoff = HISTORY_WINDOW * 2
    older   = history[:-recent_cutoff] if len(history) > recent_cutoff else []
    recent  = history[-recent_cutoff:] if len(history) > recent_cutoff else history

    gemini_history: list[types.Content] = []

    if older:
        summary = _summarize_older_history(older)
        if summary:
            gemini_history.append(
                types.Content(
                    role="user",
                    parts=[types.Part(text=(
                        "[Summary of earlier conversation, for context — "
                        f"do not repeat this back verbatim]\n{summary}"
                    ))],
                )
            )

    for msg in recent:
        role = "user" if msg.role == "user" else "model"
        gemini_history.append(
            types.Content(role=role, parts=[types.Part(text=msg.text[:800])])
        )

    return gemini_history


def build_context(recipe: dict) -> str:
    return (
        f"Recipe Name:\n  {recipe.get('recipe_name', '')}\n\n"
        f"Recipe Content:\n  {recipe.get('recipe_text', '')}"
    )

# =====================================================
# RECIPE JSON SCHEMA (shared by fresh + follow-up generation)
# =====================================================

RECIPE_JSON_SCHEMA = """{
  "recipe_name": "",
  "prep_time": "",
  "servings": "",
  "ingredients": [],
  "instructions": [],
  "notes": "",
  "image_urls": [],
  "scaled": false,
  "scale_info": ""
}"""

# =====================================================
# LANGSMITH LOGGING
# =====================================================

def log_rag_trace(query, retrieved_name, retrieved_text, answer, chat_id="", user_id=""):
    if not langsmith_client:
        return
    try:
        langsmith_client.create_run(
            project_name="CocoaGPT-RAG",
            name="rag_trace",
            run_type="chain",
            inputs={"query": query, "chat_id": chat_id, "user_id": user_id},
            outputs={
                "retrieved_recipe": retrieved_name,
                "context": (retrieved_text or "")[:1000],
                "answer": json.dumps(answer),
            },
        )
    except Exception as e:
        print(f"[LangSmith] Logging error: {e}")

# =====================================================================
# =====================================================================
#                          LANGGRAPH PIPELINE
# =====================================================================
# =====================================================================

class GraphState(TypedDict, total=False):
    # ---- inputs ----
    query:              str
    history:             list          # list[HistoryMessage]
    last_recipe:          Optional[dict]  # structured recipe from previous turn (if any)
    detected_cake_type:    Optional[str]   # set only on the /chat-image path

    # ---- routing ----
    route:               str            # "off_topic" | "followup" | "general_followup" | "fresh"
    requested_qty:        Optional[dict]
    dish_keyword:          str

    # ---- parallel branch outputs ----
    rag_payload:           Optional[dict]   # qdrant payload (recipe_name, recipe_text)
    image_urls:            list

    # ---- final ----
    recipe_json:           Optional[dict]
    answer_text:            Optional[str]


# ---------------------------------------------------
# NODE: classify — decide off_topic / followup / fresh
# ---------------------------------------------------
@traceable(name="node_classify")
def node_classify(state: GraphState) -> GraphState:
    query   = state["query"]
    history = state.get("history") or []
    last_recipe = state.get("last_recipe")

    if is_domain_query(query):
        if is_recipe_followup_query(query, history, last_recipe):
            return {"route": "followup"}
        return {
            "route":          "fresh",
            "requested_qty":  detect_requested_quantity(query),
            "dish_keyword":   extract_dish_keyword(query),
        }

    # Not a domain (cake/chocolate) query on its own. Before deflecting,
    # check whether prior conversation history gives it meaning — e.g.
    # "I'm allergic to nuts" earlier, then "does that apply to what I
    # asked before". This is what fixes context memory for turns that
    # never produced a structured recipe.
    if is_general_followup_query(query, history):
        return {"route": "general_followup"}

    return {"route": "off_topic"}


# ---------------------------------------------------
# NODE: off_topic — dynamic deflection (Gemini, no tools, no RAG)
# ---------------------------------------------------
@traceable(name="node_off_topic")
def node_off_topic(state: GraphState) -> GraphState:
    gemini_history = build_gemini_history(state.get("history") or [])

    system_prompt = (
        "You are CocoaGPT, a cake & chocolate recipe assistant. "
        "The user just asked something unrelated to cakes, chocolate, baking, "
        "or recipes/ingredients. Politely tell them, in ONE short friendly "
        "sentence, that you can only help with cake/chocolate recipes and "
        "ingredients instructions. Vary your wording naturally, don't sound "
        "robotic or repeat a fixed template. Do not answer their actual question. "
        "You may reference the earlier conversation naturally if relevant."
    )
    current_message = types.Content(role="user", parts=[types.Part(text=state["query"])])
    messages = gemini_history + [current_message]

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=messages,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
        )
        text = (response.text or "").strip()
    except Exception as e:
        print(f"[off_topic] Gemini error: {e}")
        text = "Sorry, I can only help with cake and chocolate recipes and ingredient instructions!"

    return {"answer_text": text}


# ---------------------------------------------------
# NODE: retrieve_rag  (parallel branch A)
# ---------------------------------------------------
@traceable(name="node_retrieve_rag")
def node_retrieve_rag(state: GraphState) -> GraphState:
    payload = hybrid_retrieve(state["query"])
    return {"rag_payload": payload}


# ---------------------------------------------------
# NODE: search_images  (parallel branch B — runs concurrently with RAG)
# ---------------------------------------------------
@traceable(name="node_search_images")
def node_search_images(state: GraphState) -> GraphState:
    urls = search_recipe_image(state["dish_keyword"])
    return {"image_urls": urls}


# ---------------------------------------------------
# NODE: generate_recipe  (join point — runs after BOTH parallel branches)
# ---------------------------------------------------
@traceable(name="node_generate_recipe")
def node_generate_recipe(state: GraphState) -> GraphState:
    rag_payload = state.get("rag_payload")
    image_urls  = state.get("image_urls") or []

    if rag_payload is None:
        return {
            "answer_text": "No matching recipe found. Try asking for a specific cake or chocolate recipe!",
            "image_urls":  [],
        }

    requested_qty   = state.get("requested_qty")
    recipe_base_qty = None
    if requested_qty:
        recipe_base_qty = estimate_recipe_quantity(rag_payload.get("recipe_text", ""))

    context = build_context(rag_payload)

    qty_instruction = ""
    if requested_qty and recipe_base_qty:
        qty_instruction = (
            f"\nNote: the ingredients list you receive back will already be "
            f"scaled from {recipe_base_qty['value']} {recipe_base_qty['unit']} to "
            f"{requested_qty['value']} {requested_qty['unit']} by the system — "
            f"just extract the ORIGINAL ingredients as listed in the context, "
            f"set scaled=false and scale_info=\"\" (the system will override "
            f"these fields afterwards).\n"
        )

    system_prompt = f"""You are CocoaGPT, a warm and helpful cake & recipe assistant.
Use the recipe context below to answer the user's request.
{qty_instruction}
Return ONLY valid JSON in this exact schema, no markdown fences, no extra text:

{RECIPE_JSON_SCHEMA}

Leave image_urls as an empty list — it will be filled in separately.
"""

    gemini_history  = build_gemini_history(state.get("history") or [])
    current_message = types.Content(
        role="user",
        parts=[types.Part(text=f"Context:\n{context}\n\nUser query: {state['query']}")]
    )
    messages = gemini_history + [current_message]

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=messages,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
        )
        raw_text = (response.text or "").strip()
        raw_text = raw_text.replace("```json", "").replace("```", "").strip()
        recipe_dict = json.loads(raw_text)
    except Exception as e:
        print(f"[generate_recipe] Gemini/parse error: {e}")
        recipe_dict = {
            "recipe_name": rag_payload.get("recipe_name", ""),
            "prep_time": "", "servings": "",
            "ingredients": [], "instructions": [],
            "notes": "Could not fully generate the recipe — please try again.",
            "image_urls": [], "scaled": False, "scale_info": "",
        }

    # Deterministic quantity scaling — don't trust the LLM to do arithmetic.
    if requested_qty and recipe_base_qty and recipe_base_qty.get("value"):
        scale_factor = requested_qty["value"] / recipe_base_qty["value"]
        original_ings = recipe_dict.get("ingredients") or []
        recipe_dict["ingredients"] = scale_ingredients(
            original_ings, recipe_base_qty["value"], requested_qty["value"]
        )
        recipe_dict["scaled"] = True
        recipe_dict["scale_info"] = (
            f"Scaled from {recipe_base_qty['value']}{recipe_base_qty['unit']} "
            f"to {requested_qty['value']}{requested_qty['unit']} "
            f"(×{scale_factor:.2f})"
        )
    elif requested_qty and not recipe_base_qty:
        # Couldn't determine a base quantity to scale from — leave the
        # LLM's own best-effort ingredients as-is and flag it in notes.
        recipe_dict["scaled"] = False
        recipe_dict["scale_info"] = ""
        existing_notes = recipe_dict.get("notes") or ""
        recipe_dict["notes"] = (
            existing_notes
            + (" " if existing_notes else "")
            + "Note: couldn't determine the original recipe quantity to scale precisely — "
              "amounts above are best-effort."
        ).strip()

    recipe_dict["image_urls"] = list(dict.fromkeys(image_urls))

    return {"recipe_json": recipe_dict, "image_urls": recipe_dict["image_urls"]}


@traceable(name="node_general_followup")
def node_general_followup(state: GraphState) -> GraphState:
    """
    Handles conversational continuations that AREN'T about a specific
    recipe already retrieved — e.g. the user mentioned a dietary
    restriction, a preference, or asked a clarifying question that
    only makes sense in light of earlier turns. Uses full history,
    no RAG, no image search, no structured recipe required.
    """
    gemini_history = build_gemini_history(state.get("history") or [])

    system_prompt = """You are CocoaGPT, a warm and helpful cake & chocolate
recipe assistant. The user's current message continues the earlier
conversation (it references something said before, asks a clarifying
question, or follows up on context) but isn't itself a direct recipe
request. Use the conversation history to understand what they mean and
answer naturally and helpfully in plain text (NOT JSON). Keep it concise.
If you genuinely cannot tell what they're referring to from the history,
say so plainly and ask them to clarify — don't guess wildly."""

    current_message = types.Content(role="user", parts=[types.Part(text=state["query"])])
    messages = gemini_history + [current_message]

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=messages,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
        )
        text = (response.text or "").strip()
    except Exception as e:
        print(f"[general_followup] Gemini error: {e}")
        text = "Sorry, I had trouble answering that — could you try rephrasing?"

    return {"answer_text": text}


# ---------------------------------------------------
# NODE: followup_answer — reuse last recipe + history, no RAG / no image search
# ---------------------------------------------------
@traceable(name="node_followup_answer")
def node_followup_answer(state: GraphState) -> GraphState:
    last_recipe = state.get("last_recipe") or {}

    system_prompt = f"""You are CocoaGPT, a warm and helpful cake & recipe assistant.
The user is asking a FOLLOW-UP question about a recipe you already gave them
in this conversation. Use the conversation history and the previous recipe
below to answer naturally and conversationally — like a baker giving advice.

Previous recipe you gave:
{json.dumps(last_recipe, ensure_ascii=False)}

Answer the user's follow-up question directly and helpfully in plain text
(NOT JSON). Keep it concise — a few sentences is usually enough. If the
question implies a change (different egg count, sugar amount, etc.), explain
the effect on the recipe and give a clear recommendation.
"""

    gemini_history  = build_gemini_history(state.get("history") or [])
    current_message = types.Content(role="user", parts=[types.Part(text=state["query"])])
    messages = gemini_history + [current_message]

    try:
        response = gemini_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=messages,
            config=types.GenerateContentConfig(system_instruction=system_prompt),
        )
        text = (response.text or "").strip()
    except Exception as e:
        print(f"[followup_answer] Gemini error: {e}")
        text = "Sorry, I had trouble answering that — could you try rephrasing?"

    return {"answer_text": text}


# =====================================================
# BUILD THE GRAPH
# =====================================================

def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("classify",          node_classify)
    graph.add_node("off_topic",         node_off_topic)
    graph.add_node("followup_answer",   node_followup_answer)
    graph.add_node("general_followup",  node_general_followup)
    graph.add_node("retrieve_rag",      node_retrieve_rag)
    graph.add_node("search_images",     node_search_images)
    graph.add_node("generate_recipe",   node_generate_recipe)

    graph.add_edge(START, "classify")

    def route_after_classify_fanout(state: GraphState):
        route = state["route"]
        if route == "off_topic":
            return "off_topic"
        if route == "followup":
            return "followup_answer"
        if route == "general_followup":
            return "general_followup"
        # route == "fresh" -> fan out to BOTH parallel branches at once.
        # Returning a list of node names tells LangGraph to schedule all
        # of them as parallel branches in the same superstep; they each
        # run concurrently and both feed into generate_recipe at the join.
        return ["retrieve_rag", "search_images"]

    graph.add_conditional_edges(
        "classify",
        route_after_classify_fanout,
        {
            "off_topic":        "off_topic",
            "followup_answer":  "followup_answer",
            "general_followup": "general_followup",
            "retrieve_rag":     "retrieve_rag",
            "search_images":    "search_images",
        },
    )

    # Join: generate_recipe waits for BOTH retrieve_rag and search_images
    graph.add_edge("retrieve_rag",  "generate_recipe")
    graph.add_edge("search_images", "generate_recipe")

    graph.add_edge("off_topic",         END)
    graph.add_edge("followup_answer",   END)
    graph.add_edge("general_followup",  END)
    graph.add_edge("generate_recipe",   END)

    return graph.compile()


COCOA_GRAPH = build_graph()

# =====================================================
# =====================================================
#            MONGODB — USERS / CHATS / MESSAGES / IMAGES
# =====================================================
# =====================================================
# These endpoints replace the old direct-from-frontend Firestore calls.
# The React client now talks to Mongo exclusively through this REST layer
# instead of holding its own DB SDK/credentials.

@app.post("/api/users/ensure")
def api_ensure_user(request: EnsureUserRequest):
    """Upserts the user doc. Call once after Firebase Auth login succeeds."""
    return ensure_user(request.uid, email=request.email, display_name=request.display_name)


@app.post("/api/chats")
def api_create_chat(request: CreateChatRequest):
    return create_chat(request.user_id, title=request.title)


@app.get("/api/chats/{user_id}")
def api_get_user_chats(user_id: str):
    return get_user_chats(user_id)


@app.delete("/api/chats/{chat_id}")
def api_delete_chat(chat_id: str):
    """Cascades: deletes the chat, its messages, and any GridFS images they reference."""
    return delete_chat(chat_id)


@app.patch("/api/chats/{chat_id}/title")
def api_update_chat_title(chat_id: str, request: UpdateChatTitleRequest):
    updated = update_chat_title(chat_id, request.title)
    return {"updated": updated}


@app.get("/api/chats/{chat_id}/messages")
def api_get_chat_messages(chat_id: str):
    return get_chat_messages(chat_id)


@app.get("/api/chats/{chat_id}/last_recipe")
def api_get_last_recipe(chat_id: str):
    return {"recipe": get_last_recipe_message(chat_id)}


@app.post("/api/chats/{chat_id}/messages")
def api_save_message(chat_id: str, request: SaveMessageRequest):
    """Saves a plain text/recipe message (no file attached)."""
    return save_message(
        chat_id=chat_id,
        user_id=request.user_id,
        role=request.role,
        text=request.text,
        recipe=request.recipe,
        image_urls=request.image_urls,
        is_file=request.is_file,
        file_name=request.file_name,
        detected_cake_type=request.detected_cake_type,
    )


@app.post("/api/chats/{chat_id}/messages/file")
async def api_save_file_message(
    chat_id:  str,
    user_id:  str        = Form(...),
    text:     str        = Form(default=""),
    file:     UploadFile = File(...),
):
    """
    Stores the uploaded file's bytes in GridFS, then saves a message that
    references it via image_refs. Works for images and other attachments
    alike — GridFS doesn't care about content type, only the frontend's
    `isImage` check (based on content_type) decides how to render it.
    """
    file_bytes = await file.read()
    content_type = file.content_type or "application/octet-stream"

    file_id = save_image(
        file_bytes,
        filename=file.filename or "upload",
        content_type=content_type,
        user_id=user_id,
        chat_id=chat_id,
    )

    msg = save_message(
        chat_id=chat_id,
        user_id=user_id,
        role="user",
        text=text,
        image_refs=[file_id],
        is_file=True,
        file_name=file.filename or "upload",
    )
    msg["file_id"]      = file_id
    msg["is_image"]     = content_type.startswith("image/")
    msg["file_url"]     = f"/images/{file_id}"
    return msg


@app.get("/images/{file_id}")
def api_get_image(file_id: str):
    """Streams a GridFS-stored image/file back by id."""
    result = get_image(file_id)
    if result is None:
        return Response(status_code=404)
    data, content_type = result
    return Response(content=data, media_type=content_type)


# =====================================================
# /chat  ENDPOINT
# =====================================================

@app.post("/chat")
async def chat(request: ChatRequest):
    query   = request.query
    history = request.history

    last_recipe = request.last_recipe

    initial_state: GraphState = {
        "query":        query,
        "history":      history,
        "last_recipe":  last_recipe,
    }

    result = COCOA_GRAPH.invoke(initial_state)

    recipe_json = result.get("recipe_json")
    answer_text = result.get("answer_text")
    image_urls  = result.get("image_urls") or []

    if recipe_json:
        log_rag_trace(
            query=query,
            retrieved_name=recipe_json.get("recipe_name", ""),
            retrieved_text=(result.get("rag_payload") or {}).get("recipe_text", ""),
            answer=recipe_json,
        )
        return {"recipe": recipe_json, "image_urls": image_urls}

    return {"answer": answer_text or "Sorry, something went wrong.", "image_urls": []}

# =====================================================
# /chat-image  ENDPOINT
# =====================================================

@app.post("/chat-image")
async def chat_image(
    file:        UploadFile = File(...),
    query:       str        = Form(default=""),
    history:     str        = Form(default="[]"),
    last_recipe: str        = Form(default="null"),
):
    image_bytes = await file.read()
    mime_type   = file.content_type or "image/jpeg"

    try:
        history_data   = json.loads(history)
        parsed_history = [HistoryMessage(**m) for m in history_data]
    except Exception:
        parsed_history = []

    try:
        parsed_last_recipe = json.loads(last_recipe)
    except Exception:
        parsed_last_recipe = None

    try:
        detected_name = detect_cake_type(image_bytes, mime_type=mime_type)
    except Exception as e:
        print(f"[Gemini Vision] Error: {e}")
        return {"answer": "Could not analyze the image. Please try again.", "image_urls": []}

    effective_query = detected_name
    if query.strip():
        effective_query = f"{detected_name} {query.strip()}"

    initial_state: GraphState = {
        "query":             effective_query,
        "history":           parsed_history,
        "last_recipe":       parsed_last_recipe,
        "detected_cake_type": detected_name,
    }

    result = COCOA_GRAPH.invoke(initial_state)

    recipe_json = result.get("recipe_json")
    answer_text = result.get("answer_text")
    image_urls  = result.get("image_urls") or []

    if recipe_json is None and result.get("route") != "off_topic":
        return {
            "answer": f"Detected '{detected_name}', but no matching recipe found.",
            "detected_cake_type": detected_name,
            "image_urls": [],
        }

    log_rag_trace(
        query=effective_query,
        retrieved_name=(recipe_json or {}).get("recipe_name", ""),
        retrieved_text=(result.get("rag_payload") or {}).get("recipe_text", ""),
        answer=recipe_json or {"answer_text": answer_text},
    )

    if recipe_json:
        return {
            "recipe":             recipe_json,
            "image_urls":         image_urls,
            "detected_cake_type": detected_name,
        }

    return {
        "answer":             answer_text,
        "image_urls":          [],
        "detected_cake_type":  detected_name,
    }

# =====================================================
# DEBUG / HEALTH
# =====================================================

@app.get("/debug")
def debug():
    result = qdrant.scroll(collection_name=COLLECTION_NAME, limit=3, with_payload=True)
    return [p.payload for p in result[0]]

@app.get("/")
def root():
    return {"status": "running", "service": "CocoaGPT RAG API"}