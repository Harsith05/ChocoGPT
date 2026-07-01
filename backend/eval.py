"""
eval.py — CocoaGPT RAG Evaluation with LangSmith
=================================================
Project : CocoaGPT
LLM Judge: Gemini 2.5 Flash

What this script does
---------------------
1. DATASET LOADING
   - Reads langsmith_eval_dataset.json (your hand-curated file)
   - Pushes it to LangSmith on first run; skips push on subsequent runs
     (checks for an existing dataset by name before creating)
   - Each example carries: query, expected recipe name, ground-truth
     answer (full ingredients + instructions text)

2. PIPELINE UNDER TEST
   - Calls your live CocoaGPT /chat endpoint for each query
   - Captures: retrieved_recipe, ingredients list, instructions list,
     full answer JSON, image_urls

3. EVALUATORS  (Gemini 2.5 Flash as LLM judge where needed)
   - retrieval_relevance  : Did we retrieve the right recipe?
   - groundedness         : Is the answer grounded in the ground-truth answer?
   - answer_relevance     : Does the answer address what the user asked?
   - hallucination        : Does the answer invent facts not in ground truth?
   - completeness         : Does the answer have name + ingredients + steps?
   - ingredient_coverage  : Do returned ingredients match the expected ones?

Run
---
    python eval.py

Requirements
------------
    pip install langsmith python-dotenv httpx
    (google-genai already installed for the main app)
"""

import os
import re
import json
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

# =====================================================
# CONFIG
# =====================================================

DATASET_JSON_PATH = "langsmith_eval_dataset.json"   # your hand-curated file

LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY")
GEMINI_API_KEY    = os.getenv("GEMINI_API_KEY")

PROJECT_NAME  = "CocoaGPT-RAG"
DATASET_NAME  = "CocoaGPT-Eval-Dataset"             # name shown in LangSmith UI
BACKEND_URL   = "http://localhost:8000/chat"

# =====================================================
# LANGSMITH CLIENT
# =====================================================

os.environ["LANGCHAIN_API_KEY"]    = LANGSMITH_API_KEY or ""
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"]    = PROJECT_NAME

from langsmith import Client
from langsmith.evaluation import evaluate

langsmith_client = Client(api_key=LANGSMITH_API_KEY)

# =====================================================
# GEMINI CLIENT  (LLM judge — no query generation needed)
# =====================================================

from google import genai
from google.genai import types as gtypes

gemini = genai.Client(api_key=GEMINI_API_KEY)


def gemini_call(prompt: str, max_tokens: int = 512, retries: int = 5) -> str:
    """Gemini wrapper with exponential back-off for rate-limit (429) errors."""
    delay = 15
    for attempt in range(retries):
        try:
            response = gemini.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt,
                config=gtypes.GenerateContentConfig(
                    max_output_tokens=max_tokens,
                ),
            )
            return (response.text or "").strip()
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                wait = delay * (2 ** attempt)
                print(f"  [RateLimit] Quota hit — waiting {wait}s "
                      f"(attempt {attempt + 1}/{retries})...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError("Gemini rate limit: all retries exhausted")


def gemini_score(prompt: str) -> tuple[float, str]:
    """
    Calls Gemini with a scoring prompt.
    Expected response format: {"score": 0.9, "reason": "..."}
    Returns (score, reason).
    """
    try:
        raw = gemini_call(prompt, max_tokens=256)
        raw = raw.replace("```json", "").replace("```", "").strip()
        # Extract first {...} block in case Gemini adds surrounding text
        match = re.search(r'\{.*?\}', raw, re.DOTALL)
        if match:
            raw = match.group(0)
        data   = json.loads(raw)
        score  = float(data.get("score", 0.0))
        reason = data.get("reason", "")
        return score, reason
    except Exception as e:
        return 0.0, f"Parse error: {e}"


# =====================================================
# STEP 1 — LOAD JSON DATASET
# =====================================================

def load_json_dataset(path: str) -> list[dict]:
    """
    Reads langsmith_eval_dataset.json and returns a flat list of examples.
    Each item: { query, expected_recipe, expected_answer, source }
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    examples = []
    for ex in raw.get("examples", []):
        inp  = ex.get("inputs", {})
        out  = ex.get("outputs", {})
        meta = ex.get("metadata", {})

        examples.append({
            "query":           inp.get("query", ""),
            "expected_recipe": out.get("recipe_name", meta.get("recipe", "")),
            "expected_answer": out.get("answer", ""),
            "source":          meta.get("source", ""),
        })

    print(f"[Dataset] Loaded {len(examples)} examples from '{path}'")
    return examples


# =====================================================
# STEP 2 — PUSH DATASET TO LANGSMITH  (first run only)
# =====================================================

def push_dataset_to_langsmith(examples: list[dict]) -> str:
    """
    Pushes examples to LangSmith as a reusable dataset.
    Skips creation if the dataset already exists.
    Returns the dataset name.
    """
    existing = list(langsmith_client.list_datasets(dataset_name=DATASET_NAME))
    if existing:
        print(f"[Dataset] '{DATASET_NAME}' already exists in LangSmith — skipping push.")
        print(f"          Delete it in the UI to re-push from the JSON file.")
        return DATASET_NAME

    print(f"[Dataset] Creating '{DATASET_NAME}' in LangSmith "
          f"({len(examples)} examples)...")

    dataset = langsmith_client.create_dataset(
        dataset_name=DATASET_NAME,
        description=(
            "Hand-curated evaluation dataset for CocoaGPT. "
            "Loaded from langsmith_eval_dataset.json. "
            "Each example pairs a user query with the expected recipe name "
            "and full ground-truth answer (ingredients + instructions)."
        ),
    )

    for i, ex in enumerate(examples, 1):
        langsmith_client.create_example(
            inputs={
                "query": ex["query"],
            },
            outputs={
                "expected_recipe": ex["expected_recipe"],
                "expected_answer": ex["expected_answer"],
            },
            dataset_id=dataset.id,
        )
        if i % 10 == 0 or i == len(examples):
            print(f"  Pushed {i}/{len(examples)} examples...")

    print(f"[Dataset] Done — '{DATASET_NAME}' is live in LangSmith.")
    return DATASET_NAME


# =====================================================
# STEP 3 — PIPELINE UNDER TEST
# Calls live CocoaGPT /chat endpoint
# =====================================================

def run_cocoagpt(inputs: dict) -> dict:
    """
    Called by LangSmith's evaluate() for each example.
    Hits the live /chat endpoint and returns structured output.
    """
    query = inputs.get("query", "")
    try:
        resp = httpx.post(
            BACKEND_URL,
            json={"query": query},
            timeout=90.0,
        )
        resp.raise_for_status()
        data = resp.json()

        recipe       = data.get("recipe") or {}
        recipe_name  = recipe.get("recipe_name", "")
        ingredients  = recipe.get("ingredients", [])
        instructions = recipe.get("instructions", [])
        image_urls   = data.get("image_urls", [])

        return {
            "retrieved_recipe": recipe_name,
            "ingredients":      ingredients,
            "instructions":     instructions,
            "answer":           json.dumps(recipe),   # full JSON for LLM judge
            "image_urls":       image_urls,
        }

    except Exception as e:
        print(f"[Pipeline] Error for query '{query}': {e}")
        return {
            "retrieved_recipe": "",
            "ingredients":      [],
            "instructions":     [],
            "answer":           "",
            "image_urls":       [],
            "error":            str(e),
        }


# =====================================================
# STEP 4 — EVALUATORS
# =====================================================

# ---- 1. Retrieval Relevance ----

def eval_retrieval_relevance(run, example) -> dict:
    """
    Did the system retrieve the correct recipe for the query?
    Semantic match between retrieved_recipe and expected_recipe.
    """
    retrieved = (run.outputs or {}).get("retrieved_recipe", "")
    expected  = (example.outputs or {}).get("expected_recipe", "")
    query     = (example.inputs  or {}).get("query", "")

    prompt = f"""
You are evaluating a recipe retrieval system called CocoaGPT.

User query      : {query}
Expected recipe : {expected}
Retrieved recipe: {retrieved}

Score how relevant the retrieved recipe is to what was expected.
- 1.0 = exact or very close match (same recipe, minor wording differences)
- 0.5 = partially relevant (same category but different variant)
- 0.0 = completely wrong recipe

Return ONLY valid JSON: {{"score": <0.0-1.0>, "reason": "<one sentence>"}}
"""
    score, reason = gemini_score(prompt)
    return {"key": "retrieval_relevance", "score": score, "comment": reason}


# ---- 2. Groundedness ----

def eval_groundedness(run, example) -> dict:
    """
    Is the generated answer grounded in the ground-truth answer?
    """
    answer          = (run.outputs  or {}).get("answer", "")
    expected_answer = (example.outputs or {}).get("expected_answer", "")

    prompt = f"""
You are evaluating groundedness for CocoaGPT, a recipe assistant.

GROUND TRUTH ANSWER (from the Baker's Choice recipe book):
{expected_answer[:1200]}

SYSTEM'S GENERATED ANSWER (JSON):
{answer[:800]}

Score groundedness: how well is the system's answer supported by the ground truth?
- 1.0 = all key facts, ingredients, and steps trace back to the ground truth
- 0.5 = mostly grounded but some unverifiable details
- 0.0 = contains facts not present in the ground truth

Return ONLY valid JSON: {{"score": <0.0-1.0>, "reason": "<one sentence>"}}
"""
    score, reason = gemini_score(prompt)
    return {"key": "groundedness", "score": score, "comment": reason}


# ---- 3. Answer Relevance ----

def eval_answer_relevance(run, example) -> dict:
    """Does the answer actually address what the user asked?"""
    query  = (example.inputs or {}).get("query", "")
    answer = (run.outputs or {}).get("answer", "")

    prompt = f"""
You are evaluating CocoaGPT, a recipe chatbot.

User query          : {query}
System answer (JSON): {answer[:800]}

Score how well the answer addresses the user's query.
- 1.0 = directly answers with a complete, on-topic recipe
- 0.5 = partially answers (missing elements like servings or steps)
- 0.0 = does not answer the query at all

Return ONLY valid JSON: {{"score": <0.0-1.0>, "reason": "<one sentence>"}}
"""
    score, reason = gemini_score(prompt)
    return {"key": "answer_relevance", "score": score, "comment": reason}


# ---- 4. Hallucination ----

def eval_hallucination(run, example) -> dict:
    """
    Does the answer contain information NOT in the ground-truth answer?
    Score: 1.0 = no hallucination, 0.0 = hallucination detected.
    """
    answer          = (run.outputs  or {}).get("answer", "")
    expected_answer = (example.outputs or {}).get("expected_answer", "")

    prompt = f"""
You are a hallucination detector for CocoaGPT, a recipe assistant.

GROUND TRUTH (from the Baker's Choice recipe book):
{expected_answer[:1200]}

SYSTEM ANSWER:
{answer[:800]}

Does the system answer contain ingredients, steps, or facts that are NOT
present in the ground truth and cannot reasonably be inferred from it?

- 1.0 = no hallucination (everything traceable to the ground truth)
- 0.5 = minor additions that are reasonable cooking inferences
- 0.0 = clear hallucination (invented ingredients, wrong temperatures, etc.)

Return ONLY valid JSON: {{"score": <0.0-1.0>, "reason": "<one sentence>"}}
"""
    score, reason = gemini_score(prompt)
    return {"key": "hallucination", "score": score, "comment": reason}


# ---- 5. Completeness  (structural, no LLM needed) ----

def eval_completeness(run, example) -> dict:
    """
    Does the answer include all essential recipe sections?
    Checks for: recipe_name present, ingredients (>=3 items), instructions (>=2 steps).
    """
    recipe_name  = (run.outputs or {}).get("retrieved_recipe", "")
    ingredients  = (run.outputs or {}).get("ingredients", [])
    instructions = (run.outputs or {}).get("instructions", [])

    has_name  = bool(recipe_name)
    has_ings  = len(ingredients) >= 3
    has_steps = len(instructions) >= 2

    score  = sum([has_name, has_ings, has_steps]) / 3.0
    reason = (
        f"name={'v' if has_name else 'x'} | "
        f"ingredients={len(ingredients)} ({'v' if has_ings else 'x'}) | "
        f"instructions={len(instructions)} ({'v' if has_steps else 'x'})"
    )
    return {"key": "completeness", "score": score, "comment": reason}


# ---- 6. Ingredient Coverage  (deterministic keyword overlap) ----

def eval_ingredient_coverage(run, example) -> dict:
    """
    Do the returned ingredients mention the expected key ones?
    Parses ingredient lines from expected_answer and does a keyword-overlap
    check against the system's ingredients list. No LLM needed.
    """
    expected_answer = (example.outputs or {}).get("expected_answer", "")
    sys_ingredients = (run.outputs or {}).get("ingredients", [])

    # Parse ingredient lines from the ground-truth answer text
    expected_lines: list[str] = []
    in_ingredients = False
    for line in expected_answer.split("\n"):
        stripped = line.strip()
        if stripped.lower().startswith("ingredient"):
            in_ingredients = True
            continue
        if in_ingredients:
            if stripped.startswith("-"):
                expected_lines.append(stripped.lstrip("- ").lower())
            elif stripped == "" or stripped.lower().startswith("instruction"):
                break

    if not expected_lines:
        return {
            "key":     "ingredient_coverage",
            "score":   1.0,
            "comment": "No expected ingredients parsed — skipped",
        }

    sys_text = " ".join(sys_ingredients).lower()

    # Match by first two meaningful words of each expected ingredient
    # e.g. "all-purpose flour", "cocoa powder", "baking soda"
    hits = 0
    for ing in expected_lines:
        key = " ".join(ing.split()[:2])
        if key and key in sys_text:
            hits += 1

    score  = round(hits / len(expected_lines), 3)
    reason = f"{hits}/{len(expected_lines)} expected ingredients matched in system response"
    return {"key": "ingredient_coverage", "score": score, "comment": reason}


# =====================================================
# STEP 5 — RUN EVALUATION
# =====================================================

def run_evaluation():
    print("\n" + "=" * 58)
    print("  CocoaGPT RAG Evaluation — LangSmith + Gemini Judge")
    print("=" * 58)

    # 1. Load JSON dataset
    examples = load_json_dataset(DATASET_JSON_PATH)
    if not examples:
        print(f"[ERROR] No examples found in '{DATASET_JSON_PATH}'.")
        return

    # 2. Push to LangSmith (skipped if dataset already exists)
    dataset_name = push_dataset_to_langsmith(examples)

    # 3. Run evaluation
    print(f"\n[Eval] Running evaluation on '{dataset_name}'...")
    print(f"[Eval] Backend : {BACKEND_URL}")
    print(f"[Eval] Examples: {len(examples)}")
    print(f"[Eval] Judge   : gemini-2.5-flash\n")

    results = evaluate(
        run_cocoagpt,
        data=dataset_name,
        evaluators=[
            eval_retrieval_relevance,
            eval_groundedness,
            eval_answer_relevance,
            eval_hallucination,
            eval_completeness,
            eval_ingredient_coverage,
        ],
        experiment_prefix="cocoagpt-eval",
        metadata={
            "project":  "cocoagpt",
            "judge":    "gemini-2.5-flash",
            "embedder": "BAAI/bge-m3",
            "reranker": "BAAI/bge-reranker-v2-m3",
            "dataset":  DATASET_JSON_PATH,
        },
        max_concurrency=1,   # sequential — avoids Gemini rate limits
    )

    # 4. Summary
    print("\n" + "=" * 58)
    print("  Evaluation Complete")
    print("=" * 58)
    print(results)
    print(f"\nFull results at:")
    print(f"   https://smith.langchain.com/projects/{PROJECT_NAME}")
    print(f"\n   Dataset : {dataset_name}")
    print(f"   Project : {PROJECT_NAME}")


if __name__ == "__main__":
    assert LANGSMITH_API_KEY, "LANGSMITH_API_KEY not set in .env"
    assert GEMINI_API_KEY,    "GEMINI_API_KEY not set in .env"
    assert os.path.exists(DATASET_JSON_PATH), \
        f"Dataset JSON not found: {DATASET_JSON_PATH}"

    run_evaluation()