import uuid
import fitz

from sentence_transformers import SentenceTransformer

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct
)

# ==========================================
# CONFIG
# ==========================================

PDF_PATH = "data/Bakers-Choice-Recipe-Book.pdf"

COLLECTION_NAME = "recipes"

# ==========================================
# EMBEDDING MODEL
# ==========================================

print("Loading BGE-M3...")

embed_model = SentenceTransformer(
    "BAAI/bge-m3"
)

VECTOR_SIZE = embed_model.get_sentence_embedding_dimension()

print(
    f"Embedding Dimension = {VECTOR_SIZE}"
)

# ==========================================
# QDRANT
# ==========================================

client = QdrantClient(
    host="localhost",
    port=6333
)

if client.collection_exists(
    COLLECTION_NAME
):
    client.delete_collection(
        COLLECTION_NAME
    )

client.create_collection(
    collection_name=COLLECTION_NAME,
    vectors_config=VectorParams(
        size=VECTOR_SIZE,
        distance=Distance.COSINE
    )
)

# ==========================================
# OPEN PDF
# ==========================================

pdf = fitz.open(PDF_PATH)

# ==========================================
# HELPER
# ==========================================

def is_recipe_title(text):

    text = text.strip()

    if len(text) < 5:
        return False

    words = text.split()

    upper_words = [
        w for w in words
        if w.isupper()
    ]

    return len(upper_words) >= 2


# ==========================================
# FIND RECIPES
# ==========================================

recipes = []

page = 0

while page < len(pdf):

    page_text = pdf[page].get_text()

    lines = [
        x.strip()
        for x in page_text.split("\n")
        if x.strip()
    ]

    title = None

    for line in lines[:5]:

        if is_recipe_title(line):

            title = line
            break

    if title:

        content_page = page + 1

        recipes.append(
            {
                "title": title,
                "title_page": page,
                "content_page": content_page
            }
        )

        page += 2

    else:
        page += 1

print(
    f"Recipes Found: {len(recipes)}"
)

# ==========================================
# INGEST
# ==========================================

points = []

for recipe in recipes:

    title_page = recipe["title_page"]

    content_page = recipe["content_page"]

    title_text = pdf[title_page].get_text()

    content_text = ""

    if content_page < len(pdf):
        content_text = pdf[
            content_page
        ].get_text()

    full_text = (
        title_text +
        "\n\n" +
        content_text
    )

    # ======================================
    # EMBEDDING
    # ======================================

    embedding = embed_model.encode(
        full_text,
        normalize_embeddings=True
    )

    payload = {
        "recipe_name": recipe["title"],
        "recipe_text": full_text
    }

    points.append(
        PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding.tolist(),
            payload=payload
        )
    )

# ==========================================
# UPSERT
# ==========================================

client.upsert(
    collection_name=COLLECTION_NAME,
    points=points
)

print(
    f"Inserted {len(points)} recipes"
)

print("Done")