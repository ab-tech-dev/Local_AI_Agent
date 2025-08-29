from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain.schema import Document
import os
import pandas as pd

def build_vector_store(
    csv_path="data/realistic_restaurant_reviews.csv",
    db_location="./chrome_langchain_db",
    collection_name="restaurant_reviews",
    embed_model="mxbai-embed-large",
):
    df = pd.read_csv(csv_path)

    embeddings = OllamaEmbeddings(model=embed_model)

    add_documents = not os.path.exists(db_location)

    vector_store = Chroma(
        collection_name=collection_name,
        persist_directory=db_location,
        embedding_function=embeddings,
    )

    if add_documents:
        documents = []
        ids = []

        for i, row in df.iterrows():
            text = f"{row['Title']} {row['Review']}"
            document = Document(
                page_content=text,
                metadata={"rating": row.get("Rating"), "date": row.get("Date")},
            )

            ids.append(str(i))
            documents.append(document)

        vector_store.add_documents(documents, ids=ids)

        # Persist if the Chroma wrapper exposes a persist method (safe call)
        if hasattr(vector_store, "persist"):
            try:
                vector_store.persist()
            except Exception:
                # best-effort persist; ignore errors to keep script usable
                pass

    return vector_store

def get_retriever(
    csv_path="data/realistic_restaurant_reviews.csv",
    db_location="./chrome_langchain_db",
    collection_name="restaurant_reviews",
    embed_model="mxbai-embed-large",
    k=3,
    rebuild_if_missing=False,
):
    """
    Lazily load (or build) the Chroma vector store and return a retriever.
    - If the database exists, load it (fast).
    - If missing and rebuild_if_missing=True, build it from CSV.
    - Default k reduced to 3 to lower retrieval / prompt size.
    """
    db_exists = os.path.exists(db_location)
    if not db_exists and not rebuild_if_missing:
        # If DB missing and we don't rebuild automatically, raise a clear error
        raise FileNotFoundError(
            f"Chroma DB not found at {db_location}. Run build_vector_store or set rebuild_if_missing=True."
        )

    vector_store = build_vector_store(csv_path, db_location, collection_name, embed_model)
    retriever = vector_store.as_retriever(search_kwargs={"k": k})
    return retriever

if __name__ == "__main__":
    # Build DB if missing when executed directly
    if not os.path.exists("./chrome_langchain_db"):
        print("No Chroma DB found. Building from CSV (this may take time)...")
        build_vector_store()
        print("Build complete.")
    else:
        print("Chroma DB exists. Nothing to do.")
