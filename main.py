from langchain_ollama import OllamaLLM
from vector import get_retriever
import threading
import time

model = OllamaLLM(model="llama3.2")

template = """
You're an expert in answering questions about a pizza restaurant.

Here are some relevant reviews:
{reviews}

Here is the question to answer: {question}
"""

# Lazy-load retriever once with a smaller k to reduce retrieval cost
try:
    retriever = get_retriever(k=3, rebuild_if_missing=False)
except FileNotFoundError:
    # If DB missing, build it (may take time) then load
    retriever = get_retriever(k=3, rebuild_if_missing=True)

# Simple in-memory exact-match cache (keeps insertion order)
_query_cache = {}
_CACHE_MAX = 100


def _cache_put(key, value):
    _query_cache[key] = value
    if len(_query_cache) > _CACHE_MAX:
        # pop oldest item
        _query_cache.pop(next(iter(_query_cache)))


def get_relevant_documents_for_query(query):
    # Exact-match cache
    if query in _query_cache:
        return _query_cache[query]

    # Prefer invoke API if available
    docs = None
    if hasattr(retriever, "invoke"):
        try:
            docs = retriever.invoke(query)
        except Exception:
            docs = None

    if not docs:
        for method in ("get_relevant_documents", "retrieve", "get_relevant_items"):
            if hasattr(retriever, method):
                try:
                    docs = getattr(retriever, method)(query)
                except Exception:
                    docs = None
                break

    if docs is None:
        try:
            docs = retriever(query)
        except Exception:
            docs = []

    _cache_put(query, docs)
    return docs


def _truncate_reviews(docs, max_chars=1500):
    """
    Concatenate page_content from docs but limit total characters to max_chars.
    Keep whole documents and stop when limit would be exceeded.
    """
    if not docs:
        return ""
    parts = []
    total = 0
    for d in docs:
        text = getattr(d, "page_content", str(d))
        if total + len(text) + 2 > max_chars:
            break
        parts.append(text)
        total += len(text) + 2
    return "\n\n".join(parts) if parts else (getattr(docs[0], "page_content", str(docs[0]))[:max_chars])


# Warm the model in a background thread to reduce first-request latency
def _warm_model():
    try:
        # small, harmless prompt
        if hasattr(model, "invoke"):
            try:
                model.invoke("Ready?")
            except Exception:
                pass
        elif hasattr(model, "predict"):
            try:
                model.predict("Ready?")
            except Exception:
                pass
        else:
            try:
                model("Ready?")
            except Exception:
                pass
    except Exception:
        pass


_warm_thread = threading.Thread(target=_warm_model, daemon=True)
_warm_thread.start()
time.sleep(0.05)  # give warm thread a moment to start

while True:
    print("\n\n====================================")
    question = input("Ask your question (q to quit): ")
    if question.lower() == "q":
        break

    docs = get_relevant_documents_for_query(question) or []
    reviews_text = _truncate_reviews(docs, max_chars=1500) if docs else "No relevant reviews found."

    prompt_text = template.format(reviews=reviews_text, question=question)

    # Prefer invoke API on the LLM with fallbacks
    result = None
    if hasattr(model, "invoke"):
        try:
            result = model.invoke(prompt_text)
        except Exception:
            try:
                result = model.invoke({"input": prompt_text})
            except Exception as e:
                result = f"Model invoke error: {e}"
    elif hasattr(model, "predict"):
        try:
            result = model.predict(prompt_text)
        except Exception as e:
            result = f"Model predict error: {e}"
    else:
        try:
            result = model(prompt_text)
        except Exception as e:
            result = f"Model call error: {e}"

    print("\nAnswer:\n", result)