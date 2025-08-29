# Local AI Agent (minimal)

This repository builds a Chroma vector store of restaurant reviews using Ollama embeddings and runs an interactive prompt that queries an Ollama LLM.

Latency improvements made
- Retriever loading is now lazy: the Chroma store is not built on import. The project loads the vector store only when needed.
- Retrieval cost reduced by default (k=3) to limit the number of documents retrieved per query.
- An in-memory exact-match cache reduces repeated-query latency.
- Concatenated reviews are truncated to limit prompt size (default 1500 chars), reducing model input and inference time.
- The LLM is warmed asynchronously at startup to reduce first-request latency.

What changed
- Code now prefers the newer LangChain "invoke" API for both retriever and LLM to avoid deprecation warnings.
- Fallbacks remain in place for environments where older APIs (predict/retrieve/get_relevant_documents) are used.

How to run
1. Install dependencies:
   pip install -r requirements.txt

2. Ensure dataset exists at:
   data/realistic_restaurant_reviews.csv
   (CSV headers required: Title, Review, Rating, Date)

3. Build or load the vector DB:
   - If ./chrome_langchain_db exists, it will be loaded quickly on first use.
   - If not, the code will build it from the CSV when needed (this may take time once).

4. Run interactive prompt:
   python main.py
   - Ask questions at the prompt. Enter `q` to quit.

Notes
- The code will try retriever.invoke(query) and model.invoke(prompt) first; if those methods are not present it will fall back to prior methods (e.g., retrieve, get_relevant_documents, predict).
- Confirm that the Ollama models referenced (e.g., "mxbai-embed-large", "llama3.2") are available in your local Ollama installation.
- Do not commit model files or the ./chrome_langchain_db directory to version control.

If you want, I can make the invocation behavior configurable (force-invoke vs force-fallback) or add logging to surface which method is used at runtime.
   python main.py
   - Type questions and press Enter.
   - Enter `q` to quit.

Behavior notes (as implemented)
- Embeddings: OllamaEmbeddings(model="mxbai-embed-large")
- Chroma collection: name "restaurant_reviews", persisted to ./chrome_langchain_db
- Retriever configured with k=5
- LLM: OllamaLLM(model="llama3.2") — ensure the model identifier matches your local Ollama installation
- main.py formats retrieved reviews into the prompt before calling the model

Fixes applied to code
- vector.py: corrected variable names and imports, removed unsupported id argument from Document, and attempted safe persistence after adding documents.
- main.py: fixed prompt text typo ("expert"), used the retriever correctly (supports multiple retriever API names), formatted retrieved reviews into the prompt, and called the model via predict or direct call with error handling.

Notes and troubleshooting
- Ensure Ollama is installed and the requested models are available locally.
- If imports fail, confirm langchain and adapter package versions; adapter APIs vary across releases.
- Do not commit ./chrome_langchain_db or model files into version control.
- If Chroma persistence does not occur automatically for your adapter, inspect the Chroma wrapper in your environment and call the appropriate persist method.

If you want, I can further:
- Add a small CLI option to run only DB build or only interactive mode.
- Add basic logging and a requirements pin file for reproducibility.
- Sanitize any text before using it in system commands (this project currently does not run shell commands, but be cautious if you extend it).

