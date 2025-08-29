# Local AI Restaurant Review Agent

A lightweight Python application that uses local language models to answer questions about restaurant reviews. The system uses Ollama for embeddings and inference, with Chroma as a vector store for efficient retrieval.

## Project Structure

```
.
├── requirements.txt      # Python dependencies
├── vector.py            # Vector store and retriever implementation
├── main.py             # Interactive QA interface
├── data/               # Directory for CSV data (not included)
└── .gitignore         # Git ignore patterns
```

## Prerequisites

- Python 3.8+
- Ollama installed locally with models:
  - mxbai-embed-large (for embeddings)
  - llama3.2 (for inference)
- Restaurant reviews dataset in CSV format

## Dependencies

Install required packages:
```bash
pip install -r requirements.txt
```

Dependencies list:
- langchain
- langchain-ollama
- langchain-chroma
- pandas

## Dataset Format

Place your CSV file at `data/realistic_restaurant_reviews.csv` with these columns:
- Title
- Review
- Rating
- Date

## Setup & Usage

1. **Prepare the Vector Store**
   ```bash
   python vector.py
   ```
   This creates a Chroma database at `./chrome_langchain_db/`

2. **Run the Interactive QA**
   ```bash
   python main.py
   ```
   - Type your questions at the prompt
   - Enter 'q' to quit

## Performance Features

The implementation includes several optimizations:
- Lazy loading of the vector store
- Query result caching (100 most recent queries)
- Review text truncation (max 1500 chars per prompt)
- Background model warming
- Default k=3 for retrieval to reduce latency
- Multiple API fallbacks for compatibility

## Implementation Details

### Vector Store (vector.py)
- Uses Chroma for document storage
- Combines review title and content
- Stores rating and date as metadata
- Supports lazy loading and rebuild options

### QA Interface (main.py)
- Interactive command-line interface
- Caches query results
- Handles multiple retriever APIs
- Includes model warming
- Graceful error handling

## Error Handling

The system handles common issues:
- Missing vector store (auto-rebuilds if needed)
- API compatibility issues (multiple fallbacks)
- Model errors (graceful error messages)
- Import-time failures (lazy loading)

## Development

Files ignored by git:
- `chrome_langchain_db/` (vector store)
- `data/*.csv` (datasets)
- Python cache files
- Virtual environments
- IDE configs

## Troubleshooting

- **Missing Database**: Run `vector.py` first to create the vector store
- **Slow Responses**: Ensure Ollama models are downloaded locally
- **Import Errors**: Verify all dependencies are installed
- **API Errors**: Check LangChain package versions match

## Notes

- The vector store is persisted locally and reused between sessions
- All processing happens locally through Ollama
- The system is optimized for repeated queries

