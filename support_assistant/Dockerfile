FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/app/.hf_cache \
    MOCK_LLM=1

WORKDIR /app

# CPU-only torch keeps the image ~2 GB smaller than the default CUDA wheel.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY docs/ ./docs/
COPY ingest.py prompts.py schemas.py graph.py main.py demo.py ./

# Bake the embedding model and the ChromaDB index into the image so the
# container serves /ask without any network access at runtime.
RUN python ingest.py
# Model is now cached in the image: never contact the Hugging Face Hub at runtime.
ENV HF_HUB_OFFLINE=1

# Hugging Face Spaces runs containers as a non-root user (uid 1000).
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
