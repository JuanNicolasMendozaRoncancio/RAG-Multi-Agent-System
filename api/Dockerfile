FROM python:3.11-slim

WORKDIR /app

# Install all dependencies the API actually needs at runtime
COPY api/requirements.txt ./api_requirements.txt
COPY nlp_worker/requirements.txt ./nlp_requirements.txt
COPY agent_worker/requirements.txt ./agent_requirements.txt

RUN pip install --no-cache-dir \
    -r api_requirements.txt \
    -r nlp_requirements.txt \
    -r agent_requirements.txt

# Download spaCy models (needed by nlp_worker/pipeline.py)
RUN python -m spacy download es_core_news_md && \
    python -m spacy download en_core_web_md && \
    python -m spacy download fr_core_news_md

# Copy all modules the API imports
COPY shared/ ./shared/
COPY api/ ./api/
COPY nlp_worker/ ./nlp_worker/
COPY agent_worker/ ./agent_worker/
COPY models/ ./models/

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8080"]