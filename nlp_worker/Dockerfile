FROM python:3.11-slim
 
WORKDIR /app
 
COPY nlp_worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
RUN python -m spacy download es_core_news_md && \
    python -m spacy download en_core_web_md && \
    python -m spacy download fr_core_news_md
 
COPY shared/ ./shared/
COPY nlp_worker/ ./nlp_worker/
 
CMD ["python", "-m", "nlp_worker.pipeline"]
 