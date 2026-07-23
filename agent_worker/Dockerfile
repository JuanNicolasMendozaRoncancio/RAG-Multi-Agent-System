FROM python:3.11-slim

WORKDIR /app

COPY agent_worker/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY shared/ ./shared/
COPY agent_worker/ ./agent_worker/

CMD ["python", "-m", "agent_worker.agents"]