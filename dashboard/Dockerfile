FROM python:3.11-slim
 
WORKDIR /app
 
COPY dashboard/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
 
COPY shared/ ./shared/
COPY dashboard/ ./dashboard/
 
CMD ["streamlit", "run", "dashboard/app.py", "--server.port=8501", "--server.address=0.0.0.0"]