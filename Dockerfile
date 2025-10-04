FROM python:3.11-slim

ENV PIP_NO_CACHE_DIR=1 PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
# Cloud Run will send traffic here
ENV PORT=8080
CMD ["python","-m","streamlit","run","app.py","--server.port=8080","--server.address=0.0.0.0","--server.headless=true"]
