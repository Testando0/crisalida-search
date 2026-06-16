FROM python:3.11-slim

WORKDIR /app

# Install deps first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Render injects $PORT at runtime; default to 10000
ENV PORT=10000

CMD ["python", "main.py"]
