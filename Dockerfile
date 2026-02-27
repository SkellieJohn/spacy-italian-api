FROM python:3.11-slim

# Cache buster: 2026-02-27-v2
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy Italian model
RUN python -m spacy download it_core_news_lg

# Copy application
COPY main.py .

# Expose port (Railway provides PORT env var)
EXPOSE 8000

# Run the application - explicit shell for variable expansion
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
