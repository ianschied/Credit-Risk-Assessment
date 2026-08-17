# Serves the FastAPI credit-risk API.
#
# Build (after training a model locally -- models/lightgbm.joblib and
# models/feature_cols.joblib must exist, since they're copied in below):
#     docker build -t credit-risk-api .
#
# Run:
#     docker run -p 8000:8000 credit-risk-api
#
# Test:
#     curl http://localhost:8000/health

FROM python:3.11-slim

WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install deps first for better layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code + trained model artifacts
COPY src/ src/
COPY models/ models/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
