# Churn prediction API - the containerized FastAPI inference service deployed to Kubernetes.
#
#   python serving/export_model.py --alias champion   # 1. export the registered model (+ verified lineage)
#   docker build -t churn-api:1.0.0 .                 # 2. build from the repository root
#
# (python start.py / scripts/deploy_serving.sh do both, then load the image into kind and deploy it.)
# Only serving/app.py, serving/requirements.txt and serving/model/ enter the build context (.dockerignore).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 MODEL_DIR=/app/model
WORKDIR /app

COPY serving/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# model exported from the MLflow registry by serving/export_model.py
COPY serving/model/ /app/model/
COPY serving/app.py .

RUN useradd --uid 10001 --no-create-home appuser
USER 10001

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health/live')" || exit 1
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
