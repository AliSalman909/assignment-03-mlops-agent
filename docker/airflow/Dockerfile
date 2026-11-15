FROM apache/airflow:3.4.0

# Training dependencies for the DAG tasks. Airflow is pinned so pip can't silently upgrade it.
RUN pip install --no-cache-dir \
    "apache-airflow==3.4.0" \
    "mlflow-skinny==3.18.0" \
    "scikit-learn==1.9.1" \
    "pandas>=2.1" \
    "joblib>=1.3" \
    "skops==0.16.0"

# pre-create volume mount points so named volumes inherit airflow-user ownership
RUN mkdir -p /opt/airflow/db /opt/airflow/work

COPY --chown=airflow:root docker/airflow/entrypoint.sh /opt/airflow/entrypoint.sh
ENTRYPOINT ["/usr/bin/dumb-init", "--", "bash", "/opt/airflow/entrypoint.sh"]
