FROM python:3.12-slim AS python-ml
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY python/ python/
COPY shared/models shared/models/
ENV PYTHONPATH=/app/python

FROM debian:bookworm-slim AS cpp-runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake g++ curl && rm -rf /var/lib/apt/lists/*
COPY cpp/ /cpp/
RUN rm -rf /cpp/build && cmake -S /cpp -B /cpp/build && cmake --build /cpp/build --config Release

FROM python:3.12-slim
WORKDIR /app
COPY --from=python-ml /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages
COPY --from=python-ml /app /app
COPY --from=cpp-runtime /cpp/build/schedulerd /usr/local/bin/schedulerd
COPY configs/ configs/
ENV PYTHONPATH=/app/python
CMD ["python", "python/training/train.py"]
