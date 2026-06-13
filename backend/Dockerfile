<<<<<<< HEAD
FROM python:3.10-slim
=======
FROM python:3.11-slim
>>>>>>> master

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
<<<<<<< HEAD
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
=======
    gcc g++ \
    libglib2.0-0 \
    libsm6 libxext6 libxrender1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir --default-timeout=1000 --retries=5 \
       --extra-index-url https://download.pytorch.org/whl/cpu \
       -r requirements.txt

COPY . .

RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
>>>>>>> master
