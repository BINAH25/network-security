# ----------- Stage 1: Build Layer -----------
    FROM python:3.10-slim as builder

    WORKDIR /app
    
    # Install build tools for scientific packages
    RUN apt-get update && apt-get install -y \
        build-essential \
        gcc \
        g++ \
        libffi-dev \
        libssl-dev \
        python3-dev \
        git \
        curl
    
    # Pre-install dependencies into a custom directory
    COPY requirements.txt ./
    RUN pip install --upgrade pip setuptools wheel \
        && pip install --prefix=/install --no-cache-dir -r requirements.txt
    
    # ----------- Stage 2: Runtime Layer -----------
    FROM python:3.10-slim
    
    WORKDIR /app
    
    # Minimal runtime deps
    RUN apt-get update && apt-get install -y \
        libffi7 \
        libssl1.1 \
        && apt-get clean \
        && rm -rf /var/lib/apt/lists/*
    
    # Copy installed Python packages
    COPY --from=builder /install /usr/local
    
    # Copy app source
    COPY . .
    
    EXPOSE 8000
    CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
    