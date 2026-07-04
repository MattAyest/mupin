FROM python:3.11-slim

WORKDIR /app

# Install dependencies needed for our Docker-in-Docker functionality
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    gnupg \
    ca-certificates \
    && curl -fsSL https://download.docker.com/linux/debian/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg \
    && . /etc/os-release \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/debian $VERSION_CODENAME stable" > /etc/apt/sources.list.d/docker.list \
    && apt-get update && apt-get install -y --no-install-recommends docker-ce-cli \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY llm_config.yaml .
COPY profiles/ ./profiles/
COPY src/ ./src/

# Make sure we have a directory for workspaces
RUN mkdir -p .workspaces

# Default command: dev API proxy. Override to "python -m src.worker" for ARQ worker.
CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8000"]
