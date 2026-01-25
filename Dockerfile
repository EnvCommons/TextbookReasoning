FROM ubuntu:22.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update && apt upgrade -y && apt install -y \
    software-properties-common \
    ca-certificates \
    python3 \
    python3-pip \
    curl \
    git \
    && apt clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"
RUN uv venv --python 3.11

# Copy application files
COPY . /app

# Install dependencies
RUN uv pip install -r /app/requirements.txt

# Expose port
EXPOSE 8080

# Start server
CMD ["uv", "run", "python", "/app/server.py"]
