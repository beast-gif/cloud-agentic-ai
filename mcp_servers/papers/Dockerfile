# Agentic Cloud - Phase 3, Step 3.2
# MCP server "notify" (mock provider). Build context = ./mcp_servers.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1
WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY common/ ./common/
COPY notify/server.py ./server.py

RUN useradd --create-home --uid 10003 mcp
USER mcp

EXPOSE 9000
CMD ["python", "server.py"]