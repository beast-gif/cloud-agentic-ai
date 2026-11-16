# Agentic Cloud - Phase 1, Step 1.2
# Agent container image: research-assistant LangGraph agent served by FastAPI.
#
# Build (from the agentic-cloud/ folder):
#   docker build -t agentic-cloud/agent:phase1 ./agent_runtime

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first so this layer is cached when only code changes
COPY requirements.txt .
RUN pip install -r requirements.txt

# Agent code
COPY llm.py tools.py memory_client.py graph.py server.py ./

# Run as a non-root user
RUN useradd --create-home --uid 10001 agent

# Phase 2 (L2): workspace mount point, owned by the agent user.
# When Docker mounts a NEW empty named volume here, it copies this directory's
# ownership into the volume - so the non-root agent can write to it.
RUN mkdir -p /workspace && chown agent:agent /workspace
ENV WORKSPACE_DIR=/workspace

USER agent

EXPOSE 8080

# server.py starts uvicorn itself, so its first line runs before any heavy import
# (needed for accurate startup timing).
CMD ["python", "server.py"]