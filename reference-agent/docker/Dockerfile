FROM python:3.11-slim

WORKDIR /app

COPY pyproject.toml ./
COPY src ./src

RUN pip install --no-cache-dir fastapi uvicorn openai

EXPOSE 8000

CMD ["sh", "-c", "uvicorn reference_agent.service.app:app --host 0.0.0.0 --port 8000"]
