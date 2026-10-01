FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DATABASE_URL=sqlite:///./data/insight_engine.db

RUN apt-get update \
    && apt-get install -y --no-install-recommends nginx gettext-base curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt
COPY . .
COPY deploy/nginx.conf.template /etc/nginx/templates/insight-engine.conf.template
RUN mkdir -p /app/data/uploads /app/data/models /run/nginx

EXPOSE 10000

CMD ["bash", "deploy/start.sh"]
