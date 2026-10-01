#!/usr/bin/env bash
# Start FastAPI and Streamlit behind the Render-facing Nginx proxy.
set -Eeuo pipefail

export PORT="${PORT:-10000}"
export MAX_UPLOAD_SIZE_MB="${MAX_UPLOAD_SIZE_MB:-25}"
api_key="${INSIGHT_API_KEY:-}"

if [[ ${#api_key} -lt 32 ]]; then
    echo "INSIGHT_API_KEY must contain at least 32 characters." >&2
    exit 1
fi

mkdir -p /app/data/uploads /app/data/models /run/nginx
export STREAMLIT_SERVER_MAXUPLOADSIZE="$MAX_UPLOAD_SIZE_MB"
envsubst '${PORT} ${MAX_UPLOAD_SIZE_MB}' \
    < /etc/nginx/templates/insight-engine.conf.template \
    > /etc/nginx/conf.d/default.conf

python -m uvicorn app.api.server:app --host 127.0.0.1 --port 8000 &
api_pid=$!
python -m streamlit run app/main.py \
    --server.address 127.0.0.1 \
    --server.port 8501 \
    --server.headless true &
streamlit_pid=$!

shutdown() {
    kill "$api_pid" "$streamlit_pid" "${nginx_pid:-}" 2>/dev/null || true
}
trap shutdown EXIT INT TERM

for _ in $(seq 1 60); do
    if curl --fail --silent http://127.0.0.1:8000/health >/dev/null \
        && curl --fail --silent http://127.0.0.1:8501/_stcore/health >/dev/null; then
        break
    fi
    if ! kill -0 "$api_pid" 2>/dev/null || ! kill -0 "$streamlit_pid" 2>/dev/null; then
        echo "FastAPI or Streamlit exited before becoming healthy." >&2
        exit 1
    fi
    sleep 1
done

if ! curl --fail --silent http://127.0.0.1:8000/health >/dev/null \
    || ! curl --fail --silent http://127.0.0.1:8501/_stcore/health >/dev/null; then
    echo "FastAPI or Streamlit did not become healthy within 60 seconds." >&2
    exit 1
fi

nginx -g 'daemon off;' &
nginx_pid=$!
set +e
wait -n "$api_pid" "$streamlit_pid" "$nginx_pid"
exit_code=$?
set -e
exit "$exit_code"
