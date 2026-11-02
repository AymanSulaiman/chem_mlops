# Chat app: Bun server + Python tool bridge (LanceDB, RDKit).
# Build from the repo root:  docker build -f deploy/web.Dockerfile .
FROM oven/bun:1.3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /usr/local/bin/uv
ENV UV_PYTHON_INSTALL_DIR=/opt/python \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    # tools.ts calls `uv run`; without this it would try to sync the train/dev groups.
    UV_NO_SYNC=1

WORKDIR /app

# Serving deps only (see pyproject.toml). The tools run as `python -m app...`
# from /app, so the project itself doesn't need installing.
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --frozen --no-default-groups --no-install-project

COPY web/package.json web/bun.lock web/
RUN cd web && bun install --frozen-lockfile --production

COPY app app
COPY web web

# server.ts rebuilds public/frontend.js on boot, so the runtime user must own public/.
RUN chown -R bun:bun web/public
USER bun

WORKDIR /app/web
ENV PORT=3000 \
    LANCEDB_DIR=/mnt/lancedb
EXPOSE 3000
CMD ["bun", "run", "server.ts"]
