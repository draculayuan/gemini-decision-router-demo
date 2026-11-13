# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

FROM python:3.12-slim

RUN pip install --no-cache-dir "uv>=0.12.2"

WORKDIR /code

COPY ./pyproject.toml ./README.md ./LICENSE ./uv.lock ./
COPY ./app ./app

RUN uv sync --frozen --no-dev

EXPOSE 8080

CMD ["uv", "run", "--no-dev", "uvicorn", "app.demo_server:demo_app", "--host", "0.0.0.0", "--port", "8080"]
