# Reproducible environment for local, GitHub Codespaces, and CI.
# Includes build-essential so `tmu` (C/CUDA clause kernels) compiles.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential git make \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
COPY . .
RUN pip install --no-cache-dir -e ".[dev]" && \
    (pip install --no-cache-dir tmu || echo "tmu install skipped - run 'pip install tmu' in the container")

CMD ["bash"]
