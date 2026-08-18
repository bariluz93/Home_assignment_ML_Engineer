# Runs the Streamlit NER-comparison dashboard (app.py) with no local setup
# beyond a container engine (Docker/Podman).
#
#   docker build -t dnrti-ner .
#   docker run -p 8501:8501 dnrti-ner
#
# then open http://localhost:8501

FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# libarchive-tools ships bsdtar, which (unlike plain GNU tar) can read RAR
# archives - needed once below to unpack the DNRTI.rar dataset at build time,
# and by ensure_dnrti_dataset() at runtime if the dataset is ever re-downloaded.
# curl backs the container HEALTHCHECK.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libarchive-tools \
        curl \
    && ln -sf /usr/bin/bsdtar /usr/bin/tar \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# This project only ever runs on CPU (see README), so pull the CPU-only
# torch/torchvision wheels - a fraction of the size of the default CUDA build.
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt

COPY . .

# Bake the DNRTI dataset into the image (same download+extract
# ensure_dnrti_dataset() does at runtime) so the container works out of the
# box even without the repo's local, gitignored data/ directory.
RUN python -c "from src.compare_models import ensure_dnrti_dataset; ensure_dnrti_dataset('data/train.txt')"

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501"]
