FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    ffmpeg \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir \
    whisperx \
    faster-whisper[gpu] \
    huggingface_hub

WORKDIR /audio

# Download large-v3 model at build time (requires HF_TOKEN build arg for faster downloads)
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}
RUN python3 -c "\
import os; \
os.environ.setdefault('HF_TOKEN', '$HF_TOKEN'); \
from huggingface_hub import hf_hub_download, list_repo_files; \
import os; \
dst = '/opt/whisper-models/large-v3'; \
os.makedirs(dst, exist_ok=True); \
for f in list_repo_files('Systran/faster-whisper-large-v3'): \
    if f.endswith(('.bin', '.json')): \
        path = hf_hub_download('Systran/faster-whisper-large-v3', f); \
        os.system(f'cp {{path}} {dst}/'); \
        print(f'  {f}'); \
print('Model downloaded.')"

ENV WHISPER_MODEL_DIR=/opt/whisper-models

ENTRYPOINT ["python3", "-c"]
CMD ["--help"]
