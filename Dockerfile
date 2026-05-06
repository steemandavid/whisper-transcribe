# Multi-stage build: builder pre-bakes models, runtime is lean
# Build: docker build -t whisper-transcribe:2.0 .
# Slim:  docker build --build-arg PREBAKE_ALL=0 -t whisper-transcribe:2.0-slim .

ARG BASE_IMAGE=nvidia/cuda:12.4.0-runtime-ubuntu22.04

# ---- Builder stage ----
FROM ${BASE_IMAGE} AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-dev \
    ffmpeg git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Pin all dependencies
RUN pip3 install --no-cache-dir \
    torch==2.5.1+cu124 \
    torchaudio==2.5.1 \
    -f https://download.pytorch.org/whl/torch_stable.html || \
    pip3 install --no-cache-dir torch==2.5.1 torchaudio==2.5.1

# cuDNN for ctranslate2/pyannote (not included in runtime image)
RUN pip3 install --no-cache-dir nvidia-cudnn-cu12==9.* && \
    ln -sf /usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib/libcudnn*.so* /usr/local/lib/ || true

RUN pip3 install --no-cache-dir \
    ctranslate2==4.5.0 \
    faster-whisper==1.1.1 \
    whisperx==3.4.2 \
    pyannote.audio==3.4.0 \
    openai==1.54.0 \
    huggingface_hub==0.30.2 \
    soundfile \
    numpy \
    scipy \
    matplotlib \
    loguru appdirs sympy unidecode \
    progressbar2 librosa timm torchlibrosa && \
    pip3 install --no-cache-dir --no-deps audiosr deepfilternet==0.5.6

# Pre-bake models (optional, controlled by PREBAKE_ALL)
ARG PREBAKE_ALL=1
ENV PREBAKE_ALL=${PREBAKE_ALL}
ARG HF_TOKEN=""
ENV HF_TOKEN=${HF_TOKEN}

# Download large-v3 model
RUN python3 <<'PYEOF'
import os
from huggingface_hub import hf_hub_download, list_repo_files
os.makedirs('/opt/whisper-models/large-v3', exist_ok=True)
for f in list_repo_files('Systran/faster-whisper-large-v3'):
    if f.endswith(('.bin', '.json')):
        path = hf_hub_download('Systran/faster-whisper-large-v3', f)
        os.system(f'cp {path} /opt/whisper-models/large-v3/')
        print(f'  {f}')
print('large-v3 downloaded.')
PYEOF

# Convert Dutch model (pre-bake only)
RUN if [ "$PREBAKE_ALL" = "1" ]; then \
    python3 -c "from faster_whisper import WhisperModel; print('Downloading Dutch whisper-large-v3-dutch for conversion...')" && \
    ct2-transformers-converter \
        --model golesheed/whisper-large-v3-dutch \
        --output_dir /opt/whisper-models/nl-large-v3 \
        --quantization float16 \
        --copy_files tokenizer.json preprocessor_config.json || \
    echo "Dutch model conversion skipped (may need HF auth)"; \
    fi

# Pre-download alignment models
RUN python3 <<'PYEOF'
import os
if os.environ.get('PREBAKE_ALL', '1') != '1':
    print('Skipping alignment model download (PREBAKE_ALL=0)')
    raise SystemExit(0)
from whisperx import load_align_model
for lang, name in [('nl', 'jonatasgrosman/wav2vec2-large-xlsr-53-dutch'), ('en', 'WAV2VEC2_ASR_BASE_960H')]:
    try:
        print(f'Downloading alignment model for {lang}...')
        load_align_model(language_code=lang, model_name=name, device='cpu')
    except Exception as e:
        print(f'  skipped {lang}: {e}')
PYEOF

# ---- Runtime stage ----
FROM ${BASE_IMAGE} AS runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/opt
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/dist-packages/nvidia/cublas/lib:/usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib:/usr/local/lib/x86_64-linux-gnu:/usr/local/lib

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip \
    ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# Pin runtime deps (no build tools)
RUN pip3 install --no-cache-dir \
    torch==2.5.1+cu124 \
    torchaudio==2.5.1 \
    -f https://download.pytorch.org/whl/torch_stable.html || \
    pip3 install --no-cache-dir torch==2.5.1 torchaudio==2.5.1

# cuDNN for ctranslate2/pyannote
RUN pip3 install --no-cache-dir nvidia-cudnn-cu12==9.* && \
    ln -sf /usr/local/lib/python3.10/dist-packages/nvidia/cudnn/lib/libcudnn*.so* /usr/local/lib/ || true

RUN pip3 install --no-cache-dir \
    ctranslate2==4.5.0 \
    faster-whisper==1.1.1 \
    whisperx==3.4.2 \
    pyannote.audio==3.4.0 \
    openai==1.54.0 \
    huggingface_hub==0.30.2 \
    soundfile \
    numpy \
    scipy \
    matplotlib \
    loguru appdirs sympy unidecode \
    progressbar2 librosa timm torchlibrosa && \
    pip3 install --no-cache-dir --no-deps audiosr deepfilternet==0.5.6

# Copy pre-baked models from builder
COPY --from=builder /opt/whisper-models /opt/whisper-models

# Copy pipeline code
COPY pipeline/ /opt/pipeline/

ENV WHISPER_MODEL_DIR=/opt/whisper-models

# Smoke test
RUN python3 -c "import faster_whisper, whisperx, torch, pyannote.audio, openai; print('imports ok')"

WORKDIR /audio

ENTRYPOINT ["python3", "-m"]
CMD ["pipeline.glossary", "--dump"]
