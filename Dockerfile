# F5-TTS (русский файнтюн) на RunPod serverless. Модель качается на cold start (репо в ENV F5_REPO),
# референсы (муж/жен) зашиты в образ. torch<2.9 — чтобы не тянуть torchcodec.
FROM python:3.11-slim

ENV HF_HOME=/root/.cache/huggingface \
    PYTHONUNBUFFERED=1 \
    NUMBA_CACHE_DIR=/tmp/numba

RUN apt-get update && apt-get install -y --no-install-recommends \
        git ffmpeg libsndfile1 build-essential && \
    rm -rf /var/lib/apt/lists/*

# torch<2.9 (CUDA-сборка с PyPI на Linux), ставим первым — f5-tts сам torch корректно не закрепляет
RUN pip install --no-cache-dir "torch>=2.6,<2.9" "torchaudio>=2.6,<2.9"

# F5-TTS + ударения + раннер
RUN pip install --no-cache-dir runpod f5-tts ruaccent soundfile huggingface_hub

# smoke-import: ловим проблемы зависимостей на сборке, а не в рантайме
RUN python -c "import f5_tts; from f5_tts.api import F5TTS; print('f5 import ok')"

COPY refs /refs
COPY handler.py /handler.py
CMD ["python", "-u", "/handler.py"]
