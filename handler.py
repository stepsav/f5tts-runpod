"""RunPod serverless handler для F5-TTS (русский файнтюн) — озвучка реплик мультика.

F5 клонирует тембр из РЕФЕРЕНС-клипа. Зашиты мужской и женский эталон (refs/),
пользователь выбирает голос строкой speaker: "male" / "female".

input:
  text       — реплика (рус.)
  speaker    — "male" | "female" (по умолчанию male)
  accentize  — ставить ли ударения через RUAccent ('+' перед ударным). По умолчанию False
               (RU-файнтюн обычно сам неплохо ставит; включим, если будет криво).
output: {audio_base64 (wav 24k), format:"wav", speaker, gen_seconds}

ENV (можно менять на эндпоинте без пересборки кода):
  F5_REPO        — HF-репозиторий русского файнтюна (по умолчанию ниже)
  F5_MODEL_NAME  — имя конфигурации арки ("F5TTS_Base" / "F5TTS_v1_Base")
"""
import os
import re
import glob
import base64
import time
import uuid

os.environ.setdefault("HF_HOME", "/root/.cache/huggingface")

import soundfile as sf
import torch

F5_REPO = os.environ.get("F5_REPO", "Misha24-10/F5-TTS_RUSSIAN")
F5_MODEL_NAME = os.environ.get("F5_MODEL_NAME", "F5TTS_Base")
MODEL_DIR = os.environ.get("F5_MODEL_DIR", "/models/f5ru")
REFS_DIR = os.environ.get("F5_REFS_DIR", "/refs")

_device = "cuda" if torch.cuda.is_available() else "cpu"

# Референсы: speaker -> (wav, точная транскрипция)
REFS = {
    "male": (os.path.join(REFS_DIR, "male.wav"),
             "Сегодня прекрасный день, и я с радостью расскажу вам одну удивительную историю про настоящих друзей."),
    "female": (os.path.join(REFS_DIR, "female.wav"),
               "Привет! Я очень рада тебя видеть. Давай вместе отправимся в это волшебное приключение прямо сейчас."),
}

RU_VOWELS = "аеёиоуыэюяАЕЁИОУЫЭЮЯ"


def _ensure_model():
    """Скачать русский файнтюн, если не зашит/не скачан. Возвращает (ckpt_path, vocab_path)."""
    if not os.path.isdir(MODEL_DIR) or not os.listdir(MODEL_DIR):
        from huggingface_hub import snapshot_download
        print(f"[f5] downloading {F5_REPO} -> {MODEL_DIR}", flush=True)
        snapshot_download(repo_id=F5_REPO, local_dir=MODEL_DIR)
    ckpts = (glob.glob(os.path.join(MODEL_DIR, "**", "*.safetensors"), recursive=True)
             + glob.glob(os.path.join(MODEL_DIR, "**", "*.pt"), recursive=True))
    if not ckpts:
        raise RuntimeError(f"no checkpoint (.safetensors/.pt) in {MODEL_DIR}")
    # предпочитаем 'last'/самый большой
    ckpt = max(ckpts, key=lambda p: (("last" in p.lower()) * 10_000_000_000 + os.path.getsize(p)))
    vocabs = glob.glob(os.path.join(MODEL_DIR, "**", "vocab.txt"), recursive=True)
    vocab = vocabs[0] if vocabs else ""
    print(f"[f5] ckpt={ckpt}\n[f5] vocab={vocab}", flush=True)
    return ckpt, vocab


def _make_f5(ckpt, vocab):
    from f5_tts.api import F5TTS
    # API менялся: новый параметр model=, старый model_type=
    try:
        return F5TTS(model=F5_MODEL_NAME, ckpt_file=ckpt, vocab_file=vocab, device=_device)
    except TypeError:
        return F5TTS(model_type="F5-TTS", ckpt_file=ckpt, vocab_file=vocab, device=_device)


print("[f5] loading model...", flush=True)
_ckpt, _vocab = _ensure_model()
_f5 = _make_f5(_ckpt, _vocab)
print("[f5] ready on", _device, flush=True)

_acc = None


def _accentize(text: str) -> str:
    global _acc
    try:
        if _acc is None:
            from ruaccent import RUAccent
            _acc = RUAccent()
            _acc.load(omograph_model_size="turbo3", use_dictionary=True)
        return _acc.process_all(text)  # '+' перед ударным гласным
    except Exception as e:
        print("[f5] accent fail:", repr(e), flush=True)
        return text


def handler(event):
    inp = event.get("input", {}) or {}
    text = (inp.get("text") or "").strip()
    if not text:
        return {"error": "empty text"}
    speaker = (inp.get("speaker") or "male").lower()
    if speaker not in REFS:
        speaker = "male"
    if inp.get("accentize", False):
        text = _accentize(text)
    ref_file, ref_text = REFS[speaker]

    t0 = time.time()
    out_path = f"/tmp/{uuid.uuid4().hex}.wav"
    try:
        wav, sr, _ = _f5.infer(ref_file=ref_file, ref_text=ref_text, gen_text=text,
                               remove_silence=True, file_wave=out_path)
    except Exception as e:
        return {"error": f"f5 infer failed: {e}", "speaker": speaker}
    if not os.path.exists(out_path):
        sf.write(out_path, wav, sr)
    with open(out_path, "rb") as f:
        data = f.read()
    try:
        os.remove(out_path)
    except OSError:
        pass
    return {
        "audio_base64": base64.b64encode(data).decode("utf-8"),
        "format": "wav",
        "speaker": speaker,
        "gen_seconds": round(time.time() - t0, 1),
    }


if __name__ == "__main__":
    import runpod
    runpod.serverless.start({"handler": handler})
