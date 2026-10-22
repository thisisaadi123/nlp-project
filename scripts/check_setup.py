"""Step 0 checks. Run on both machines: python scripts/check_setup.py

Checks tools, config, API keys, Hugging Face gated-model access and (if torch is
installed) the GPU. Each check prints PASS / WARN / FAIL with how to fix it.
"""
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.settings import config, secret  # noqa: E402

GATED_MODELS = [
    "pyannote/speaker-diarization-3.1",
    "pyannote/segmentation-3.0",
    "ai4bharat/indic-conformer-600m-multilingual",
    "ai4bharat/indictrans2-indic-indic-dist-320M",
    "ai4bharat/indic-parler-tts",
]

results = []


def report(status: str, name: str, detail: str = "") -> None:
    results.append(status)
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))


def check_python() -> None:
    v = sys.version_info
    if (v.major, v.minor) == (3, 11):
        report("PASS", "Python", f"{v.major}.{v.minor}.{v.micro}")
    else:
        report("WARN", "Python", f"{v.major}.{v.minor} found, project uses 3.11")


def check_ffmpeg() -> None:
    if not shutil.which("ffmpeg"):
        report("FAIL", "ffmpeg", "not found (mac: brew install ffmpeg, wsl: sudo apt install ffmpeg)")
        return
    first = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[0]
    report("PASS", "ffmpeg", first)


def check_config() -> None:
    cfg = config()
    langs = ", ".join(l["name"] for l in cfg["languages"].values())
    total = sum(cfg["scoring"]["weights"].values())
    if total != 100:
        report("FAIL", "config.yaml", f"scoring weights sum to {total}, expected 100")
    else:
        report("PASS", "config.yaml", f"languages: {langs}")


def check_groq() -> None:
    key = secret("GROQ_API_KEY")
    if not key:
        report("FAIL", "Groq", "GROQ_API_KEY missing in .env (create one at console.groq.com)")
        return
    from groq import Groq

    client = Groq(api_key=key)
    try:
        live = sorted(m.id for m in client.models.list().data)
    except Exception as e:
        report("FAIL", "Groq", f"key rejected or network error: {e}")
        return

    llm = config()["llm"]
    for role in ("text_model", "vision_model"):
        model = llm[role]
        if model in live:
            report("PASS", f"Groq {role}", model)
        else:
            report("FAIL", f"Groq {role}", f"'{model}' is not offered any more; update config.yaml")

    if llm["text_model"] in live:
        reply = client.chat.completions.create(
            model=llm["text_model"],
            messages=[{"role": "user", "content": "Translate 'bhaiya' into Tamil. Reply with one word."}],
            max_tokens=10,
        )
        report("PASS", "Groq test call", reply.choices[0].message.content.strip())

    print("        Live Groq models:", ", ".join(live))


def check_huggingface() -> None:
    token = secret("HF_TOKEN")
    if not token:
        report("FAIL", "Hugging Face", "HF_TOKEN missing in .env (huggingface.co/settings/tokens)")
        return
    from huggingface_hub import auth_check, whoami
    from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

    try:
        user = whoami(token=token)["name"]
    except Exception as e:
        report("FAIL", "Hugging Face", f"token rejected: {e}")
        return
    report("PASS", "Hugging Face", f"logged in as {user}")

    for repo in GATED_MODELS:
        try:
            auth_check(repo, token=token)
            report("PASS", f"HF access {repo}")
        except GatedRepoError:
            report("FAIL", f"HF access {repo}", f"accept the terms at https://huggingface.co/{repo}")
        except RepositoryNotFoundError:
            report("WARN", f"HF access {repo}", "repo not found; the model id may have changed")


def check_gemini() -> None:
    key = secret("GEMINI_API_KEY")
    if not key:
        report("WARN", "Gemini (backup)", "GEMINI_API_KEY missing; optional (aistudio.google.com)")
        return
    import httpx

    r = httpx.get(
        "https://generativelanguage.googleapis.com/v1beta/models",
        headers={"x-goog-api-key": key},
        timeout=15,
    )
    if r.status_code == 200:
        report("PASS", "Gemini (backup)", f"{len(r.json().get('models', []))} models visible")
    else:
        report("FAIL", "Gemini (backup)", f"HTTP {r.status_code}")


def check_gpu() -> None:
    try:
        import torch
    except ImportError:
        report("WARN", "GPU", "torch not installed (expected on the Mac; required on the Omen)")
        return
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        report("PASS", "GPU", f"{name}, {vram:.1f} GB VRAM, torch {torch.__version__}")
    elif torch.backends.mps.is_available():
        report("WARN", "GPU", "Apple MPS only; heavy models run on the Omen")
    else:
        report("FAIL", "GPU", "torch installed but CUDA not available (check nvidia-smi in WSL)")


if __name__ == "__main__":
    for check in (check_python, check_ffmpeg, check_config, check_groq,
                  check_huggingface, check_gemini, check_gpu):
        try:
            check()
        except Exception as e:
            report("FAIL", check.__name__, f"unexpected error: {e}")
    fails = results.count("FAIL")
    print(f"\n{results.count('PASS')} passed, {results.count('WARN')} warnings, {fails} failed")
    sys.exit(1 if fails else 0)
