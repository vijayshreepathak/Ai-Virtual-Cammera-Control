# AI Virtual Camera Control

Real-time **AI virtual camera** proof-of-concept: control a simulated camera with **hand gestures**, then describe cinematic shots by **voice** and get structured **JSON** from an LLM (Gemini or OpenAI).

**Repository:** [github.com/vijayshreepathak/Ai-Virtual-Cammera-Control](https://github.com/vijayshreepathak/Ai-Virtual-Cammera-Control)

---

## Overview

| Layer | Technology | Purpose |
|-------|------------|---------|
| Gesture detection | MediaPipe Hand Landmarker | Pan, zoom, tilt from hand motion |
| Virtual camera | OpenCV crop/transform | Visible zoom/pan on live preview |
| Speech-to-text | Gemini / OpenAI Whisper / faster-whisper | Transcribe microphone commands |
| Cinematic LLM | Gemini / OpenAI | Structured JSON camera plan |
| UI | Gradio (dark theme) | Web demo at `http://127.0.0.1:7860` |

---

## Features

- Live webcam preview with gesture overlays (FPS, confidence, zoom/pan state)
- **5 gestures:** Pan Left, Pan Right, Zoom In, Zoom Out, Tilt Up
- **Virtual camera** — preview actually zooms/pans (not just labels)
- Hand skeleton overlay when palm is tracked
- Action history log with timestamps
- Voice recording → transcription → cinematic JSON output
- Optional fusion: last gesture sent as LLM context
- Pause webcam, restart camera, reset zoom/pan, sensitivity slider
- Smoke tests for pre-demo validation

---

## Project Structure

```text
.
├── app.py                 # Gradio UI + event wiring
├── gesture_detector.py    # MediaPipe gestures + history logic
├── camera_simulator.py    # Visible zoom/pan/tilt on frames
├── webcam.py              # Auto-detect working camera index
├── voice_to_text.py       # STT: gemini | openai | faster-whisper
├── llm_cinematic.py       # LLM JSON cinematic output
├── utils.py               # Overlays, .env, API key validation
├── smoke_test.py          # Dependency + module checks
├── run.bat                # Windows: smoke test + launch
├── requirements.txt
├── .env.example           # Template (copy to .env)
└── README.md
```

On first run, `models/hand_landmarker.task` is downloaded automatically (gitignored).

---

## Requirements

- **Python 3.9+** (3.11 recommended)
- Webcam + microphone
- Internet for cloud APIs (Gemini / OpenAI)
- Windows tested; Linux/macOS should work with the same Python commands

---

## Installation

```powershell
# 1) Clone or open project folder
cd "F:\Ai Engg Assignment"

# 2) Create virtual environment
python -m venv .venv

# 3) Activate (Windows PowerShell)
.venv\Scripts\activate

# 4) Install dependencies
pip install -r requirements.txt

# 5) Create environment file
copy .env.example .env
```

---

## Environment Setup

### Option A — Gemini only (recommended if you have a Google AI Studio key)

Edit `.env`:

```env
OPENAI_API_KEY=
OPENAI_MODEL=gpt-4o-mini
OPENAI_WHISPER_MODEL=whisper-1

GEMINI_API_KEY=AIzaSy_your_key_from_ai_studio
GEMINI_MODEL=gemini-2.0-flash

LLM_PROVIDER=gemini
STT_BACKEND=gemini
WHISPER_MODEL_SIZE=tiny
```

### Option B — OpenAI only

```env
OPENAI_API_KEY=sk-your_openai_key
OPENAI_MODEL=gpt-4o-mini
OPENAI_WHISPER_MODEL=whisper-1

GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.0-flash

LLM_PROVIDER=openai
STT_BACKEND=openai
WHISPER_MODEL_SIZE=tiny
```

### Option C — Local speech (no cloud STT)

```env
STT_BACKEND=faster-whisper
WHISPER_MODEL_SIZE=tiny
```

First launch downloads the Whisper model and may take several minutes.

---

## Environment Variables Reference

| Variable | Values | Description |
|----------|--------|-------------|
| `GEMINI_API_KEY` | `AIzaSy...` | Google AI Studio key — **required for Gemini mode** |
| `GEMINI_MODEL` | e.g. `gemini-2.0-flash` | Model for LLM + audio transcription |
| `OPENAI_API_KEY` | `sk-...` | OpenAI key for LLM and/or Whisper |
| `OPENAI_MODEL` | e.g. `gpt-4o-mini` | Chat model for cinematic JSON |
| `OPENAI_WHISPER_MODEL` | `whisper-1` | OpenAI STT model |
| `LLM_PROVIDER` | `gemini` \| `openai` | Which API generates cinematic JSON |
| `STT_BACKEND` | `gemini` \| `openai` \| `faster-whisper` | Speech-to-text backend |
| `WHISPER_MODEL_SIZE` | `tiny`, `base`, … | Local Whisper size (faster-whisper only) |

---

## Gemini API Key — Important

This app uses the **Google Generative Language API**. Keys must come from **[Google AI Studio](https://aistudio.google.com/apikey)** and start with **`AIza`**.

| Key prefix | Valid? | Notes |
|------------|--------|-------|
| `AIzaSy...` | Yes | Create at [aistudio.google.com/apikey](https://aistudio.google.com/apikey) |
| `AQ....` | **No** | Wrong type → `API_KEY_INVALID` / HttpError 400 |
| Empty | No | Status bar shows LLM/STT warnings |

**If you see `API key not valid`:**

1. Create a new key at [Google AI Studio](https://aistudio.google.com/apikey)
2. Paste into `.env` as `GEMINI_API_KEY=AIzaSy...`
3. Remove any `GEMINI_API_KEY` from **Windows Environment Variables** (can override `.env`)
4. Restart the app

At startup the terminal prints a safe key check, e.g. `GEMINI_API_KEY: AIza… (39 chars) — format OK`.

---

## Run the App

### Recommended (smoke tests first)

```powershell
.\run.bat
```

### Manual

```powershell
.venv\Scripts\python.exe smoke_test.py
.venv\Scripts\python.exe app.py
```

**Always use the venv Python** — not system `python` — to avoid missing packages.

Wait until the terminal shows:

```text
Running on local URL:  http://127.0.0.1:7860
```

Open that URL in your browser. If port 7860 is busy, Gradio uses the next free port (7861, etc.) — use the URL printed in the terminal.

Opening the browser **before** `Running on local URL` appears causes `ERR_CONNECTION_REFUSED`.

---

## UI Guide

| Section | Description |
|---------|-------------|
| **Status bar** | LLM / STT / Gesture engine readiness |
| **Live Webcam** | Video + overlays (gesture, action, confidence, FPS, zoom/pan) |
| **Performance** | Current FPS estimate |
| **Pause Webcam** | Stops processing to save CPU |
| **Restart Camera** | Re-scan camera indices if preview is black |
| **Reset Zoom/Pan** | Reset virtual camera to default |
| **Sensitivity** | Lower = easier gesture triggers (default ~0.75) |
| **Gesture Status** | Current gesture, camera action, confidence |
| **Action History** | Log of triggered gestures |
| **Voice** | Record → **Transcribe & Generate** → Transcript / Cinematic / JSON tabs |

---

## Gesture Guide

| Gesture | Hand movement | Camera action | Preview effect |
|---------|---------------|---------------|----------------|
| **Pan Right** | Slow sweep **left → right**, palm toward camera | Pan right | Image shifts right |
| **Pan Left** | Slow sweep **right → left** | Pan left | Image shifts left |
| **Zoom In** | Move open palm **toward** camera | Zoom in | Preview zooms in (`Z` bar increases) |
| **Zoom Out** | Pull hand **away** from camera | Zoom out | Preview zooms out |
| **Tilt Up** | Move hand **upward** | Tilt up | Preview shifts up |

### Tips for reliable detection

1. **One hand** in frame, **good lighting**
2. **Palm facing camera** — orange dots/lines = hand tracked
3. **One motion at a time** — do not mix pan and zoom in the same movement
4. Move **slowly** for 2–3 seconds
5. Lower **sensitivity** slider (0.5–0.7) if gestures do not trigger
6. Status **Hand Tracked** = hand seen; move more clearly to trigger a gesture

---

## Voice + LLM Pipeline

1. Select **Microphone** in the audio dropdown (not Stereo Mix)
2. Click **Record**, speak clearly for 3+ seconds, e.g.  
   *"Slow dolly in on the hero as rain starts, tense mood."*
3. Click **Transcribe & Generate** (webcam pauses briefly during processing)
4. Check **Transcript**, **Cinematic Plan**, and **JSON** tabs

Optional: enable **Use last gesture as LLM context** to combine gesture + voice.

### Example JSON output

```json
{
  "camera_movement": "dolly in",
  "emotion": "tense",
  "shot_style": "close-up",
  "reasoning": "A slow dolly in builds tension as the rain begins.",
  "confidence": 0.85
}
```

---

## Architecture Flow

```text
Webcam → MediaPipe → Gesture → Virtual Camera (zoom/pan preview)
                    ↓
Microphone → STT (Gemini/OpenAI/Whisper) → Transcript
                    ↓
              LLM (Gemini/OpenAI) → Validated cinematic JSON
```

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ERR_CONNECTION_REFUSED` | Wait for `Running on local URL` in terminal |
| Stuck on `Loading Whisper tiny` | Set `STT_BACKEND=gemini` or `openai` with valid API key |
| `API_KEY_INVALID` / HttpError 400 | Use `AIza` key from [AI Studio](https://aistudio.google.com/apikey), not `AQ.` |
| Status `[!!] LLM / STT` | Fix `.env`, restart app |
| No hand skeleton / no gesture | Better lighting, one hand, lower sensitivity, Restart Camera |
| Black camera preview | Close Zoom/Teams/Camera app → **Restart Camera** |
| `No module named gradio` | Run `.venv\Scripts\python.exe app.py` |
| Transcript empty / too quiet | Speak louder, 3+ seconds, correct mic device |
| Transcript stuck on processing | Restart app; ensure webcam pauses during voice (built-in) |

---

## Test Before Demo / Interview

```powershell
.venv\Scripts\python.exe smoke_test.py
```

Expected output: all `[OK]` checks (imports, utils, gesture, webcam, voice, LLM, Gradio UI).

### Suggested 5-minute demo script

1. **Intro** — gesture camera + voice cinematic planning
2. **Gestures** — show skeleton → pan → zoom → action log
3. **Voice** — record line → Transcribe & Generate → show JSON
4. **Fusion** — gesture then voice with context checkbox
5. **Controls** — sensitivity, pause, restart camera

---

## Security

- `.env` is listed in `.gitignore` — **never commit API keys**
- Rotate any key that was shared in chat, screenshots, or logs
- Use `.env.example` as the public template only

---

## Tech Stack

- Python 3.9+
- OpenCV, MediaPipe Tasks API, NumPy, SciPy
- Gradio 4.x, python-dotenv
- google-generativeai (Gemini)
- openai (optional)
- faster-whisper (optional local STT)

---

## Author

**Created by Vijayshree**

GitHub: [vijayshreepathak/Ai-Virtual-Cammera-Control](https://github.com/vijayshreepathak/Ai-Virtual-Cammera-Control)
