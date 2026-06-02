# AI Virtual Camera Control

Real-time AI virtual camera demo with:
- hand gesture based camera controls
- voice-to-text cinematic command input
- structured JSON cinematic plan generation with an LLM

## Project Structure

```text
.
├── app.py
├── camera_simulator.py
├── gesture_detector.py
├── llm_cinematic.py
├── voice_to_text.py
├── webcam.py
├── utils.py
├── smoke_test.py
├── run.bat
├── requirements.txt
├── .env.example
└── README.md
```

## Features

- Live webcam preview with overlays
- Gesture detection: Pan Left, Pan Right, Zoom In, Zoom Out, Tilt Up
- Virtual camera simulation (actual visible pan/zoom transform)
- Confidence score + FPS + action history
- Voice command recording and transcription
- LLM-generated strict JSON cinematic output
- Dark modern Gradio UI

## Requirements

- Python 3.9+ (3.11 recommended)
- Webcam
- Microphone
- Internet for OpenAI/Gemini APIs (if enabled)

## Installation

```bash
# 1) Open project folder
cd "f:\Ai Engg Assignment"

# 2) Create venv
python -m venv .venv

# 3) Activate venv
# Windows (PowerShell)
.venv\Scripts\activate

# 4) Install dependencies
pip install -r requirements.txt
```

## Environment Setup

1) Copy env template:

```bash
copy .env.example .env
```

2) Update `.env` values:

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
OPENAI_WHISPER_MODEL=whisper-1

GEMINI_API_KEY=
GEMINI_MODEL=gemini-2.0-flash

LLM_PROVIDER=openai
STT_BACKEND=openai
WHISPER_MODEL_SIZE=tiny
```

### Variable Notes

- `STT_BACKEND=openai` gives faster transcription.
- `STT_BACKEND=faster-whisper` works locally but first load may be slow.
- Keep `.env` private. It is already ignored by git.

## Run the App

### Recommended (runs smoke tests first)
```bash
.\run.bat
```

### Manual
```bash
.venv\Scripts\python.exe app.py
```

Open:
- [http://127.0.0.1:7860](http://127.0.0.1:7860)
- if 7860 is busy, the next available local port is used

## Usage Steps

1) Start app and confirm camera preview appears.
2) Test gestures (move slowly and clearly):
   - left to right -> Pan Right
   - right to left -> Pan Left
   - toward camera -> Zoom In
   - away from camera -> Zoom Out
3) Record voice command from microphone.
4) Click **Transcribe & Generate**.
5) Check Transcript, Cinematic Plan, and JSON tabs.

## Accuracy Tips

- Use good lighting
- Keep only one hand in frame
- Move slowly and consistently
- Avoid mixing pan and zoom in one motion
- Use sensitivity around `1.0` (adjust if needed)
- If camera is black: click **Restart Camera**

## Troubleshooting

- **No module named gradio**: run via `.venv\Scripts\python.exe app.py`
- **Blank camera**: close other camera apps (Zoom/Teams), click Restart Camera
- **Transcript stuck**: ensure microphone source is selected (not Stereo Mix)
- **API errors**: verify keys in `.env`

## Test Before Demo

```bash
.venv\Scripts\python.exe smoke_test.py
```

Expected: all checks pass.

## Security

- `.env` is ignored by git
- API keys should never be committed

## Author

Created by Vijayshree
