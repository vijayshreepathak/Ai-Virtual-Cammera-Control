"""AI Virtual Camera Control — Gradio demo application."""

from __future__ import annotations

import atexit
from typing import Optional, Tuple

import cv2
import gradio as gr
import numpy as np

from gesture_detector import GestureDetector
from llm_cinematic import generate_cinematic_safe
from utils import (
    format_error,
    gemini_key_debug_label,
    gemini_key_hint,
    has_llm_credentials,
    patch_gradio_client,
    system_status,
)
from voice_to_text import preload_stt, transcribe_audio_safe
from webcam import find_working_camera, is_black_frame, read_frame

patch_gradio_client()

_detector: Optional[GestureDetector] = None
_webcam: Optional[cv2.VideoCapture] = None
_webcam_error: Optional[str] = None
_webcam_info: str = "Not connected"
_black_frame_count: int = 0
_paused: bool = False


def get_detector() -> GestureDetector:
    global _detector
    if _detector is None:
        _detector = GestureDetector()
    return _detector


def release_webcam() -> None:
    global _webcam, _webcam_error, _webcam_info, _black_frame_count
    if _webcam is not None:
        _webcam.release()
        _webcam = None
    _webcam_error = None
    _webcam_info = "Not connected"
    _black_frame_count = 0


def get_webcam() -> Optional[cv2.VideoCapture]:
    global _webcam, _webcam_error, _webcam_info
    if _webcam is None:
        cap, info = find_working_camera()
        if cap is None:
            _webcam_error = info + ". Close Zoom/Teams/Camera app and click Restart Camera."
            return None
        _webcam = cap
        _webcam_info = info
        _webcam_error = None
        print(f"Webcam connected: {info}")
    return _webcam


def restart_webcam():
    release_webcam()
    cap = get_webcam()
    if cap is None:
        return format_error(_webcam_error or "Could not open camera."), _webcam_info
    return f"Connected: {_webcam_info}", _webcam_info


def cleanup() -> None:
    global _detector
    if _detector is not None:
        _detector.close()
        _detector = None
    release_webcam()


atexit.register(cleanup)


def process_webcam_tick():
    """Read webcam frame, detect gestures, return UI updates."""
    global _black_frame_count
    placeholder = np.zeros((360, 480, 3), dtype=np.uint8)

    if _paused:
        return placeholder, "Paused", "—", "—", "—", get_detector().format_action_log(), _webcam_info

    cap = get_webcam()
    if cap is None:
        msg = format_error(_webcam_error or "Webcam not available.")
        return placeholder, msg, "—", "—", "—", get_detector().format_action_log(), _webcam_info

    ok, frame = read_frame(cap)
    if not ok or frame is None:
        release_webcam()
        msg = format_error("Camera disconnected. Click Restart Camera.")
        return placeholder, msg, "—", "—", "—", get_detector().format_action_log(), _webcam_info

    if is_black_frame(frame):
        _black_frame_count += 1
        if _black_frame_count >= 5:
            release_webcam()
            cap = get_webcam()
            _black_frame_count = 0
            if cap is None:
                msg = format_error(_webcam_error or "Camera returned black frames.")
                return placeholder, msg, "—", "—", "—", get_detector().format_action_log(), _webcam_info
            ok, frame = read_frame(cap)
            if not ok or frame is None or is_black_frame(frame):
                msg = format_error("Camera feed is black. Click Restart Camera.")
                return placeholder, msg, "—", "—", "—", get_detector().format_action_log(), _webcam_info
    else:
        _black_frame_count = 0

    try:
        detector = get_detector()
        result = detector.process_frame(frame)
        rgb_out = cv2.cvtColor(result.frame, cv2.COLOR_BGR2RGB)
        conf = f"{result.confidence_score:.0%}" if result.confidence_score > 0 else "—"
        fps_str = f"{result.fps:.0f} FPS"
        log = detector.format_action_log()
        cam_line = result.camera_state or _webcam_info
        return rgb_out, result.gesture_label, result.action_label, conf, fps_str, log, cam_line
    except Exception as exc:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return rgb, format_error(str(exc)), "—", "—", "—", get_detector().format_action_log(), _webcam_info


def toggle_pause():
    global _paused
    _paused = not _paused
    return "▶ Resume Webcam" if _paused else "⏸ Pause Webcam"


def set_sensitivity(level: float):
    get_detector().set_sensitivity(level)
    return f"Sensitivity: {level:.1f}x (lower = easier to trigger)"


def reset_camera_view():
    get_detector().reset_camera_view()
    return "Camera view reset to default.", get_detector().camera.state.label()


def process_voice(
    audio: Optional[Tuple[int, np.ndarray]],
    use_gesture_context: bool,
):
    """Transcribe audio and generate cinematic output. Pauses webcam to avoid queue blocking."""
    global _paused
    was_paused = _paused
    _paused = True

    try:
        yield "Transcribing audio… (webcam paused)", "{}", "Processing step 1/2: speech-to-text…"

        transcript, stt_error = transcribe_audio_safe(audio)
        if stt_error:
            yield stt_error, "{}", format_error("Cinematic output skipped — fix transcription first.")
            return

        yield transcript, "{}", "Processing step 2/2: generating cinematic plan…"

        if not has_llm_credentials():
            err = format_error("No LLM API key found. Set OPENAI_API_KEY or GEMINI_API_KEY.")
            yield transcript, "{}", err
            return

        gesture_ctx = ""
        if use_gesture_context:
            d = get_detector()
            if d.state.last_confidence > 0:
                gesture_ctx = f"{d.state.last_action_label} (confidence {d.state.last_confidence:.0%})"

        cinematic, cinematic_json, llm_error = generate_cinematic_safe(transcript, gesture_ctx)
        if llm_error:
            yield transcript, cinematic_json, llm_error + "\n\n" + cinematic
        else:
            yield transcript, cinematic_json, cinematic
    finally:
        _paused = was_paused


CUSTOM_CSS = """
.gradio-container { max-width: 1400px !important; margin: auto; background: #0a0c10 !important; }
.hero { text-align: center; padding: 1.2rem 0 0.5rem; }
.hero h1 { font-size: 2rem; font-weight: 700; background: linear-gradient(90deg, #6ea8fe, #a78bfa); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin: 0; }
.hero p { color: #9ca3af; margin: 0.4rem 0 0; }
.status-bar { background: #12151c; border: 1px solid #2a3040; border-radius: 10px; padding: 10px 16px; font-size: 0.85rem; color: #a3e635; font-family: monospace; }
.panel { background: #12151c !important; border: 1px solid #252a35 !important; border-radius: 12px !important; padding: 4px !important; }
.gesture-card { background: linear-gradient(135deg, #1a1f2e, #12151c); border-radius: 10px; padding: 12px; border: 1px solid #2a3040; }
.gesture-card label { color: #94a3b8 !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.05em; }
footer { display: none !important; }
"""

DARK_THEME = gr.themes.Base(primary_hue="blue", neutral_hue="slate").set(
    body_background_fill="#0a0c10",
    body_background_fill_dark="#0a0c10",
    block_background_fill="#12151c",
    block_background_fill_dark="#12151c",
    block_border_color="#252a35",
    block_label_text_color="#94a3b8",
    body_text_color="#e2e8f0",
    body_text_color_subdued="#64748b",
    button_primary_background_fill="linear-gradient(90deg, #3b82f6, #6366f1)",
)

TESTING_GUIDE = """
### How to Test This Demo

**1. Gesture Detection (30 sec)**
- Sit ~2 feet from webcam, good lighting on your hand
- Hold one hand up, palm facing camera
- **Pan Right**: slowly move hand left → right
- **Pan Left**: slowly move hand right → left
- **Zoom In**: move hand toward camera
- **Zoom Out**: pull hand away
- **Tilt Up**: move hand upward
- Watch confidence score and Action Log update

**2. Voice + LLM (60 sec)**
- Click Record, speak clearly: *"Slow dolly in on the hero as rain starts, tense mood"*
- Click **Transcribe & Generate**
- Check Transcript + JSON panels
- Enable **Use last gesture as context** and try again after a gesture

**3. Controls**
- **Pause Webcam** stops processing (saves CPU)
- **Sensitivity slider**: lower = easier to trigger, higher = harder
- FPS counter should show 15–30 on a typical laptop

**4. Troubleshooting**
- No gesture? Increase lighting, move slower, lower sensitivity
- Slow FPS? Click Pause, or close other camera apps
- Voice fails? Select **Microphone** not Stereo Mix in audio dropdown
"""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="AI Virtual Camera Control", css=CUSTOM_CSS, theme=DARK_THEME) as demo:
        gr.HTML(
            """
            <div class="hero">
              <h1>🎬 AI Virtual Camera Control</h1>
              <p>Real-time gesture camera actions · Voice-driven cinematic AI planning</p>
            </div>
            """
        )
        _gemini_warn = gemini_key_hint()
        if _gemini_warn:
            gr.Markdown(
                f"### ⚠ Gemini API key problem\n{_gemini_warn}",
                elem_classes=["status-bar"],
            )
        status_bar = gr.Markdown(system_status(), elem_classes=["status-bar"])

        with gr.Row():
            # ── LEFT: Webcam ──
            with gr.Column(scale=5, elem_classes=["panel"]):
                webcam = gr.Image(label="📷 Live Webcam", height=420, interactive=False)
                with gr.Row():
                    fps_box = gr.Textbox(label="Performance", value="—", interactive=False, scale=1)
                    pause_btn = gr.Button("⏸ Pause Webcam", scale=1, size="sm")
                    restart_cam_btn = gr.Button("🔄 Restart Camera", scale=1, size="sm")
                    reset_view_btn = gr.Button("↺ Reset Zoom/Pan", scale=1, size="sm")
                    sensitivity = gr.Slider(0.5, 1.5, value=0.75, step=0.1, label="Gesture Sensitivity (lower = easier)", scale=2)
                with gr.Row():
                    sens_label = gr.Markdown("Sensitivity: 1.0x — move hand slowly for best accuracy")
                    cam_status = gr.Textbox(label="Camera", value="Connecting…", interactive=False, scale=2)

                with gr.Accordion("📋 Gesture Guide & Testing Instructions", open=False):
                    gr.Markdown(
                        """
| Gesture | Hand Movement | Camera Action |
|---------|--------------|---------------|
| Pan Right | Sweep hand **slowly left → right** (keep distance from camera) | Preview pans right |
| Pan Left | Sweep hand **slowly right → left** | Preview pans left |
| Zoom In | Move open palm **straight toward** camera | Preview zooms in |
| Zoom Out | Pull hand **straight away** from camera | Preview zooms out |
| Tilt Up | Move hand **upward** | Preview tilts up |

**Important:** Pan uses horizontal motion only. Zoom uses moving closer/farther only — do not mix both at once.
                        """
                    )
                    gr.Markdown(TESTING_GUIDE)

            # ── RIGHT: Status + Voice ──
            with gr.Column(scale=4, elem_classes=["panel"]):
                gr.Markdown("### 🖐 Live Gesture Status")
                with gr.Row():
                    gesture_box = gr.Textbox(label="Detected Gesture", value="—", interactive=False)
                    action_box = gr.Textbox(label="Camera Action", value="—", interactive=False)
                confidence_box = gr.Textbox(label="Confidence", value="—", interactive=False)

                gr.Markdown("### 📜 Action History")
                action_log = gr.Textbox(label="Recent Triggers", lines=5, interactive=False, max_lines=8)

                gr.Markdown("### 🎙 Voice Cinematic Command")
                mic = gr.Audio(sources=["microphone"], type="numpy", label="Record Instruction")
                gr.Markdown(
                    "*Tip: Select **Microphone** in the audio dropdown — not Stereo Mix.*",
                    elem_classes=["status-bar"],
                )
                use_gesture = gr.Checkbox(label="Use last gesture as LLM context", value=True)
                transcribe_btn = gr.Button("✨ Transcribe & Generate", variant="primary", size="lg")

                with gr.Tabs():
                    with gr.Tab("Transcript"):
                        transcript_box = gr.Textbox(lines=3, interactive=False, show_label=False, placeholder="Transcribed speech…")
                    with gr.Tab("Cinematic Plan"):
                        cinematic_box = gr.Textbox(lines=8, interactive=False, show_label=False, placeholder="Structured cinematic output…")
                    with gr.Tab("JSON"):
                        json_box = gr.Code(language="json", lines=10, interactive=False, label="Raw JSON")

        # ── Event wiring ──
        timer = gr.Timer(0.09)
        timer.tick(
            fn=process_webcam_tick,
            outputs=[webcam, gesture_box, action_box, confidence_box, fps_box, action_log, cam_status],
            show_progress="hidden",
        )

        pause_btn.click(fn=toggle_pause, outputs=[pause_btn])
        restart_cam_btn.click(fn=restart_webcam, outputs=[gesture_box, cam_status])
        reset_view_btn.click(fn=reset_camera_view, outputs=[gesture_box, cam_status])
        sensitivity.change(fn=set_sensitivity, inputs=[sensitivity], outputs=[sens_label])

        transcribe_btn.click(
            fn=process_voice,
            inputs=[mic, use_gesture],
            outputs=[transcript_box, json_box, cinematic_box],
            show_progress="full",
        )

    return demo


def main() -> None:
    print(gemini_key_debug_label())
    hint = gemini_key_hint()
    if hint:
        print("\n" + "=" * 60)
        print("GEMINI API KEY ERROR")
        print(hint)
        print("Edit:", "F:\\Ai Engg Assignment\\.env")
        print("=" * 60 + "\n")
    print("Pre-loading gesture detector…")
    get_detector()
    get_detector().set_sensitivity(0.75)
    print("Connecting webcam…")
    cap = get_webcam()
    if cap is None:
        print("Webcam warning:", _webcam_error)
    else:
        print(f"Webcam ready: {_webcam_info}")
    print("Pre-loading speech-to-text…")
    try:
        stt_info = preload_stt()
        print(f"STT ready: {stt_info}")
    except Exception as exc:
        print(f"STT preload warning: {exc}")
    print("System:", system_status())
    demo = build_ui()
    demo.queue(default_concurrency_limit=2)
    demo.launch(show_api=False, share=False)


if __name__ == "__main__":
    main()
