import re
import os
import wave
import httpx
from io import BytesIO
from types import SimpleNamespace

from STT_config import OPENAI_API_KEY, API_URL, MODEL_NAME
from STT_silence_func import is_silence, is_english

# === Shared State ===
state = SimpleNamespace(prev_transcript=None)

# === Transcribe raw PCM file using Whisper API ===
def transcribe_file(audio_file_name, sample_rate=16000, channels=1):
    with open(audio_file_name, "rb") as f:
        pcm_data = f.read()

    bio = BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_data)
    bio.seek(0)

    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    files = {"file": ("chunk.wav", bio, "audio/wav")}
    data = {"model": MODEL_NAME}

    response = httpx.post(API_URL, headers=headers, files=files, data=data, timeout=30.0)
    return response.json().get("text", "").strip()

# === Extract only new part of transcript based on previous ===
def extract_new_segment(previous, current):
    def clean(text):
        return re.sub(r'[^\w\s]', '', text.lower())

    prev_clean = clean(previous).split()
    curr_clean = clean(current).split()
    curr_original = current.split()  # Keep original punctuation

    max_overlap = 0
    for i in range(len(prev_clean)):
        overlap_candidate = prev_clean[i:]
        if curr_clean[:len(overlap_candidate)] == overlap_candidate:
            max_overlap = len(overlap_candidate)

    new_words = curr_original[max_overlap:]
    return ' '.join(new_words)

# === Thread‑safe transcription task for streaming audio ===
def transcribe_task(curr_file: str, transcript_queue: "queue.Queue[str]"):
    """
    • Ensures returned_transcript never becomes "".
    • Pushes exactly one item to transcript_queue every call.
    • Safely removes the temp file.
    """
    returned_transcript = "<silence>"          # <- hard‑safety default
    curr_transcript     = "<silence>"          #    (updated below)

    try:
        # -------- 1. Get current transcript (or silence marker) --------
        if not is_silence(curr_file):          # we only call STT if audio isn’t silent
            text = transcribe_file(curr_file).strip()
            if text and is_english(text):      # drop non‑English / garbage
                curr_transcript = text

        # -------- 2. Decide what to return --------
        prev = getattr(state, "prev_transcript", None)

        if prev and not prev.startswith("<silence>"):
            # We had real speech last time → return only the new tail
            new_segment = extract_new_segment(prev, curr_transcript)
            returned_transcript = new_segment or "<silence>"
        else:
            # First chunk or we were in silence → pass the whole thing
            returned_transcript = curr_transcript or "<silence>"

        # -------- 3. Update state --------
        state.prev_transcript = curr_transcript

    except Exception as e:
        # Any failure still yields "<silence>" instead of blocking/blank
        print("⚠️ Error during transcription:", e)
        state.prev_transcript = "<silence>"

    finally:
        # Always push one (non‑blank) result
        transcript_queue.put(returned_transcript)

        # Cleanup temp file, ignore if already gone
        try:
            os.remove(curr_file)
        except FileNotFoundError:
            pass

