import pyaudio
import os

# === Transcription Settings ===
SAVE_INTERVAL_SECONDS = 1
WINDOW_LENGTH_SECONDS = 5
RECORD_DURATION_SECONDS = 60

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
CHUNK_DURATION = CHUNK / RATE

CHUNKS_PER_SECOND = int(1 / CHUNK_DURATION)
CHUNKS_PER_WINDOW = int(WINDOW_LENGTH_SECONDS * CHUNKS_PER_SECOND)

# === OpenAI Whisper Settings ===
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_URL = "https://api.openai.com/v1/audio/transcriptions"
MODEL_NAME = "whisper-1"