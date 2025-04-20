import time
import pyaudio
from collections import deque

from STT_config import (
    SAVE_INTERVAL_SECONDS,
    WINDOW_LENGTH_SECONDS,
    RECORD_DURATION_SECONDS,
    FORMAT,
    CHANNELS,
    RATE,
    CHUNK,
    CHUNK_DURATION,
    CHUNKS_PER_SECOND,
    CHUNKS_PER_WINDOW,
)

def initialize_listener():
    """
    Initializes the PyAudio stream and audio buffer for real-time speech capture.
    
    Returns:
        stream (pyaudio.Stream): The open PyAudio stream.
        audio_buffer (deque): Rolling buffer for audio chunks.
        start_time (float): Timestamp when recording began.
        saved_second (int): Initialized counter for saved intervals.
        pa (pyaudio.PyAudio): The PyAudio instance (must be terminated at the end).
    """
    pa = pyaudio.PyAudio()
    stream = pa.open(format=FORMAT,
                     channels=CHANNELS,
                     rate=RATE,
                     input=True,
                     frames_per_buffer=CHUNK)

    audio_buffer = deque(maxlen=CHUNKS_PER_WINDOW)
    start_time = time.time()
    saved_second = 0

    print("🎙️ Listening...")

    return stream, audio_buffer, start_time, saved_second, pa
