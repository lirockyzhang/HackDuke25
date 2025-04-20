import time
import pyaudio
from collections import deque

from STT_config import (
    FORMAT,
    CHANNELS,
    RATE,
    CHUNK,
    CHUNKS_PER_WINDOW,
)

# STT_init.py
import time, collections, pyaudio
from STT_config import CHUNK, WINDOW_LENGTH_SECONDS          # your existing constants

def initialize_listener(*, rate: int = 16_000, channels: int = 1,
                        pa: pyaudio.PyAudio | None = None):
    """
    Open a fresh *input* stream and return everything the caller needs.

    Returns
    -------
    stream, audio_buffer, start_time, saved_second, pa
    """
    pa = pa or pyaudio.PyAudio()      # create once, reuse forever

    stream = pa.open(format=pyaudio.paInt16,
                     channels=channels,
                     rate=rate,
                     input=True,
                     frames_per_buffer=CHUNK)

    audio_buffer = collections.deque(
        maxlen=int(rate * WINDOW_LENGTH_SECONDS * 2)  # 2 bytes/sample
    )

    start_time   = time.time()
    saved_second = 0                  # handy if you still track this elsewhere
    return stream, audio_buffer, start_time, saved_second, pa
