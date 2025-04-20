"""
STT_audio_listener.py
A robust, self‑contained `audio_listener()` that you can call from main.py:

    transcript = audio_listener(stream, audio_buffer, start_time, pa, q)

It handles:
• all stop‑conditions
• stream‑read errors (–9988, –9997, etc.)
• stream cleanup (but leaves `pa` alive so you can reuse it)
"""

import errno, queue, threading, time
from STT_save_func       import save_audio_snapshot, should_save_snapshot
from STT_transcribe_func import transcribe_task
from STT_config import (
    RECORD_DURATION_SECONDS,
    CHUNK,
    WINDOW_LENGTH_SECONDS,
    SAVE_INTERVAL_SECONDS,
)

# ---------------------------------------------------------------------------
def audio_listener(stream,
                   audio_buffer,
                   start_time,
                   pa,
                   transcript_queue):
    """
    Listen on `stream`, spawn STT jobs, and return the full transcript.

    Parameters
    ----------
    stream : pyaudio.Stream     (already open for input)
    audio_buffer : collections.deque[bytes]
    start_time : float          (time.time() when recording began)
    pa : pyaudio.PyAudio        (keep this alive for reuse)
    transcript_queue : queue.Queue[str]

    Returns
    -------
    transcript_str : str
    """
    saved_second   = 0
    transcript_str = ""

    try:
        while True:
            # 1) Read next chunk; bail if the stream closes
            try:
                chunk_data = stream.read(CHUNK, exception_on_overflow=False)
            except OSError:
                # PyAudio throws –9988/–9997 when the device closes;
                # treat any OSError here as “stop recording”.
                break

            audio_buffer.append(chunk_data)

            # 2) Time bookkeeping
            elapsed = time.time() - start_time

            # 3) Save window & spawn transcription thread
            if should_save_snapshot(elapsed, saved_second,
                                    WINDOW_LENGTH_SECONDS, SAVE_INTERVAL_SECONDS):
                saved_second = int(elapsed)
                wav_file = save_audio_snapshot(saved_second, audio_buffer)
                threading.Thread(
                    target=transcribe_task,
                    args=(wav_file, transcript_queue),
                    daemon=True,
                ).start()

            # 4) Drain completed STT jobs
            try:
                while True:
                    text = transcript_queue.get_nowait()
                    transcript_str = (transcript_str + " " + text).strip()
            except queue.Empty:
                pass

            # 5) Stop conditions
            if transcript_str.endswith("<silence> <silence>"):
                break
            if elapsed >= RECORD_DURATION_SECONDS:
                break

    finally:
        # 6) Clean up stream (safe even if already closed)
        try:
            if stream.is_active():
                stream.stop_stream()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass
        # DO NOT call pa.terminate() here—only do that once,
        # right before your whole program exits.

    return transcript_str