import threading
import time

from STT_config import (RECORD_DURATION_SECONDS, CHUNK)
from STT_save_func import save_audio_snapshot, should_save_snapshot
from STT_transcribe_func import transcribe_task

def audio_listener(stream, audio_buffer, start_time, pa, transcript_queue):
    saved_second = 0

    while True:
        chunk = stream.read(CHUNK, exception_on_overflow=False)
        audio_buffer.append(chunk)

        elapsed_time = time.time() - start_time

        if should_save_snapshot(elapsed_time, saved_second):
            saved_second = int(elapsed_time)
            curr_file = save_audio_snapshot(saved_second, audio_buffer)
            threading.Thread(
                target=transcribe_task,
                args=(curr_file, transcript_queue)
            ).start()

        if elapsed_time > RECORD_DURATION_SECONDS:
            break

    stream.stop_stream()
    stream.close()
    pa.terminate()
    transcript_queue.put(None)  # Signal transcript processor to exit