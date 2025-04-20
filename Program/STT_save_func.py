from collections import deque

# === Save audio snapshot ===
def save_audio_snapshot(current_second: int, audio_buffer: deque) -> str:
    """
    Save the current audio buffer to a .pcm file and return the file path.
    """
    filename = f"recorded_{current_second}s.pcm"
    with open(filename, "wb") as f:
        f.writelines(audio_buffer)
    return filename

# === Determine if it's time to save ===
def should_save_snapshot(elapsed_time: float, last_saved_second: int,
                         window_length, save_interval) -> bool:
    """
    Return True if the current elapsed time meets the conditions for saving a snapshot.
    """
    current_second = int(elapsed_time)
    return (
        elapsed_time >= 3 and
        current_second % save_interval == 0 and
        current_second != last_saved_second
    )
