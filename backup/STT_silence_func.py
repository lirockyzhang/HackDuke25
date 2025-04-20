import audioop

def is_silence(pcm_file_path: str, silence_threshold: int = 500) -> bool:
    """
    Determine if a raw PCM file is silent based on RMS energy.
    Returns True if the energy is below the silence threshold.
    """
    try:
        with open(pcm_file_path, "rb") as f:
            pcm_data = f.read()

        if len(pcm_data) == 0:
            return True

        rms = audioop.rms(pcm_data, 2)  # 2 bytes/sample for 16-bit audio
        return rms < silence_threshold

    except Exception as e:
        print(f"⚠️ Error in is_silence(): {e}")
        return True

import re

def is_english(text: str) -> bool:
    """
    Returns True if the text contains mostly English letters.
    You can tune the threshold depending on your needs.
    """
    # Count English words or alphabet characters
    english_chars = re.findall(r'[a-zA-Z]', text)
    return len(english_chars) / max(len(text), 1) > 0.5  # more than 50% english
