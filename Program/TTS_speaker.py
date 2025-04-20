import threading
import pyttsx3

def say(text: str) -> threading.Thread:
    def _speak():
        engine = pyttsx3.init()
        engine.say(text)
        engine.runAndWait()
    thread = threading.Thread(target=_speak, daemon=True)
    thread.start()
    return thread
