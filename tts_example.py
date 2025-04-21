import unittest
import time
from tts_engine import TTSEngine

# Set this to a valid speaker wav file for XTTS
SPEAKER_WAV_PATH = "voice/female.wav"

class TestTTSEngine(unittest.TestCase):
    def setUp(self):
        self.engine = TTSEngine(speaker_wav=SPEAKER_WAV_PATH)

    def tearDown(self):
        self.engine.shutdown()

    def test_initialization(self):
        self.assertTrue(self.engine.is_ready(), "Engine should initialize successfully")

    def test_speak_immediately(self):
        result = self.engine.speak_immediately("This is a test of immediate speech.")
        self.assertTrue(result, "speak_immediately should return True")
        time.sleep(2)  # Allow time for playback

    def test_speak_text_queue(self):
        spoken = []
        def on_complete(text):
            spoken.append(text)
        result = self.engine.speak_text("Hello. This is a queued test.", callback_on_this_speech_complete=on_complete)
        self.assertTrue(result, "speak_text should return True")
        time.sleep(4)  # Allow time for playback
        self.assertTrue(spoken, "Callback should be called after speech")

    def test_clear_queue(self):
        self.engine.speak_text("This will be interrupted. This should not play.")
        time.sleep(1)
        self.engine.clear_queue()
        self.assertTrue(self.engine.speech_interrupted, "Speech should be marked as interrupted")

    def test_error_on_uninitialized(self):
        engine = TTSEngine(speaker_wav=None)
        engine.is_initialized = False
        result = engine.speak_immediately("Should fail")
        self.assertFalse(result, "Should not speak if not initialized")

if __name__ == "__main__":
    unittest.main()