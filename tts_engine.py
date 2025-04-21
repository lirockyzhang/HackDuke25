"""
TTS Engine Module

This module provides text-to-speech functionality using Coqui TTS.
It handles model initialization, audio synthesis, and queue management
for sequential playback of text chunks.

Dev Plan:
Return response on what I said back with speak_text()
Fix the interrupt function to allow for immediate speech synthesis
"""

import os
import io
import threading
import queue
import re
from pydub import AudioSegment
from pydub.playback import play as play_with_pydub
import torch

# Try to import TTS, but handle import errors gracefully
try:
    from TTS.api import TTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("WARNING: Coqui TTS not available - TTS features will be disabled")

SPEAKER_WAV = ["voice/female.wav", "voice/29.wav", "voice/30.wav", "voice/1.wav", "voice/2.wav", "voice/3.wav", "voice/4.wav"]  # voice/female.wav

class TTSEngine:
    """
    Text-to-Speech Engine using Coqui TTS.
    
    Provides:
    - Model initialization
    - Text chunking by punctuation
    - Queued playback with a dedicated worker thread
    - Error reporting
    - Speech completion tracking
    """
    
    def __init__(self, model_name="tts_models/multilingual/multi-dataset/xtts_v2", device=None,
                 speaker_wav=None, # Add speaker_wav parameter
                 callback_on_chunk_start=None, callback_on_chunk_end=None,
                 callback_on_speech_complete=None):
        """
        Initialize the TTS Engine.
        
        Args:
            model_name: Coqui TTS model name to use
            device: 'cuda' or 'cpu' (None for auto-detection)
            speaker_wav: Path to the reference speaker WAV file (required for multi-speaker models like XTTS)
            callback_on_chunk_start: Function to call when starting a chunk
            callback_on_chunk_end: Function to call when ending a chunk
            callback_on_speech_complete: Function to call when speech is complete or interrupted
        """
        self.model_name = model_name
        self.model = None
        self.tts_available = TTS_AVAILABLE
        self.last_error = None
        self.is_initialized = False
        self.speaker_wav = speaker_wav # Store speaker_wav path
        
        # Set device if provided, otherwise auto-detect
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # UI Callback functions
        self.callback_on_chunk_start = callback_on_chunk_start
        self.callback_on_chunk_end = callback_on_chunk_end
        self.callback_on_speech_complete = callback_on_speech_complete
        
        # Playback queue and thread control
        self.tts_queue = queue.Queue()
        self.thread_should_run = threading.Event()
        self.thread_should_run.set()
        self.playback_thread = None
        
        # Speech tracking
        self.current_speech_id = None
        self.current_chunks = []
        self.spoken_chunks = []
        self.speech_interrupted = False

        # Check if speaker_wav is provided for XTTS model
        if "xtts" in model_name and not speaker_wav:
             print("WARNING: XTTS model selected, but no speaker_wav provided. Synthesis might fail.")
             # You might want to raise an error or use a default speaker wav here
             # For now, we'll proceed but log a warning.
             # Example: self.speaker_wav = "path/to/default/speaker.wav"
             # Or: raise ValueError("speaker_wav must be provided for XTTS models")

        # Initialize the model if TTS is available
        if self.tts_available:
            self._initialize_model()
            
        # Start the playback thread if initialization was successful
        if self.is_initialized:
            self._start_playback_thread()

    def _initialize_model(self):
        """Initialize the Coqui TTS model."""
        if not self.tts_available:
            self.last_error = "Coqui TTS not available. Check installation."
            return False
            
        try:
            print(f"Initializing Coqui TTS model '{self.model_name}' on {self.device}...")
            print("(This might download model files on the first run)")
            self.model = TTS(model_name=self.model_name, progress_bar=True).to(self.device)
            print(f"Coqui TTS model '{self.model_name}' initialized on {self.device}.")
            self.is_initialized = True
            return True
        except Exception as e:
            import traceback
            self.last_error = f"Error initializing Coqui TTS model: {str(e)}"
            print(f"*** {self.last_error}")
            print(traceback.format_exc())
            self.is_initialized = False
            return False

    def _start_playback_thread(self):
        """Start the dedicated playback thread."""
        if self.playback_thread is None or not self.playback_thread.is_alive():
            self.playback_thread = threading.Thread(
                target=self._playback_worker, 
                daemon=True
            )
            self.playback_thread.start()
            print("TTS playback thread started")

    def _synthesize_and_play(self, text):
        """Synthesize text to speech and play it."""
        if not self.is_initialized or not text or not str(text).strip():
            return
            
        try:
            # Synthesize to a BytesIO buffer in memory
            buf = io.BytesIO()
            # Pass the speaker_wav argument
            self.model.tts_to_file(
                text=str(text),
                file_path=buf,
                speaker_wav=self.speaker_wav, # Add this line
                language="en" # XTTS requires language, adjust if needed
            )
            buf.seek(0)
            sound = AudioSegment.from_file(buf, format="wav")
            play_with_pydub(sound)
            return True
        except Exception as e:
            self.last_error = f"TTS synthesis error: {str(e)}"
            print(self.last_error)
            # Print traceback for more details during debugging
            import traceback
            traceback.print_exc()
            return False

    def chunk_text(self, text):
        """Split text into chunks by punctuation (. ; ! ?) and ensure each chunk has at least one word."""
        # Split on punctuation followed by whitespace or end of string
        chunks = re.findall(r'[^.,;!?]*\w[^.,;!?]*[.,;!?]', text)
        # Merge chunks to ensure each has at least 10 spaces (i.e., 11 words or so)
        merged_chunks = []
        buffer = ""
        for chunk in chunks:
            if buffer:
                buffer += " " + chunk
            else:
                buffer = chunk
            if buffer.count(" ") >= 10:
                merged_chunks.append(buffer.strip())
                buffer = ""
        if buffer.strip():
            merged_chunks.append(buffer.strip())
        chunks = merged_chunks
        return [chunk.strip() for chunk in chunks if chunk.strip()]
    
    def speak_text(self, text, callback_on_this_speech_complete=None):
        """
        Queue text to be spoken.

        Args:
            text: The text to be spoken.
            callback_on_this_speech_complete: A one-time callback for this specific speech

        Returns:
            bool: True if text was queued successfully, False otherwise.
        """
        if not self.is_initialized:
            self.last_error = "TTS Engine not initialized"
            return False

        try:
            # Split text into chunks by punctuation
            chunks = self.chunk_text(text)
            if not chunks:
                chunks = [text]  # If no punctuation, use the whole text as one chunk

            # If there's speech in progress, mark it as interrupted
            if self.current_speech_id:
                self.speech_interrupted = True
                # Signal interruption to any existing callback
                if self.callback_on_speech_complete:
                    spoken_text = " ".join(self.spoken_chunks)
                    self.callback_on_speech_complete(spoken_text + " <|interrupt|>")

            # Clear the queue if there are items
            self.clear_queue()

            # Set up tracking for the new speech
            self.current_speech_id = id(text)
            self.current_chunks = chunks.copy()
            self.spoken_chunks = []
            self.speech_interrupted = False

            # Store any one-time callback for this specific speech
            one_time_callback = callback_on_this_speech_complete

            # Synthesize audio for each chunk and queue (chunk, audio_data)
            for chunk in chunks:
                audio_data = self._synthesize_to_audiosegment(chunk)
                if audio_data is not None:
                    self.tts_queue.put((chunk, audio_data))
                else:
                    # If synthesis fails, skip this chunk
                    continue

            # Add end marker to signal completion
            self.tts_queue.put("__END_OF_SPEECH__")

            # If there's a one-time callback, set it up to execute when this speech completes
            if one_time_callback:
                original_callback = self.callback_on_speech_complete

                def combined_callback(spoken_text):
                    # Call the one-time callback
                    if one_time_callback:
                        one_time_callback(spoken_text)

                    # Restore the original callback
                    self.callback_on_speech_complete = original_callback

                    # Call the original callback if it exists
                    if original_callback:
                        original_callback(spoken_text)

                # Set the combined callback
                self.callback_on_speech_complete = combined_callback

            return True
        except Exception as e:
            self.last_error = f"Error queuing text for TTS: {str(e)}"
            return False

    def _synthesize_to_audiosegment(self, text):
        """Synthesize text to speech and return an AudioSegment."""
        if not self.is_initialized or not text or not str(text).strip():
            return None

        try:
            buf = io.BytesIO()
            self.model.tts_to_file(
                text=str(text),
                file_path=buf,
                speaker_wav=self.speaker_wav,  # <-- use 'speaker' not 'speaker_wav'
                language="en"
            )
            buf.seek(0)
            sound = AudioSegment.from_file(buf, format="wav")
            return sound
        except Exception as e:
            self.last_error = f"TTS synthesis error: {str(e)}"
            print(self.last_error)
            import traceback
            traceback.print_exc()
            return None

    def _playback_worker(self):
        """Worker thread that processes the TTS queue and plays audio."""
        while self.thread_should_run.is_set():
            try:
                item = self.tts_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if item is None:
                break  # Shutdown signal

            if item == "__END_OF_SPEECH__":
                # Special marker to indicate end of speech
                if self.callback_on_speech_complete and self.current_speech_id:
                    spoken_text = " ".join(self.spoken_chunks)
                    if self.speech_interrupted:
                        spoken_text += " <|interrupt|>"
                    self.callback_on_speech_complete(spoken_text)
                self.current_speech_id = None
                continue

            chunk, audio_data = item

            try:
                # Notify UI about chunk start
                if self.callback_on_chunk_start:
                    self.callback_on_chunk_start(chunk)

                # Play the audio data
                play_with_pydub(audio_data)

                # Track spoken chunks
                self.spoken_chunks.append(chunk)

            except Exception as e:
                self.last_error = f"TTS error: {str(e)}"
                print(self.last_error)
                self.speech_interrupted = True

                if self.callback_on_speech_complete and self.current_speech_id:
                    spoken_text = " ".join(self.spoken_chunks)
                    if self.speech_interrupted:
                        spoken_text += " <|interrupt|>"
                    self.callback_on_speech_complete(spoken_text)

                self.current_speech_id = None
                break  # Stop queue on error
            finally:
                # Notify UI about chunk end
                if self.callback_on_chunk_end:
                    self.callback_on_chunk_end(chunk)

    def speak_immediately(self, text):
            """
            Speak text immediately without queuing.
    
            Args:
                text: The text to be spoken.
    
            Returns:
                bool: True if synthesis and playback was successful, False otherwise.
            """
            if not self.is_initialized:
                self.last_error = "TTS Engine not initialized"
                return False
    
            audio_data = self._synthesize_to_audiosegment(text)
            if audio_data is not None:
                play_with_pydub(audio_data)
                return True
            else:
                return False
    
    # ...rest of the code unchanged...
    def clear_queue(self):
        """
        Clear all pending items from the queue.
        If speech is in progress, this will mark it as interrupted.
        """
        # Mark as interrupted if speech is in progress
        if self.current_speech_id and not self.speech_interrupted:
            self.speech_interrupted = True
            # Signal interruption to callback if it exists
            if self.callback_on_speech_complete:
                spoken_text = " ".join(self.spoken_chunks)
                self.callback_on_speech_complete(spoken_text + " <|interrupt|>")
            
        # Clear the queue
        while not self.tts_queue.empty():
            try:
                self.tts_queue.get_nowait()
            except queue.Empty:
                break

    def shutdown(self):
        """Stop the playback thread and release resources."""
        self.thread_should_run.clear()
        self.tts_queue.put(None)  # Signal thread to exit
        if self.playback_thread and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=1.0)  # Wait for thread to exit
        print("TTS Engine shut down")

    def get_last_error(self):
        """Get the last error message."""
        return self.last_error

    def is_ready(self):
        """Check if the TTS engine is ready for use."""
        return self.is_initialized and self.tts_available

# Create a global instance for easy import
tts_engine = None

# Update initialize to accept speaker_wav
def initialize(speaker_wav): # Add speaker_wav parameter
    """Initialize the global TTS engine instance."""
    global tts_engine
    # Pass speaker_wav to the constructor
    tts_engine = TTSEngine(speaker_wav=speaker_wav)
    return tts_engine.is_ready()

# Initialize the engine when module is imported
# You'll need to provide a path to a speaker wav file here
# For example: initialize(speaker_wav="path/to/your/speaker.wav")
# If you don't provide one, the warning in __init__ will be printed.
initialize(SPEAKER_WAV) # Consider adding a default path or handling the missing path