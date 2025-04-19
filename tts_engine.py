"""
TTS Engine Module

This module provides text-to-speech functionality using Coqui TTS.
It handles model initialization, audio synthesis, and queue management
for sequential playback of text chunks.

Dev Plan:
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
import tempfile

# Try to import TTS, but handle import errors gracefully
try:
    from TTS.api import TTS
    TTS_AVAILABLE = True
except ImportError:
    TTS_AVAILABLE = False
    print("WARNING: Coqui TTS not available - TTS features will be disabled")

class TTSEngine:
    """
    Text-to-Speech Engine using Coqui TTS.
    
    Provides:
    - Model initialization
    - Text chunking by punctuation
    - Queued playback with a dedicated worker thread
    - Error reporting
    """
    
    def __init__(self, model_name="tts_models/en/ljspeech/tacotron2-DDC", device=None, 
                 callback_on_chunk_start=None, callback_on_chunk_end=None):
        """
        Initialize the TTS Engine.
        
        Args:
            model_name: Coqui TTS model name to use
            device: 'cuda' or 'cpu' (None for auto-detection)
            callback_on_chunk_start: Function to call when starting a chunk
            callback_on_chunk_end: Function to call when ending a chunk
        """
        self.model_name = model_name
        self.model = None
        self.tts_available = TTS_AVAILABLE
        self.last_error = None
        self.is_initialized = False
        
        # Set device if provided, otherwise auto-detect
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # UI Callback functions
        self.callback_on_chunk_start = callback_on_chunk_start
        self.callback_on_chunk_end = callback_on_chunk_end
        
        # Playback queue and thread control
        self.tts_queue = queue.Queue()
        self.thread_should_run = threading.Event()
        self.thread_should_run.set()
        self.playback_thread = None
        
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

    def _playback_worker(self):
        """Worker thread that processes the TTS queue."""
        while self.thread_should_run.is_set():
            try:
                chunk = self.tts_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            if chunk is None:
                break  # Shutdown signal
                
            try:
                # Notify UI about chunk start
                if self.callback_on_chunk_start:
                    self.callback_on_chunk_start(chunk)
                
                # Synthesize and play the chunk
                self._synthesize_and_play(chunk)
            except Exception as e:
                self.last_error = f"TTS error: {str(e)}"
                print(self.last_error)
                break  # Stop queue on error
            finally:
                # Notify UI about chunk end
                if self.callback_on_chunk_end:
                    self.callback_on_chunk_end(chunk)

    def _synthesize_and_play(self, text):
        """Synthesize text to speech and play it."""
        if not self.is_initialized or not text or not str(text).strip():
            return
            
        try:
            # Synthesize to a BytesIO buffer in memory
            buf = io.BytesIO()
            self.model.tts_to_file(text=str(text), file_path=buf)
            buf.seek(0)
            sound = AudioSegment.from_file(buf, format="wav")
            play_with_pydub(sound)
            return True
        except Exception as e:
            self.last_error = f"TTS synthesis error: {str(e)}"
            print(self.last_error)
            return False

    def chunk_text(self, text):
        """Split text into chunks by punctuation (. ; ! ?) and ensure each chunk has at least one word."""
        # Split on punctuation followed by whitespace or end of string
        chunks = re.findall(r'[^.;!?]*\w[^.;!?]*[.;!?]', text)
        return [chunk.strip() for chunk in chunks if chunk.strip()]

    def speak_text(self, text):
        """
        Queue text to be spoken.
        
        Args:
            text: The text to be spoken.
            
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
                
            # Clear the queue if there are items
            self.clear_queue()
                
            # Add chunks to the queue
            for chunk in chunks:
                self.tts_queue.put(chunk)
            return True
        except Exception as e:
            self.last_error = f"Error queuing text for TTS: {str(e)}"
            return False

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
        
        return self._synthesize_and_play(text)

    def clear_queue(self):
        """Clear all pending items from the queue."""
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

def initialize():
    """Initialize the global TTS engine instance."""
    global tts_engine
    tts_engine = TTSEngine()
    return tts_engine.is_ready()

# Initialize the engine when module is imported
initialize()