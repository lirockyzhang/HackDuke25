"""
STT Engine Module

This module provides speech-to-text functionality using OpenAI's Whisper API.
It handles audio capture, voice activity detection, and transcription with 
a robust error handling mechanism.

Dev Plan:
- Improve voice activity detection
- Add more robust error handling
- Enhance transcript synchronization
"""

import os
import re
import time
import wave
import httpx
import audioop
import threading
import pyaudio
from io import BytesIO
from types import SimpleNamespace
from collections import deque
from typing import Callable, Optional, List, Dict, Any, Union
import dotenv

# Load environment variables from .env file
dotenv.load_dotenv()

# Try to import required libraries, but handle import errors gracefully
WHISPER_AVAILABLE = True
if not os.getenv("OPENAI_API_KEY"):
    print("WARNING: OpenAI API key not available - STT features will be limited")
    WHISPER_AVAILABLE = False

# === Configuration Settings ===
# Audio recording settings
SAVE_INTERVAL_SECONDS = 1
WINDOW_LENGTH_SECONDS = 5
RECORD_DURATION_SECONDS = 60  # Maximum recording duration

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
CHUNK_DURATION = CHUNK / RATE

CHUNKS_PER_SECOND = int(1 / CHUNK_DURATION)
CHUNKS_PER_WINDOW = int(WINDOW_LENGTH_SECONDS * CHUNKS_PER_SECOND)

# OpenAI Whisper API settings
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
API_URL = "https://api.openai.com/v1/audio/transcriptions"
MODEL_NAME = "whisper-1"

class STTEngine:
    """
    Speech-to-Text Engine using OpenAI's Whisper API.
    
    Provides:
    - Audio capture from microphone
    - Voice activity detection
    - Transcription using Whisper API
    - Callback functionality for new transcripts
    - Robust error handling
    """
    
    def __init__(self, callback_on_transcript: Optional[Callable[[str], None]] = None):
        """
        Initialize the STT Engine.
        
        Args:
            callback_on_transcript: Function to call when a new transcript is available
        """
        self.callback_on_transcript = callback_on_transcript
        self.is_initialized = False
        self.is_listening = False
        self.last_error = None
        self.whisper_available = WHISPER_AVAILABLE
        
        # Audio components
        self.pa = None
        self.stream = None
        self.audio_buffer = None
        
        # Thread control
        self.listener_thread = None
        self.start_time = None
        self.saved_second = 0
        self.transcript_queue = None
        self.state = SimpleNamespace(prev_transcript=None)
        
        # Voice Activity Detection
        self.silence_threshold = 300  # Default threshold
        self.speech_detected = False
        self.speech_timeout = 1.0  # Seconds of silence before considering speech ended
        self.last_speech_time = 0
        
        # Try to initialize
        self._initialize()
    
    def _initialize(self) -> bool:
        """
        Initialize the STT engine components.
        
        Returns:
            bool: True if initialization was successful
        """
        try:
            # Always initialize to allow microphone testing even without API key
            self.is_initialized = True
            return True
            
        except Exception as e:
            self.last_error = f"Error initializing STT Engine: {str(e)}"
            print(f"*** {self.last_error}")
            self.is_initialized = False
            return False
    
    def list_microphones(self) -> List[Dict[str, Any]]:
        """
        List all available microphone devices.
        
        Returns:
            List[Dict[str, Any]]: List of microphone devices with their info
        """
        try:
            if not self.pa:
                self.pa = pyaudio.PyAudio()
                
            devices = []
            info = self.pa.get_host_api_info_by_index(0)
            num_devices = info.get('deviceCount')
            
            for i in range(num_devices):
                device_info = self.pa.get_device_info_by_index(i)
                if device_info.get('maxInputChannels') > 0:  # This is an input device
                    devices.append({
                        'index': i,
                        'name': device_info.get('name'),
                        'channels': device_info.get('maxInputChannels'),
                        'sample_rate': int(device_info.get('defaultSampleRate'))
                    })
                    
            return devices
            
        except Exception as e:
            self.last_error = f"Error listing microphones: {str(e)}"
            print(f"*** {self.last_error}")
            return []
    
    def start_listening(self, device_index: Optional[int] = None, 
                        silence_threshold: int = 300) -> bool:
        """
        Start listening for audio input.
        
        Args:
            device_index: Index of the input device to use (None for default)
            silence_threshold: Threshold for silence detection (lower = more sensitive)
            
        Returns:
            bool: True if started successfully, False otherwise
        """
        if not self.is_initialized:
            self.last_error = "STT Engine not initialized"
            return False
            
        if self.is_listening:
            return True  # Already listening
            
        try:
            # Clean up any existing resources
            self._cleanup_resources()
            
            # Initialize PyAudio
            self.pa = pyaudio.PyAudio()
            
            # Try to get device info if specified
            if device_index is not None:
                try:
                    device_info = self.pa.get_device_info_by_index(device_index)
                    print(f"Using microphone: {device_info.get('name')}")
                except:
                    device_index = None
                    print("Invalid device index, using default microphone.")
            
            # Open audio stream
            self.stream = self.pa.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=CHUNK
            )
            
            # Initialize audio buffer and queue
            self.audio_buffer = deque(maxlen=CHUNKS_PER_WINDOW)
            self.transcript_queue = deque(maxlen=10)  # Store recent transcripts
            self.start_time = time.time()
            self.saved_second = 0
            self.silence_threshold = silence_threshold
            self.speech_detected = False
            self.last_speech_time = 0
            
            # Start listener thread
            self.is_listening = True
            self.listener_thread = threading.Thread(
                target=self._audio_listener_thread,
                daemon=True
            )
            self.listener_thread.start()
            
            print("🎙️ STT Engine started listening...")
            return True
            
        except Exception as e:
            self.last_error = f"Error starting STT Engine: {str(e)}"
            print(f"*** {self.last_error}")
            self._cleanup_resources()
            self.is_listening = False
            return False
    
    def stop_listening(self) -> bool:
        """
        Stop listening for audio input.
        
        Returns:
            bool: True if stopped successfully, False otherwise
        """
        if not self.is_listening:
            return True  # Already stopped
            
        try:
            self.is_listening = False
            self._cleanup_resources()
            print("🛑 STT Engine stopped listening")
            return True
            
        except Exception as e:
            self.last_error = f"Error stopping STT Engine: {str(e)}"
            print(f"*** {self.last_error}")
            return False
    
    def _cleanup_resources(self) -> None:
        """Clean up PyAudio resources."""
        try:
            # Stop and close audio stream
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None
                
            # Terminate PyAudio
            if self.pa:
                self.pa.terminate()
                self.pa = None
        except Exception as e:
            print(f"Warning during cleanup: {str(e)}")
    
    def get_last_error(self) -> str:
        """Get the last error message."""
        return self.last_error
    
    def is_ready(self) -> bool:
        """Check if the STT engine is ready for use."""
        return self.is_initialized
        
    def get_recent_transcripts(self) -> List[str]:
        """
        Get recent transcripts from the queue.
        
        Returns:
            List[str]: List of recent transcripts
        """
        return list(self.transcript_queue)
    
    def _audio_listener_thread(self) -> None:
        """Main audio listener thread that captures and processes audio chunks."""
        saved_second = 0
        
        while self.is_listening:
            try:
                # Read audio chunk
                chunk = self.stream.read(CHUNK, exception_on_overflow=False)
                self.audio_buffer.append(chunk)
                
                # Check voice activity
                self._check_voice_activity(chunk)
                
                # Check if it's time to save and process a snapshot
                elapsed_time = time.time() - self.start_time
                
                # Process audio when:
                # 1. Regular interval snapshots OR
                # 2. Speech just ended (voice activity stopped)
                process_now = self._should_save_snapshot(elapsed_time, saved_second)
                
                if process_now:
                    saved_second = int(elapsed_time)
                    curr_file = self._save_audio_snapshot(saved_second)
                    
                    # Process in separate thread to not block audio capture
                    threading.Thread(
                        target=self._process_audio_file,
                        args=(curr_file,),
                        daemon=True
                    ).start()
                    
            except Exception as e:
                self.last_error = f"Error in audio listener: {str(e)}"
                print(f"⚠️ {self.last_error}")
                time.sleep(0.1)  # Prevent tight loop on error
                
    def _check_voice_activity(self, chunk: bytes) -> bool:
        """
        Check for voice activity in an audio chunk and update state.
        
        Args:
            chunk: Raw audio chunk data
            
        Returns:
            bool: True if a speech segment was processed due to speech ending
        """
        if not chunk:
            return False
            
        # Calculate RMS energy
        rms = audioop.rms(chunk, 2)  # 2 bytes/sample for 16-bit audio
        
        current_time = time.time()
        speech_end_processed = False
        
        # Check if speech is detected
        if rms >= self.silence_threshold:
            if not self.speech_detected:
                print(f"🗣️ Speech detected (RMS: {rms})")
                self.speech_detected = True
            self.last_speech_time = current_time
        elif self.speech_detected:
            # Check if speech has ended (silence for speech_timeout seconds)
            silence_duration = current_time - self.last_speech_time
            if silence_duration >= self.speech_timeout:
                print(f"🤐 Speech ended after {silence_duration:.2f}s silence")
                self.speech_detected = False
                
                # Speech end detection only tracks state, no processing
                
        return speech_end_processed
    
    def _save_audio_snapshot(self, current_second: int) -> str:
        """
        Save the current audio buffer to a .pcm file and return the file path.
        
        Args:
            current_second: Current second count for filename
            
        Returns:
            str: Path to the saved PCM file
        """
        # Create a unique filename with timestamp
        timestamp = int(time.time())
        filename = f"recorded_{timestamp}_{current_second}s.pcm"
        
        try:
            with open(filename, "wb") as f:
                f.writelines(self.audio_buffer)
            return filename
        except Exception as e:
            self.last_error = f"Error saving audio snapshot: {str(e)}"
            print(f"⚠️ {self.last_error}")
            return ""
    
    def _should_save_snapshot(self, elapsed_time: float, last_saved_second: int) -> bool:
        """
        Return True if the current elapsed time meets the conditions for saving a snapshot.
        
        Args:
            elapsed_time: Time elapsed since recording started
            last_saved_second: Last second when a snapshot was saved
            
        Returns:
            bool: True if a snapshot should be saved
        """
        current_second = int(elapsed_time)
        return (
            elapsed_time >= WINDOW_LENGTH_SECONDS and
            current_second % SAVE_INTERVAL_SECONDS == 0 and
            current_second != last_saved_second
        )
    
    def _is_silence(self, pcm_file_path: str) -> bool:
        """
        Determine if a raw PCM file is silent based on RMS energy.
        
        Args:
            pcm_file_path: Path to the PCM file
            
        Returns:
            bool: True if the audio is silent
        """
        try:
            with open(pcm_file_path, "rb") as f:
                pcm_data = f.read()
                
            if len(pcm_data) == 0:
                return True
            
            rms = audioop.rms(pcm_data, 2)  # 2 bytes/sample for 16-bit audio
            is_silent = rms < self.silence_threshold
            
            # Debug info to help adjust the threshold
            # print(f"Audio RMS: {rms}, Threshold: {self.silence_threshold}, Is silent: {is_silent}")
            
            return is_silent
            
        except Exception as e:
            self.last_error = f"Error in silence detection: {str(e)}"
            print(f"⚠️ {self.last_error}")
            return True
    
    def _is_english(self, text: str) -> bool:
        """
        Returns True if the text contains mostly English letters.
        
        Args:
            text: Text to check
            
        Returns:
            bool: True if the text is mostly English
        """
        # Count English words or alphabet characters
        english_chars = re.findall(r'[a-zA-Z]', text)
        total_chars = max(len(text.strip()), 1)
        return len(english_chars) / total_chars > 0.5  # more than 50% english
    
    def _transcribe_file(self, audio_file_name: str) -> str:
        """
        Transcribe a PCM audio file using the Whisper API.
        
        Args:
            audio_file_name: Path to the PCM file
            
        Returns:
            str: Transcribed text or empty string on error
        """
        if not self.whisper_available or not OPENAI_API_KEY:
            return ""
            
        try:
            # Read PCM data
            with open(audio_file_name, "rb") as f:
                pcm_data = f.read()
                
            # Convert to WAV format for API
            bio = BytesIO()
            with wave.open(bio, "wb") as wf:
                wf.setnchannels(CHANNELS)
                wf.setsampwidth(2)  # 16-bit = 2 bytes
                wf.setframerate(RATE)
                wf.writeframes(pcm_data)
            bio.seek(0)
            
            # Call Whisper API
            headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
            files = {"file": ("chunk.wav", bio, "audio/wav")}
            data = {"model": MODEL_NAME}
            
            response = httpx.post(
                API_URL, 
                headers=headers, 
                files=files, 
                data=data, 
                timeout=30.0
            )
            
            if response.status_code != 200:
                self.last_error = f"Whisper API error: {response.status_code} - {response.text}"
                print(f"⚠️ {self.last_error}")
                return ""
                
            result = response.json()
            return result.get("text", "").strip()
            
        except Exception as e:
            self.last_error = f"Error transcribing audio: {str(e)}"
            print(f"⚠️ {self.last_error}")
            return ""
    
    def _extract_new_segment(self, previous: str, current: str) -> str:
        """
        Extract only the new part of the transcript based on the previous transcript.
        
        Args:
            previous: Previous transcript
            current: Current transcript
            
        Returns:
            str: New segment of the transcript
        """
        if not previous or not current:
            return current
            
        def clean(text):
            return re.sub(r'[^\w\s]', '', text.lower())
            
        prev_clean = clean(previous).split()
        curr_clean = clean(current).split()
        curr_original = current.split()  # Keep original punctuation
        
        max_overlap = 0
        for i in range(len(prev_clean)):
            overlap_candidate = prev_clean[i:]
            if len(overlap_candidate) <= len(curr_clean) and \
               curr_clean[:len(overlap_candidate)] == overlap_candidate:
                max_overlap = len(overlap_candidate)
                
        new_words = curr_original[max_overlap:]
        return ' '.join(new_words)
    
    def _process_audio_file(self, curr_file: str) -> None:
        """
        Process an audio file for transcription.
        
        Args:
            curr_file: Path to the audio file
        """
        if not curr_file or not os.path.exists(curr_file):
            return
            
        try:
            # Check if we're in testing mode without API key
            if not self.whisper_available or not OPENAI_API_KEY:
                # Just notify that we're in testing mode without transcription
                if self.callback_on_transcript and not self.whisper_available:
                    self.callback_on_transcript("Microphone active but transcription disabled. Add OPENAI_API_KEY to enable.")
                # Clean up temp file and return
                self._cleanup_temp_file(curr_file)
                return
                
            # 1. Check for silence
            if self._is_silence(curr_file):
                curr_transcript = "<silence>"
            else:
                # 2. Transcribe the audio
                text = self._transcribe_file(curr_file).strip()
                if not text:
                    self._cleanup_temp_file(curr_file)
                    return
                    
                curr_transcript = text if text and self._is_english(text) else "<silence>"
                
            # 3. Extract new content if needed
            prev = self.state.prev_transcript
            if prev is None or prev == "<silence>" or curr_transcript == "<silence>":
                returned_transcript = curr_transcript
            else:
                new_segment = self._extract_new_segment(prev, curr_transcript)
                returned_transcript = new_segment
                
            # 4. Store results and notify callback only if we have meaningful new content
            if returned_transcript and returned_transcript != "<silence>":
                print(f"🗣️ Transcript: \"{returned_transcript}\"")
                
                # Check if this transcript is substantially different from the last one
                # to avoid duplicate processing
                is_duplicate = False
                if self.transcript_queue:
                    last_transcript = self.transcript_queue[-1]
                    # If the new transcript is just an expansion of the previous one
                    # (previous is contained within the new one), don't send a new callback
                    if last_transcript in returned_transcript:
                        is_duplicate = True
                        print(f"Skipping duplicate transcript expansion: \"{returned_transcript}\"")
                
                # Only add to queue and trigger callback if it's not a duplicate
                if not is_duplicate:
                    self.transcript_queue.append(returned_transcript)
                    
                    # Notify callback if provided
                    if self.callback_on_transcript:
                        self.callback_on_transcript(returned_transcript)
                    
            # 5. Update state for next time
            if curr_transcript != "<silence>":
                self.state.prev_transcript = curr_transcript
                
            # 6. Clean up temporary file
            self._cleanup_temp_file(curr_file)
            
        except Exception as e:
            self.last_error = f"Error processing audio file: {str(e)}"
            print(f"⚠️ {self.last_error}")
            self._cleanup_temp_file(curr_file)
    
    def _cleanup_temp_file(self, file_path: str) -> None:
        """Delete a temporary file and handle errors gracefully."""
        try:
            if file_path and os.path.exists(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Warning: Could not remove temp file {file_path}: {str(e)}")
    
    def shutdown(self) -> None:
        """Stop listening and release all resources."""
        self.stop_listening()

# Create a global instance for easy import
stt_engine = None

def initialize():
    """Initialize the global STT engine instance."""
    global stt_engine
    stt_engine = STTEngine()
    return stt_engine.is_ready()

# Initialize the engine when module is imported
initialize()