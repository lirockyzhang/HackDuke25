# -*- coding: utf-8 -*-
"""
Local Conversational AI with Langchain, Gemini, Coqui TTS and OpenAI Whisper STT

This application provides a GUI interface for conversing with an AI assistant
using Gemini's language model, Coqui TTS for speech synthesis, and OpenAI Whisper
for speech recognition.

Requirements:
- langchain, langchain-google-genai
- pydub (with ffmpeg/simpleaudio for playback)
- customtkinter, python-dotenv
- TTS (Coqui TTS library - requires PyTorch)
- httpx, pyaudio (for STT functionality)
"""

# Standard library imports
import os
import threading
from dotenv import load_dotenv

# Third-party imports
import customtkinter as ctk
import torch
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_google_genai import ChatGoogleGenerativeAI

# Local imports
import tts_engine
import stt_engine


# --- Initialization ---
load_dotenv()
initialization_error = None

# Initialize AI components
try:
    # Check for API keys
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not found in environment variables")
    
    # Check for OpenAI API key for STT functionality
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("INFO: OPENAI_API_KEY not found. Microphone will work for testing but speech recognition is disabled.")
    else:
        print("OpenAI API key found, speech recognition enabled.")

    # Initialize AI components
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.7)
    memory = ConversationBufferMemory()
    conversation = ConversationChain(llm=llm, memory=memory, verbose=False)
    print("AI components initialized successfully")

except Exception as e:
    initialization_error = f"Initialization error: {e}\n\nPlease check API key and dependencies."
    print(f"\n{initialization_error}\n")


# --- GUI Application ---
class ChatApp(ctk.CTk):
    """Main application window for the conversational AI interface."""
    
    def __init__(self, conversation_chain):
        super().__init__()
        
        # Window setup
        self.title("Conversational AI with TTS and STT")
        self.geometry("700x550")
        
        # Handle initialization errors
        if initialization_error:
            self.label_status = ctk.CTkLabel(self, text=initialization_error, 
                                            text_color="red", wraplength=550)
            self.label_status.pack(pady=20, padx=20)
            return
            
        self.conversation_chain = conversation_chain
        self.is_listening = False
        
        # Configure layout
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=0)
        
        # Conversation display
        self.textbox_conversation = ctk.CTkTextbox(self, wrap="word", 
                                                 state="disabled", font=("Arial", 12))
        self.textbox_conversation.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="nsew")
        
        # Status label
        self.label_status = ctk.CTkLabel(self, text="Enter message below or click microphone to speak", height=20)
        self.label_status.grid(row=1, column=0, padx=20, pady=5, sticky="ew")
        
        # Input area
        self.input_frame = ctk.CTkFrame(self)
        self.input_frame.grid(row=2, column=0, padx=20, pady=(5, 20), sticky="ew")
        self.input_frame.grid_columnconfigure(0, weight=1)
        self.input_frame.grid_columnconfigure(1, weight=0)
        self.input_frame.grid_columnconfigure(2, weight=0)
        self.input_frame.grid_columnconfigure(3, weight=0)
        
        self.entry_message = ctk.CTkEntry(self.input_frame, placeholder_text="Type your message here...")
        self.entry_message.grid(row=0, column=0, padx=(0, 10), pady=5, sticky="ew")
        self.entry_message.bind("<Return>", self.handle_send_button)
        
        # Settings button for mic configuration
        self.button_settings = ctk.CTkButton(self.input_frame, text="⚙️", width=40,
                                           command=self.open_mic_settings)
        self.button_settings.grid(row=0, column=1, padx=(0, 10), pady=5)
        
        # Initialize settings button state based on STT availability
        self.stt_available = stt_engine.stt_engine and stt_engine.stt_engine.is_ready()
        if not self.stt_available:
            self.button_settings.configure(state="disabled")
        
        # Microphone button for speech input
        self.button_mic = ctk.CTkButton(self.input_frame, text="🎤", width=40,
                                      command=self.toggle_listening)
        self.button_mic.grid(row=0, column=2, padx=(0, 10), pady=5)
        
        # Initialize mic button state based on STT availability
        if not self.stt_available:
            self.button_mic.configure(state="disabled")
            
        # Send button
        self.button_send = ctk.CTkButton(self.input_frame, text="Send", width=80, 
                                       command=self.handle_send_button)
        self.button_send.grid(row=0, column=3, padx=(0, 0), pady=5)
        
        # Set up TTS callbacks
        self.tts_available = tts_engine.tts_engine and tts_engine.tts_engine.is_ready()
        if self.tts_available:
            tts_engine.tts_engine.callback_on_chunk_start = self.set_current_tts_chunk
            tts_engine.tts_engine.callback_on_chunk_end = lambda _: self.set_current_tts_chunk(None)
            tts_engine.tts_engine.callback_on_speech_complete = self.on_speech_complete
        
        # Set up STT callbacks
        if self.stt_available:
            stt_engine.stt_engine.callback_on_transcript = self.on_transcript_received
            
        # Speech tracking
        self.last_ai_response = ""
        self.actual_spoken_text = ""
        
        # Transcript accumulation to prevent double display
        self.pending_transcript = ""
        self.speech_in_progress = False
        self.speech_end_timer = None
        
        # --- Initial Greeting ---
        self.after(500, self.initial_greeting)
        
        # Start continuous listening by default if STT is available
        if self.stt_available:
            self.after(1000, self.start_continuous_listening)

        # Bind the close event
        self.protocol("WM_DELETE_WINDOW", self.on_app_close)
    
    def start_continuous_listening(self):
        """Start continuous listening mode automatically."""
        if self.stt_available:
            success = stt_engine.stt_engine.start_listening()
            if success:
                self.is_listening = True
                self.button_mic.configure(text="🎤", fg_color="#5cb85c")  # Green with mic icon
                self.set_status("Listening for speech. Click the microphone to mute if needed.")
            else:
                self.set_status(f"Could not start speech recognition: {stt_engine.stt_engine.get_last_error()}", "red")

    def update_conversation_display(self, sender, text, append=False):
        """Appends or updates text to the conversation history display."""
        self.textbox_conversation.configure(state="normal")
        if append:
            self.textbox_conversation.insert("end", text)
        else:
            self.textbox_conversation.insert("end", f"{sender}: {text}\n\n")
        self.textbox_conversation.configure(state="disabled")
        self.textbox_conversation.see("end")

    def set_status(self, message, color="white"):
        self.label_status.configure(text=message, text_color=color)
        self.update_idletasks()

    def set_current_tts_chunk(self, chunk):
        """Highlight the currently spoken chunk in the conversation display."""
        self.textbox_conversation.configure(state="normal")
        self.textbox_conversation.tag_remove("tts_current", "1.0", "end")
        if chunk:
            idx = self.textbox_conversation.search(chunk, "1.0", stopindex="end")
            if idx:
                end_idx = f"{idx}+{len(chunk)}c"
                self.textbox_conversation.tag_add("tts_current", idx, end_idx)
                self.textbox_conversation.tag_config("tts_current", background="#ffe066")
        self.textbox_conversation.configure(state="disabled")

    def initial_greeting(self):
        greeting = "Hello! How can I help you today?"
        self.update_conversation_display("AI", greeting)
        if tts_engine.tts_engine and tts_engine.tts_engine.is_ready():
            tts_engine.tts_engine.speak_text(greeting)
        else:
            self.set_status("TTS disabled. Enter message.", "orange")

    def handle_send_button(self, event=None):
        user_input = self.entry_message.get()
        if not user_input.strip():
            return

        self.entry_message.delete(0, "end")
        self.button_send.configure(state="disabled")
        self.entry_message.configure(state="disabled")
        self.set_status("Processing...")

        # Run processing in a separate thread
        thread = threading.Thread(target=self.process_text_input, args=(user_input,), daemon=True)
        thread.start()

    def process_text_input(self, user_input):
        self.after(0, self.update_conversation_display, "You", user_input)

        # Check for exit command
        if user_input.lower() in ["goodbye", "bye", "exit", "quit", "stop"]:
            ai_response = "Goodbye!"
            self.after(0, self.update_conversation_display, "AI", ai_response)
            self.after(0, self.set_status, "Exiting...")
            if tts_engine.tts_engine and tts_engine.tts_engine.is_ready():
                tts_engine.tts_engine.speak_text(ai_response)
            self.after(2000, self.destroy)
            return

        # Full LLM response (not streaming)
        try:
            self.after(0, self.set_status, "AI is responding...")
            ai_response = self.conversation_chain.predict(input=user_input)
            self.after(0, self.update_conversation_display, "AI", ai_response)
            self.last_ai_response = ai_response
            
            # Store a reference to self for the callback closure
            chat_app = self
            
            def on_this_speech_complete(spoken_text):
                """One-time callback for this specific speech request."""
                chat_app.actual_spoken_text = spoken_text
                # Handle what was actually spoken - this could feed into another part of the application
                print(f"AI actually spoke: {spoken_text}")
                
                # If speech was interrupted, you might want to handle that differently
                if "<|interrupt|>" in spoken_text:
                    print("Speech was interrupted before completion")
                    
                # You could now use spoken_text in another part of your application
                # For example, store it, analyze it, or feed it to another component
            
            if tts_engine.tts_engine and tts_engine.tts_engine.is_ready():
                # Pass the one-time callback for this specific speech request
                tts_engine.tts_engine.speak_text(ai_response, on_this_speech_complete)
            
            self.after(0, self.set_status, "Ready. Enter message.")

        except Exception as e:
            error_msg = f"Error during Langchain response: {e}"
            print(error_msg)
            self.after(0, self.set_status, "Error processing request.", "red")
            self.after(0, self.update_conversation_display, "System", error_msg)

        # Re-enable input elements
        self.after(0, lambda: self.button_send.configure(state="normal"))
        self.after(0, lambda: self.entry_message.configure(state="normal"))
        self.after(0, lambda: self.entry_message.focus())

    def toggle_listening(self):
        """Toggle between muted and unmuted states for speech input."""
        if not self.stt_available:
            return
            
        if self.is_listening:
            # Switch to muted state
            self.is_listening = False
            self.button_mic.configure(text="🔇", fg_color="#f0ad4e")  # Change to orange with mute icon
            self.set_status("Microphone muted. Click again to unmute.")
        else:
            # Switch to active listening state
            self.is_listening = True
            self.button_mic.configure(text="🎤", fg_color="#5cb85c")  # Change to green with mic icon
            self.set_status("Listening for speech...")
            
            # Start listening if not already running
            stt_engine.stt_engine.start_listening()
    
    def open_mic_settings(self):
        """Open the microphone settings dialog."""
        if not self.stt_available:
            self.set_status("Speech-to-text is not available.", "red")
            return
            
        # If currently listening, stop first
        was_listening = self.is_listening
        if was_listening:
            self.is_listening = False
            stt_engine.stt_engine.stop_listening()
            
        # Open the settings dialog
        settings_dialog = MicrophoneSettings(self, self.apply_mic_settings)
        settings_dialog.grab_set()  # Make the dialog modal
        
    def apply_mic_settings(self, device_index, threshold):
        """Apply the microphone settings from the dialog."""
        # Restart the STT engine with new settings
        if stt_engine.stt_engine:
            # Stop if running
            if stt_engine.stt_engine.is_listening:
                stt_engine.stt_engine.stop_listening()
                
            # Start with new settings
            success = stt_engine.stt_engine.start_listening(device_index, threshold)
            
            if success:
                self.is_listening = True
                self.button_mic.configure(text="🎤", fg_color="#5cb85c")
                
                # Get device name for status message
                device_name = "Default microphone"
                if device_index is not None:
                    try:
                        devices = stt_engine.stt_engine.list_microphones()
                        for device in devices:
                            if device['index'] == device_index:
                                device_name = device['name']
                                break
                    except:
                        pass
                
                self.set_status(f"Using {device_name} with threshold {threshold}")
            else:
                self.is_listening = False
                self.button_mic.configure(text="🔇", fg_color="#f0ad4e")
                self.set_status(f"Failed to start microphone: {stt_engine.stt_engine.get_last_error()}", "red")

    def on_transcript_received(self, transcript):
        """
        Handle the received transcript from STT engine.
        Accumulates speech segments and only processes when complete.
        """
        # Only process transcripts when listening is enabled
        if not self.is_listening:
            return
            
        # Skip silence markers
        if transcript == "<silence>":
            return
            
        # Start or continue speech
        self.speech_in_progress = True
        
        # Cancel any pending end-of-speech timer
        if self.speech_end_timer:
            self.after_cancel(self.speech_end_timer)
            self.speech_end_timer = None
        
        # Add this transcript to pending transcript if it's new content
        if transcript not in self.pending_transcript:
            if self.pending_transcript:
                self.pending_transcript += " " + transcript
            else:
                self.pending_transcript = transcript
            
            # For debugging
            print(f"Accumulating: \"{transcript}\"")
            print(f"Current pending transcript: \"{self.pending_transcript}\"")
        
        # Start a timer to detect the end of speech (1.5 seconds of silence)
        self.speech_end_timer = self.after(1500, self.on_speech_end)
    
    def on_speech_end(self):
        """
        Called when speech has ended (after a silence period).
        Processes the complete speech transcript.
        """
        if not self.pending_transcript:
            return
            
        # Get the complete transcript
        complete_transcript = self.pending_transcript
        
        # Reset speech tracking variables
        self.pending_transcript = ""
        self.speech_in_progress = False
        self.speech_end_timer = None
        
        # Log complete transcript for debugging
        print(f"Complete speech: \"{complete_transcript}\"")
        
        # Add the transcript to the conversation
        self.update_conversation_display("You", complete_transcript)
        
        # Process the transcript to get a response
        thread = threading.Thread(target=self.process_text_input, args=(complete_transcript,), daemon=True)
        thread.start()

    def on_speech_complete(self, spoken_text):
        """Handle completion of speech with the actual text that was spoken."""
        self.actual_spoken_text = spoken_text
        
        # Check if the speech was interrupted
        was_interrupted = "<|interrupt|>" in spoken_text
        
        # Log speech completion for debugging
        if was_interrupted:
            print(f"Speech was interrupted. Text spoken: {spoken_text.replace('<|interrupt|>', '')}")
        else:
            print(f"Speech completed successfully. Text spoken: {spoken_text}")
            
        # Could update UI to indicate speech completed/interrupted if needed
        # For example, you could show a notification or change a status indicator

    def on_app_close(self):
        """Clean up resources before closing the application."""
        # Shutdown the TTS engine
        if self.tts_available:
            tts_engine.tts_engine.shutdown()
            
        # Stop STT listening if active
        if self.stt_available and self.is_listening:
            stt_engine.stt_engine.stop_listening()
            
        self.destroy()

class MicrophoneSettings(ctk.CTkToplevel):
    """Dialog for selecting microphone and adjusting audio settings."""
    
    def __init__(self, parent, callback=None):
        super().__init__(parent)
        self.title("Microphone Settings")
        self.geometry("500x400")
        self.resizable(False, False)
        self.parent = parent
        self.callback = callback
        
        # Center the window
        self.update_idletasks()
        width = self.winfo_width()
        height = self.winfo_height()
        x = (self.winfo_screenwidth() // 2) - (width // 2)
        y = (self.winfo_screenheight() // 2) - (height // 2)
        self.geometry(f'{width}x{height}+{x}+{y}')
        
        # Create a frame for settings
        self.frame = ctk.CTkFrame(self)
        self.frame.pack(padx=20, pady=20, fill="both", expand=True)
        
        # Title
        self.title_label = ctk.CTkLabel(self.frame, text="Microphone Settings", font=("Arial", 18, "bold"))
        self.title_label.pack(pady=(10, 20))
        
        # Get all available microphones
        self.mics = []
        if stt_engine.stt_engine and stt_engine.stt_engine.is_ready():
            self.mics = stt_engine.stt_engine.list_microphones()
        
        # Microphone selection
        self.mic_label = ctk.CTkLabel(self.frame, text="Select Microphone:", anchor="w")
        self.mic_label.pack(padx=20, pady=(5, 0), fill="x")
        
        # Create the mic options strings
        mic_options = ["Default Microphone"]
        if self.mics:
            for mic in self.mics:
                mic_options.append(f"{mic['name']} (index: {mic['index']})")
        
        self.mic_var = ctk.StringVar(value=mic_options[0])
        self.mic_dropdown = ctk.CTkOptionMenu(self.frame, variable=self.mic_var, values=mic_options)
        self.mic_dropdown.pack(padx=20, pady=(5, 15), fill="x")
        
        # Silence threshold slider
        self.threshold_label = ctk.CTkLabel(self.frame, text=f"Silence Threshold: 300", anchor="w")
        self.threshold_label.pack(padx=20, pady=(10, 0), fill="x")
        
        self.threshold_slider = ctk.CTkSlider(self.frame, from_=100, to=1000, number_of_steps=90)
        self.threshold_slider.set(300)  # Default value
        self.threshold_slider.pack(padx=20, pady=(5, 15), fill="x")
        self.threshold_slider.configure(command=self.update_threshold_label)
        
        # Info section
        self.info_frame = ctk.CTkFrame(self.frame)
        self.info_frame.pack(padx=20, pady=(10, 15), fill="x")
        
        self.info_label = ctk.CTkLabel(
            self.info_frame, 
            text="Lower threshold = more sensitive (detects softer sounds)\nHigher threshold = less sensitive (needs louder sounds)",
            justify="left",
            wraplength=400
        )
        self.info_label.pack(padx=10, pady=10, fill="x")
        
        # Test microphone button
        self.test_button = ctk.CTkButton(
            self.frame, 
            text="Test Microphone", 
            command=self.test_microphone
        )
        self.test_button.pack(padx=20, pady=(5, 15))
        
        # Status label for test results
        self.status_label = ctk.CTkLabel(self.frame, text="", text_color="gray")
        self.status_label.pack(padx=20, pady=(0, 15), fill="x")
        
        # Apply button
        self.apply_button = ctk.CTkButton(
            self.frame, 
            text="Apply Settings", 
            command=self.apply_settings
        )
        self.apply_button.pack(padx=20, pady=(10, 20))
    
    def update_threshold_label(self, value):
        """Update the threshold label when slider is moved."""
        threshold = int(value)
        self.threshold_label.configure(text=f"Silence Threshold: {threshold}")
    
    def get_selected_device_index(self):
        """Get the device index from the selected option."""
        selected = self.mic_var.get()
        if selected == "Default Microphone":
            return None
        
        # Extract index from the option string
        try:
            index_str = selected.split("index: ")[1].rstrip(")")
            return int(index_str)
        except:
            return None
    
    def test_microphone(self):
        """Test the selected microphone with current settings."""
        self.status_label.configure(text="Testing microphone...", text_color="blue")
        
        # Get current settings
        device_index = self.get_selected_device_index()
        threshold = int(self.threshold_slider.get())
        
        # Create a temporary PyAudio instance to test
        try:
            pa = pyaudio.PyAudio()
            
            # Try to open stream with selected device
            if device_index is not None:
                try:
                    device_info = pa.get_device_info_by_index(device_index)
                    self.status_label.configure(
                        text=f"Using: {device_info.get('name')}\nSample rate: {int(device_info.get('defaultSampleRate'))}Hz",
                        text_color="green"
                    )
                except:
                    self.status_label.configure(
                        text="Invalid device selection. Using default.",
                        text_color="orange"
                    )
                    device_index = None
            
            # Try to open the stream
            stream = pa.open(
                format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=CHUNK
            )
            
            # Read a few chunks to check audio level
            audio_data = b''
            for _ in range(10):  # Capture ~0.5 seconds
                chunk = stream.read(CHUNK, exception_on_overflow=False)
                audio_data += chunk
            
            # Calculate RMS and compare to threshold
            if len(audio_data) > 0:
                rms = audioop.rms(audio_data, 2)
                self.status_label.configure(
                    text=f"Audio level: {rms}\nThreshold: {threshold}\nStatus: {'Active' if rms >= threshold else 'Silent'}",
                    text_color="green" if rms >= threshold else "orange"
                )
            
            # Clean up
            stream.stop_stream()
            stream.close()
            pa.terminate()
            
        except Exception as e:
            self.status_label.configure(
                text=f"Error testing microphone: {str(e)}",
                text_color="red"
            )
    
    def apply_settings(self):
        """Apply the selected settings and close the dialog."""
        device_index = self.get_selected_device_index()
        threshold = int(self.threshold_slider.get())
        
        if self.callback:
            self.callback(device_index, threshold)
        
        self.destroy()

# --- Main Execution ---
if __name__ == "__main__":
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme("blue")

    app = ChatApp(conversation)
    if not initialization_error:
        app.mainloop()
    else:
        print("\nApplication cannot start due to initialization errors.")
        root = ctk.CTk()
        root.title("Initialization Error")
        root.geometry("600x150")
        label = ctk.CTkLabel(root, text=initialization_error, text_color="red", wraplength=550)
        label.pack(pady=20, padx=20, expand=True, fill="both")
        root.mainloop()