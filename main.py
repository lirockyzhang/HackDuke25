# -*- coding: utf-8 -*-
"""
Local Conversational AI with Langchain, Gemini, Memory, Text Input, Coqui TTS, and UI.

This script runs a conversational AI locally using text input via a GUI,
and Coqui TTS for Text-to-Speech output.

Requirements:
- Python 3.8+
- langchain
- langchain-google-genai
- pydub
# Removed: SpeechRecognition
# Removed: PyAudio
- sounddevice (Often used by pydub/simpleaudio for playback) OR ffmpeg (for ffplay backend)
- customtkinter (for the GUI)
- python-dotenv
- TTS (Coqui TTS library - Requires PyTorch)

***************************************************************************
*** LOCAL SETUP STEPS ***
***************************************************************************

1. Install PyTorch: Coqui TTS relies on PyTorch. Follow instructions at
   https://pytorch.org/get-started/locally/ based on your OS and CUDA version (if applicable).

2. Install required Python libraries:
   pip install langchain langchain-google-genai pydub sounddevice customtkinter python-dotenv TTS google-cloud-aiplatform --upgrade
   (Note: SpeechRecognition & PyAudio are removed)

3. Install system dependencies:
   # Removed: PyAudio/PortAudio dependencies
   - Pydub Playback Backend: Pydub needs a backend to play audio.
     - Option A (Recommended): Install ffmpeg (https://ffmpeg.org/download.html) and ensure ffplay is in your system's PATH.
     - Option B: Install simpleaudio (`pip install simpleaudio`) - sounddevice is often a dependency.
   - Coqui TTS Dependencies: May require `espeak` or `espeak-ng`.
     - Debian/Ubuntu: sudo apt-get install espeak-ng
     - macOS: brew install espeak
     - Windows: Download and install espeak.

4. Create a .env file in the same directory as this script with your API key:
   GOOGLE_API_KEY="YOUR_GOOGLE_AI_API_KEY"

Running the script:
   python your_script_name.py
   (Note: The first run might take longer as Coqui TTS downloads the specified model)

***************************************************************************
"""

import os
import io
import time
import threading # To prevent blocking the UI during API calls/audio processing
# Removed: import speech_recognition as sr
from pydub import AudioSegment
from pydub.playback import play as play_with_pydub
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from TTS.api import TTS # Import Coqui TTS
import customtkinter as ctk # For the GUI
import tempfile # For temporary TTS audio file
import torch # Coqui TTS uses PyTorch

# --- Configuration & Initialization ---
load_dotenv()

# --- Global AI Components (Initialize early, handle errors) ---
llm = None
memory = None
conversation = None
coqui_tts_model = None
# Removed: recognizer = None
# Removed: microphone = None

initialization_error = None # Store any critical initialization error

try:
    # --- Environment Variable Checks ---
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in your .env file.")

    # --- Initialize AI Components ---
    print("Initializing AI components...")

    # 1. Language Model (LLM)
    print("Initializing LLM...")
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash-001", temperature=0.7)
    print("LLM initialized.")

    # 2. Memory
    print("Initializing Memory...")
    memory = ConversationBufferMemory()
    print("Memory initialized.")

    # 3. Conversation Chain
    print("Initializing Conversation Chain...")
    conversation = ConversationChain(llm=llm, memory=memory, verbose=False)
    print("Conversation Chain initialized.")

    # 4. Coqui Text-to-Speech Model
    print("Initializing Coqui TTS model...")
    print("(This might download model files on the first run)")
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        model_name = "tts_models/en/ljspeech/tacotron2-DDC"
        coqui_tts_model = TTS(model_name=model_name, progress_bar=True).to(device)
        print(f"Coqui TTS model '{model_name}' initialized on {device}.")
    except ImportError as e:
         print(f"*** Coqui TTS Initialization Failed: Missing Dependency: {e}")
         print("Ensure PyTorch and the TTS library are installed correctly.")
         coqui_tts_model = None
         print("WARNING: Proceeding without Text-to-Speech capabilities.")
    except Exception as e:
        import traceback
        print(f"*** Error initializing Coqui TTS model: {e}")
        print(traceback.format_exc())
        print("Ensure the model name is correct and dependencies (like espeak) are installed.")
        coqui_tts_model = None
        print("WARNING: Proceeding without Text-to-Speech capabilities.")

    # 5. Speech-to-Text Recognizer & Microphone (REMOVED)
    print("STT components removed.")

    print("AI components initialized successfully (excluding STT).")

except Exception as e:
    initialization_error = f"Critical initialization error: {e}\n\nPlease check setup steps and environment variables."
    print(f"\n{initialization_error}\n")

# --- Core Functions ---

def speak_text_thread(text):
    """Target function for running Coqui TTS synthesis and playback in a separate thread."""
    # (Function remains the same as previous version)
    if not coqui_tts_model:
        print("Coqui TTS Model not available. Skipping speech output.")
        return
    if not text:
        print("No text provided to speak.")
        return

    print(f"AI Speaking (Coqui TTS thread): {text}")
    temp_wav_path = None # Ensure variable exists for finally block
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as fp:
            temp_wav_path = fp.name
            print(f"Synthesizing to temporary file: {temp_wav_path}")
            coqui_tts_model.tts_to_file(text=text, file_path=temp_wav_path)
            print("Synthesis complete.")

        print("Loading and playing audio...")
        sound = AudioSegment.from_wav(temp_wav_path)
        play_with_pydub(sound)
        print("Audio finished.")

    except ImportError:
        print("Playback failed: Ensure a pydub backend is installed (e.g., ffmpeg or simpleaudio).")
    except RuntimeError as e:
         print(f"Error during Coqui TTS synthesis: {e}")
    except Exception as e:
        import traceback
        print(f"Error during Coqui TTS synthesis or playback: {e}")
        print(traceback.format_exc())
    finally:
        if temp_wav_path and os.path.exists(temp_wav_path):
            try:
                os.remove(temp_wav_path)
                print(f"Temporary file removed: {temp_wav_path}")
            except Exception as e:
                print(f"Error removing temporary file {temp_wav_path}: {e}")

# Removed: listen_and_transcribe function

# --- GUI Application ---

class ChatApp(ctk.CTk):
    def __init__(self, conversation_chain):
        super().__init__()

        self.title("Conversational AI (Text Input + Coqui TTS)") # Updated title
        self.geometry("600x500")

        if initialization_error:
            self.label_status = ctk.CTkLabel(self, text=initialization_error, text_color="red", wraplength=550)
            self.label_status.pack(pady=20, padx=20)
            return

        self.conversation_chain = conversation_chain

        # --- Configure grid layout ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1) # Text area row
        self.grid_rowconfigure(1, weight=0) # Status label row
        self.grid_rowconfigure(2, weight=0) # Input row

        # --- Widgets ---
        self.textbox_conversation = ctk.CTkTextbox(self, wrap="word", state="disabled", font=("Arial", 12))
        self.textbox_conversation.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="nsew")

        self.label_status = ctk.CTkLabel(self, text="Enter message below and click Send", height=20)
        self.label_status.grid(row=1, column=0, padx=20, pady=5, sticky="ew")

        # --- Input Frame ---
        self.input_frame = ctk.CTkFrame(self)
        self.input_frame.grid(row=2, column=0, padx=20, pady=(5, 20), sticky="ew")
        self.input_frame.grid_columnconfigure(0, weight=1)
        self.input_frame.grid_columnconfigure(1, weight=0)

        self.entry_message = ctk.CTkEntry(self.input_frame, placeholder_text="Type your message here...")
        self.entry_message.grid(row=0, column=0, padx=(0, 10), pady=5, sticky="ew")
        # Bind Enter key to send message
        self.entry_message.bind("<Return>", self.handle_send_button)

        self.button_send = ctk.CTkButton(self.input_frame, text="Send", width=80, command=self.handle_send_button)
        self.button_send.grid(row=0, column=1, padx=(0,0), pady=5, sticky="e")

        # --- Initial Greeting ---
        self.after(500, self.initial_greeting)

    def update_conversation_display(self, sender, text):
        """Appends text to the conversation history display."""
        self.textbox_conversation.configure(state="normal")
        self.textbox_conversation.insert("end", f"{sender}: {text}\n\n")
        self.textbox_conversation.configure(state="disabled")
        self.textbox_conversation.see("end")

    def set_status(self, message, color="white"):
        """Updates the status label."""
        self.label_status.configure(text=message, text_color=color)
        self.update_idletasks()

    def initial_greeting(self):
        """Plays the initial greeting using Coqui TTS."""
        greeting = "Hello! How can I help you today?"
        self.update_conversation_display("AI", greeting)
        if coqui_tts_model:
            tts_thread = threading.Thread(target=speak_text_thread, args=(greeting,), daemon=True)
            tts_thread.start()
        else:
            self.set_status("TTS disabled. Enter message.", "orange")

    def handle_send_button(self, event=None): # Added event=None for Enter key binding
        """Handles the 'Send' button click or Enter key press."""
        user_input = self.entry_message.get()
        if not user_input.strip(): # Ignore empty input
            return

        self.entry_message.delete(0, "end") # Clear input field
        self.button_send.configure(state="disabled") # Disable button during processing
        self.entry_message.configure(state="disabled") # Disable entry during processing
        self.set_status("Processing...")

        # Run processing in a separate thread
        thread = threading.Thread(target=self.process_text_input, args=(user_input,), daemon=True)
        thread.start()

    def process_text_input(self, user_input):
        """The core logic run in a thread after clicking 'Send'."""
        # Update UI with user input (from main thread via 'after')
        self.after(0, self.update_conversation_display, "You", user_input)

        # Check for exit command
        if user_input.lower() in ["goodbye", "bye", "exit", "quit", "stop"]:
            ai_response = "Goodbye!"
            self.after(0, self.update_conversation_display, "AI", ai_response)
            self.after(0, self.set_status, "Exiting...")
            if coqui_tts_model:
                 speak_text_thread(ai_response)
            self.after(2000, self.destroy)
            return

        # Get AI response
        try:
            ai_response = self.conversation_chain.predict(input=user_input)
            self.after(0, self.update_conversation_display, "AI", ai_response)
            self.after(0, self.set_status, "Speaking (Coqui TTS)...")

            # Speak the response
            if coqui_tts_model:
                speak_text_thread(ai_response)

            self.after(0, self.set_status, "Ready. Enter message.")

        except Exception as e:
            error_msg = f"Error during Langchain processing: {e}"
            print(error_msg)
            self.after(0, self.set_status, "Error processing request.", "red")
            self.after(0, self.update_conversation_display, "System", error_msg)

        # Re-enable input elements (always do this in the 'after' call)
        self.after(0, lambda: self.button_send.configure(state="normal"))
        self.after(0, lambda: self.entry_message.configure(state="normal"))
        self.after(0, lambda: self.entry_message.focus()) # Set focus back to entry field


# --- Main Execution ---
if __name__ == "__main__":
    ctk.set_appearance_mode("system")
    ctk.set_default_color_theme("blue")

    if not initialization_error:
        if coqui_tts_model is None:
             print("\nWARNING: Coqui TTS model failed to initialize. TTS features will be disabled.")

        app = ChatApp(conversation)
        if not initialization_error:
             app.mainloop()
        else:
             print("\nApplication cannot start due to initialization errors.")
    else:
        print("\nApplication cannot start due to initialization errors.")
        root = ctk.CTk()
        root.title("Initialization Error")
        root.geometry("600x150")
        label = ctk.CTkLabel(root, text=initialization_error, text_color="red", wraplength=550)
        label.pack(pady=20, padx=20, expand=True, fill="both")
        root.mainloop()