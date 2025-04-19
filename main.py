# -*- coding: utf-8 -*-
"""
Local Conversational AI with Langchain, Gemini, and Coqui TTS

This application provides a GUI interface for conversing with an AI assistant
using Gemini's language model and Coqui TTS for speech synthesis.

Requirements:
- langchain, langchain-google-genai
- pydub (with ffmpeg/simpleaudio for playback)
- customtkinter, python-dotenv
- TTS (Coqui TTS library - requires PyTorch)
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


# --- Initialization ---
load_dotenv()
initialization_error = None

# Initialize AI components
try:
    # Check for API key
    if not os.getenv("GOOGLE_API_KEY"):
        raise ValueError("GOOGLE_API_KEY not found in environment variables")

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
        self.title("Conversational AI with TTS")
        self.geometry("600x500")
        
        # Handle initialization errors
        if initialization_error:
            self.label_status = ctk.CTkLabel(self, text=initialization_error, 
                                            text_color="red", wraplength=550)
            self.label_status.pack(pady=20, padx=20)
            return
            
        self.conversation_chain = conversation_chain
        
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
        self.label_status = ctk.CTkLabel(self, text="Enter message below and click Send", height=20)
        self.label_status.grid(row=1, column=0, padx=20, pady=5, sticky="ew")
        
        # Input area
        self.input_frame = ctk.CTkFrame(self)
        self.input_frame.grid(row=2, column=0, padx=20, pady=(5, 20), sticky="ew")
        self.input_frame.grid_columnconfigure(0, weight=1)
        self.input_frame.grid_columnconfigure(1, weight=0)
        
        self.entry_message = ctk.CTkEntry(self.input_frame, placeholder_text="Type your message here...")
        self.entry_message.grid(row=0, column=0, padx=(0, 10), pady=5, sticky="ew")
        self.entry_message.bind("<Return>", self.handle_send_button)
        
        self.button_send = ctk.CTkButton(self.input_frame, text="Send", width=80, 
                                        command=self.handle_send_button)
        self.button_send.grid(row=0, column=1, padx=(0, 0), pady=5, sticky="e")
        
        # Set up TTS callbacks
        if tts_engine.tts_engine and tts_engine.tts_engine.is_ready():
            tts_engine.tts_engine.callback_on_chunk_start = self.set_current_tts_chunk
            tts_engine.tts_engine.callback_on_chunk_end = lambda _: self.set_current_tts_chunk(None)
        

        self.title("Conversational AI (Text Input + Coqui TTS)")
        self.geometry("600x500")

        if initialization_error:
            self.label_status = ctk.CTkLabel(self, text=initialization_error, text_color="red", wraplength=550)
            self.label_status.pack(pady=20, padx=20)
            return

        self.conversation_chain = conversation_chain

        # --- Configure grid layout ---
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_rowconfigure(2, weight=0)

        # --- Widgets ---
        self.textbox_conversation = ctk.CTkTextbox(self, wrap="word", state="disabled", font=("Arial", 12))
        self.textbox_conversation.grid(row=0, column=0, padx=20, pady=(20, 10), sticky="nsew")

        self.label_status = ctk.CTkLabel(self, text="Enter message below and click Send", height=20)
        self.label_status.grid(row=1, column=0, padx=20, pady=5, sticky="ew")

        self.input_frame = ctk.CTkFrame(self)
        self.input_frame.grid(row=2, column=0, padx=20, pady=(5, 20), sticky="ew")
        self.input_frame.grid_columnconfigure(0, weight=1)
        self.input_frame.grid_columnconfigure(1, weight=0)

        self.entry_message = ctk.CTkEntry(self.input_frame, placeholder_text="Type your message here...")
        self.entry_message.grid(row=0, column=0, padx=(0, 10), pady=5, sticky="ew")
        self.entry_message.bind("<Return>", self.handle_send_button)

        self.button_send = ctk.CTkButton(self.input_frame, text="Send", width=80, command=self.handle_send_button)
        self.button_send.grid(row=0, column=1, padx=(0,0), pady=5, sticky="e")

        # --- Set up TTS callback functions ---
        if tts_engine.tts_engine and tts_engine.tts_engine.is_ready():
            tts_engine.tts_engine.callback_on_chunk_start = self.set_current_tts_chunk
            tts_engine.tts_engine.callback_on_chunk_end = lambda _: self.set_current_tts_chunk(None)
        
        # --- Initial Greeting ---
        self.after(500, self.initial_greeting)

        # Bind the close event
        self.protocol("WM_DELETE_WINDOW", self.on_app_close)

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
            if tts_engine.tts_engine and tts_engine.tts_engine.is_ready():
                tts_engine.tts_engine.speak_text(ai_response)
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

    def on_app_close(self):
        # Shutdown the TTS engine properly
        if tts_engine.tts_engine and tts_engine.tts_engine.is_ready():
            tts_engine.tts_engine.shutdown()
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