#!/usr/bin/env python3
"""
TTS Demo Application

This console application demonstrates the capabilities of the TTS engine
with interactive examples of different speech patterns and features.
"""

import os
import time
import threading
import tts_engine

def clear_screen():
    """Clear the console screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

class TTSDemo:
    """Interactive demo for the TTS Engine."""
    
    def __init__(self):
        """Initialize the TTS Demo application."""
        self.running = True
        self.tts = tts_engine.tts_engine
        
        # Sample texts for demonstration
        self.short_text = "Hello! This is a short sentence."
        self.medium_text = "This is a medium length text with multiple sentences. It contains several phrases. Each should be spoken separately."
        self.long_text = ("Today is a beautiful day for a demonstration of text-to-speech capabilities. "
                         "We will explore various features of the TTS engine. "
                         "Including queued playback, immediate speech, and interruption handling. "
                         "Pay attention to how chunks are processed sequentially. "
                         "Notice the difference between speak_text and speak_immediately methods!")
        
        # Setup callbacks for speech events
        self.tts.callback_on_chunk_start = self.on_speech_start
        self.tts.callback_on_chunk_end = self.on_speech_end
        
    def on_speech_start(self, chunk):
        """Callback when a speech chunk starts."""
        print(f"\n► SPEAKING: \"{chunk}\"")
        
    def on_speech_end(self, chunk):
        """Callback when a speech chunk ends."""
        print(f"✓ FINISHED: \"{chunk}\"")
    
    def display_menu(self):
        """Display the main menu options."""
        clear_screen()
        print("=" * 60)
        print("               TTS ENGINE INTERACTIVE DEMO")
        print("=" * 60)
        print("1. Speak Text (Queued) - Short")
        print("2. Speak Text (Queued) - Medium")
        print("3. Speak Text (Queued) - Long")
        print("4. Speak Immediately (Blocking)")
        print("5. Demonstrate Interruption (Queue a long text then interrupt)")
        print("6. Compare speak_text vs speak_immediately")
        print("7. Quit")
        print("-" * 60)
        
    def demo_queued_speech(self, text):
        """Demonstrate queued speech with the given text."""
        print(f"\nQueueing text for speech...\n\"{text}\"\n")
        success = self.tts.speak_text(text)
        
        if not success:
            print(f"Error: {self.tts.get_last_error()}")
            return
            
        print("\nText queued successfully.")
        print("Notice how the speech is processed in chunks and callbacks are triggered.")
        print("You can continue using the program while speech plays in the background.")
        input("\nPress Enter to return to the menu...")
        
    def demo_immediate_speech(self):
        """Demonstrate immediate (blocking) speech."""
        text = "This is immediate speech. It blocks the program until completed."
        print(f"\nSpeaking immediately (blocking mode)...\n\"{text}\"\n")
        
        # Clear any ongoing speech first
        print("Clearing any ongoing speech...")
        self.tts.clear_queue()
        
        # Note: No callbacks are triggered with speak_immediately
        print("► SPEAKING IMMEDIATELY (no callbacks)")
        start_time = time.time()
        success = self.tts.speak_immediately(text)
        elapsed = time.time() - start_time
        
        if not success:
            print(f"Error: {self.tts.get_last_error()}")
            return
            
        print(f"✓ COMPLETED in {elapsed:.1f} seconds")
        print("\nNotice that the program was blocked during speech.")
        print("No input was possible until speech completed.")
        input("\nPress Enter to return to the menu...")
        
    def demo_interrupt(self):
        """Demonstrate interrupting speech with new speech."""
        print("\nStarting a long speech that will be interrupted...")
        
        # Queue a long text
        self.tts.speak_text(self.long_text)
        print("\nLong text has been queued and is playing...")
        print("Waiting 3 seconds before interrupting...")
        
        # Wait a moment before interrupting
        time.sleep(3)
        
        # Now interrupt with a new text
        print("\n>> INTERRUPTING with a new speech <<")
        self.tts.clear_queue()  # Clear the queue first
        self.tts.speak_text("I have interrupted the previous speech! This is the new speech now.")
        
        input("\nPress Enter to return to the menu...")
        
    def demo_compare(self):
        """Compare speak_text vs speak_immediately side by side."""
        clear_screen()
        print("=" * 60)
        print("      COMPARING speak_text vs speak_immediately")
        print("=" * 60)
        
        print("\n1. speak_text:")
        print("   - Non-blocking: Program continues execution")
        print("   - Queued: Multiple speak_text calls stack up")
        print("   - Chunked: Text is split by punctuation")
        print("   - Callbacks: Triggers start/end callbacks")
        
        print("\n2. speak_immediately:")
        print("   - Blocking: Program waits until speech completes")
        print("   - Immediate: Bypasses the queue")
        print("   - Whole text: Doesn't chunk the text")
        print("   - No callbacks: Doesn't trigger callbacks")
        
        print("\nDemonstrating speak_text...")
        self.tts.speak_text("This is speak_text. It is non-blocking. You can see this message while I'm still talking.")
        print("See? The program continues while speech happens in the background.")
        
        print("\nWaiting for speech to complete...")
        time.sleep(3)
        
        print("\nNow demonstrating speak_immediately...")
        print("The program will pause until speech completes...")
        self.tts.speak_immediately("This is speak immediately. Notice how it blocks program execution until I finish speaking.")
        print("Now the speak_immediately call has completed and the program continues.")
        
        input("\nPress Enter to return to the menu...")
    
    def run(self):
        """Run the main demo loop."""
        if not self.tts or not self.tts.is_ready():
            print("ERROR: TTS Engine is not available or initialized.")
            print(f"Last error: {self.tts.get_last_error() if self.tts else 'TTS not available'}")
            print("Please ensure Coqui TTS is installed and try again.")
            return
            
        while self.running:
            self.display_menu()
            choice = input("Enter your choice (1-7): ").strip()
            
            if choice == '1':
                self.demo_queued_speech(self.short_text)
            elif choice == '2':
                self.demo_queued_speech(self.medium_text)
            elif choice == '3':
                self.demo_queued_speech(self.long_text)
            elif choice == '4':
                self.demo_immediate_speech()
            elif choice == '5':
                self.demo_interrupt()
            elif choice == '6':
                self.demo_compare()
            elif choice == '7':
                self.running = False
                print("\nShutting down TTS engine...")
                self.tts.shutdown()
                print("Thank you for using the TTS Demo!")
            else:
                input("Invalid choice. Press Enter to try again...")

if __name__ == "__main__":
    demo = TTSDemo()
    demo.run()