# TTS Module Documentation

## Overview

The `tts_engine.py` module provides text-to-speech functionality using the Coqui TTS library. It handles model initialization, audio synthesis, and queue management for sequential playback of text chunks.

## Installation Requirements

Before using the TTS module, ensure you have the following dependencies installed:

```
pip install TTS pydub torch
```

For audio playback, you'll also need:
- FFmpeg for file conversion (required by pydub)
- simpleaudio for sound playback on Windows

## Basic Usage

### 1. Importing the Module

The module automatically initializes a global instance when imported:

```python
import tts_engine

# Check if TTS is available and properly initialized
if tts_engine.tts_engine and tts_engine.tts_engine.is_ready():
    print("TTS engine is ready")
else:
    print("TTS engine not available")
```

### 2. Speaking Text

There are two primary ways to generate speech:

#### Queued Playback (Recommended)

```python
# This will queue text to be spoken and return immediately
# Text is automatically split into chunks by punctuation
tts_engine.tts_engine.speak_text("Hello! This is a test. How are you doing today?")
```

#### Immediate Playback

```python
# This will speak text immediately (blocking until complete)
tts_engine.tts_engine.speak_immediately("Hello there!")
```

### 3. Managing the Speech Queue

```python
# Clear all pending speech from the queue
tts_engine.tts_engine.clear_queue()

# Properly shut down the TTS engine when your application closes
tts_engine.tts_engine.shutdown()
```

## Advanced Features

### UI Callbacks

The TTS engine supports callbacks that notify your application when a chunk of text starts or ends being spoken:

```python
# Set callback functions
tts_engine.tts_engine.callback_on_chunk_start = lambda chunk: print(f"Started: {chunk}")
tts_engine.tts_engine.callback_on_chunk_end = lambda chunk: print(f"Finished: {chunk}")
```

These callbacks can be used to highlight the text currently being spoken, as demonstrated in the `ChatApp` class in `main.py`.

### Text Chunking

The TTS engine automatically splits text into smaller chunks based on punctuation:

```python
# Manually split text into chunks
chunks = tts_engine.tts_engine.chunk_text("Hello. This is a test! How are you?")
print(chunks)  # ['Hello.', 'This is a test!', 'How are you?']
```

### Error Handling

```python
# Check for errors
if not tts_engine.tts_engine.speak_text("Hello world"):
    print(f"Error: {tts_engine.tts_engine.get_last_error()}")
```

## Customization

When initializing the TTS engine, you can customize:

```python
# Create a custom TTS engine instance
custom_tts = tts_engine.TTSEngine(
    model_name="tts_models/en/ljspeech/tacotron2-DDC",  # TTS model to use
    device="cuda",  # Force GPU usage (or "cpu" for CPU only)
    callback_on_chunk_start=my_start_callback,
    callback_on_chunk_end=my_end_callback
)
```

## Integration with Applications

As shown in the `main.py` file, you can easily integrate TTS with GUI applications:

1. Initialize the TTS engine at startup
2. Set up callbacks for UI updates
3. Use `speak_text()` for asynchronous speech generation
4. Properly shut down the engine when closing the application

## Troubleshooting

- If the TTS engine fails to initialize, check that you have the Coqui TTS library installed
- The first time you run the application, it may download model files (this can take some time)
- For playback issues on Windows, ensure you have simpleaudio installed
- Errors during initialization are captured in `tts_engine.tts_engine.last_error`

## Example Usage in a Project

The TTS engine is already integrated with the chat application in `main.py`. You can see how it:
- Initializes at startup
- Uses callbacks to highlight spoken text
- Speaks AI responses automatically
- Cleanly shuts down when the application closes