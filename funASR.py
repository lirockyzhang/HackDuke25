import funasr
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import sounddevice as sd
import numpy as np
import sys
import traceback
import time # To add slight delay if needed

# --- Configuration ---
TARGET_SAMPLE_RATE = 16000 # Sample rate expected by the models
VAD_MODEL_NAME = "fsmn-vad" # VAD model for endpointing
ASR_MODEL_DIR = "iic/SenseVoiceSmall" # ASR model for processing segments
ASR_DEVICE = "cuda:0" # Device for the ASR model ("cuda:0" or "cpu")
VAD_CHUNK_SIZE_MS = 500 # VAD model processing chunk size in milliseconds (used for reading mic)

# ASR processing parameters (from example)
ASR_LANGUAGE = "en" # Or specify "en", "zh", etc.
ASR_USE_ITN = True
ASR_BATCH_SIZE_S = 60 # Might be less relevant for single segments
ASR_MERGE_VAD = True # Use ASR's internal VAD for merging/cleanup within segment
ASR_MERGE_LENGTH_S = 15 # Might be less relevant for single segments

# --- Initialize Models ---
print(f"Initializing VAD model: {VAD_MODEL_NAME}...")
try:
    vad_model = AutoModel(model=VAD_MODEL_NAME,
                          disable_update=True,
                          disable_pbar=True)
    print("VAD model initialized.")
except Exception as e:
    print(f"Error initializing VAD model: {e}")
    traceback.print_exc()
    exit()

print(f"Initializing ASR model: {ASR_MODEL_DIR} on device {ASR_DEVICE}...")
try:
    asr_model = AutoModel(model=ASR_MODEL_DIR,
                          device=ASR_DEVICE,
                          disable_update=True,
                          disable_pbar=True)
    print("ASR model initialized.")
except Exception as e:
    print(f"Error initializing ASR model: {e}")
    traceback.print_exc()
    exit()


# --- Audio Stream Configuration ---
vad_block_size = int(TARGET_SAMPLE_RATE * VAD_CHUNK_SIZE_MS / 1000) # Samples per VAD chunk read from mic
print(f"Audio Settings: SampleRate={TARGET_SAMPLE_RATE}, Mic Read BlockSize={vad_block_size} samples ({VAD_CHUNK_SIZE_MS}ms)")

# --- Real-time Processing Loop ---
vad_cache = {} # Cache for the VAD model state (Usage might be incorrect in this logic)
audio_buffer = [] # List to store numpy chunks for the current speech segment

print("\n--- Starting VAD (Full Buffer Input) + ASR Pipeline ---")
print(f"Listening... Processing full buffer ({VAD_CHUNK_SIZE_MS}ms mic chunks)... Press Ctrl+C to stop.")
print("WARNING: This method processes the entire buffer repeatedly and may be inefficient.")

try:
    with sd.InputStream(samplerate=TARGET_SAMPLE_RATE,
                        blocksize=vad_block_size,
                        channels=1,
                        dtype='float32',
                        device=None) as stream:
        """
        Implementation of a real-time VAD + ASR pipeline using FunASR models.
        When silent, i.e. no speech detected, the buffer is cleared and the VAD model is reset.
        When start of speech is detected and no end of speech is detected, the buffer is filled with audio chunks.
        """


        vad_startpoint_detected = False # Flag to indicate if VAD has started
        vad_endpoint_detected = False
        cache_vad={}
        cache_asr={}
        while True:
            if not vad_startpoint_detected or vad_endpoint_detected:
                audio_buffer = [] # Reset buffer if no speech detected or endpoint reached
                vad_endpoint_detected = False
                cache_vad={}

            if len(audio_buffer) > 0:
                print(f"Buffer Length: {len(audio_buffer)} chunks ({len(audio_buffer) * vad_block_size / TARGET_SAMPLE_RATE:.2f}s)")
            # 1. Read audio chunk from microphone
            audio_chunk, overflowed = stream.read(vad_block_size)
            if overflowed:
                print("Warning: Audio input overflowed!", file=sys.stderr)
            audio_chunk_np = audio_chunk.flatten()

            # 2. Append chunk to the buffer
            audio_buffer.append(audio_chunk_np.copy())

            # 3. Concatenate ALL chunks in the buffer (potentially inefficient)
            accumulated_audio = np.concatenate(audio_buffer)

            # 4. Feed ENTIRE accumulated buffer to VAD model
            # NOTE: This is INEFFICIENT as it re-processes audio.
            # NOTE: The behavior of the streaming VAD cache and endpoint detection
            #       when receiving the full buffer like this is UNCERTAIN.
            try:
                vad_res = vad_model.generate(
                    input=accumulated_audio,
                    cache=cache_vad, # Passing persistent cache might be wrong here
                                     # Alternatively, pass cache={} to reset state each time?
                    is_final=False,  # Still technically not the final end of the whole stream
                    chunk_size=VAD_CHUNK_SIZE_MS # Meaning of this param is unclear now
                )

                # 5. Check VAD result for endpoint signal (UNCERTAIN LOGIC)
                # This logic assumes the VAD somehow signals an endpoint for the
                # *entire buffer* via res[0]['value'] when silence occurs at the end.
                # This needs validation and might require more complex checking
                # (e.g., comparing end timestamp in 'value' to buffer length).
                if "value" in vad_res[0] and vad_res[0]['value']: # vad_res and isinstance(vad_res, list) and len(vad_res) > 0 and
                    print(f"VAD value list: {vad_res[0]['value']}")
                    beg, end = vad_res[0]["value"][0]
                    if not vad_startpoint_detected and beg >= 0:
                        vad_startpoint_detected = True
                    if vad_startpoint_detected and not vad_endpoint_detected and end >= 0:
                        vad_endpoint_detected = True

            except Exception as e_vad:
                print(f"\nError during VAD processing on accumulated buffer: {e_vad}")
                # Decide how to handle VAD errors - maybe clear buffer and restart?
                # For now, let's assume it means an endpoint to avoid infinite loops
                vad_startpoint_detected = False
                vad_endpoint_detected = False

            # 6. Process with ASR if endpoint detected
            if vad_startpoint_detected and vad_endpoint_detected and audio_buffer: # Ensure buffer isn't empty
                # print(f"\n[VAD Endpoint Detected. Processing {len(accumulated_audio)/TARGET_SAMPLE_RATE:.2f}s segment...]")
                segment_audio_to_process = accumulated_audio # Process the whole buffer
                audio_buffer = [] # Clear buffer for next segment
                print(f"Processing {len(segment_audio_to_process)/TARGET_SAMPLE_RATE:.2f}s segment...")
                # 7. Process the complete segment with the ASR model
                try:
                    asr_res = asr_model.generate(
                        input=segment_audio_to_process, # Pass the numpy array directly
                        cache=cache_asr, # ASR model likely doesn't use cache this way
                        language=ASR_LANGUAGE,
                        use_itn=ASR_USE_ITN,
                        batch_size_s=ASR_BATCH_SIZE_S,
                        merge_vad=ASR_MERGE_VAD,
                        merge_length_s=ASR_MERGE_LENGTH_S,
                    )

                    processed_text = rich_transcription_postprocess(asr_res[0]["text"])
                    print(f"ASR Result: {processed_text}")
                    
                    # Flag that we have processed the segment
                    vad_startpoint_detected = False
                    vad_endpoint_detected = False
                    
                    # Clear cache
                    cache_vad={}
                    cache_asr={}


                except Exception as e_asr:
                    print(f"\nError during ASR processing: {e_asr}")
                    traceback.print_exc()

            # Optional small sleep
            # time.sleep(0.01)


# --- Handle Termination ---
except KeyboardInterrupt:
    print("\n--- Stopping VAD + ASR Pipeline ---")
    # Process any remaining audio in the buffer
    if audio_buffer:
        print("[Processing remaining audio in buffer...]")
        accumulated_audio = np.concatenate(audio_buffer)
        audio_buffer = [] # Clear buffer

        try:
            asr_res = asr_model.generate(
                input=accumulated_audio, # Process final accumulated audio
                cache={},
                language=ASR_LANGUAGE,
                use_itn=ASR_USE_ITN,
                batch_size_s=ASR_BATCH_SIZE_S,
                merge_vad=ASR_MERGE_VAD,
                merge_length_s=ASR_MERGE_LENGTH_S,
            )
            if asr_res and isinstance(asr_res, list) and len(asr_res) > 0 and "text" in asr_res[0]:
                 processed_text = rich_transcription_postprocess(asr_res[0]["text"])
                 print(f"Final ASR: {processed_text}")
            else:
                 print("ASR: (No final text detected)")

        except Exception as e_asr:
            print(f"\nError during final ASR processing: {e_asr}")
            traceback.print_exc()

    # No final VAD call needed as we weren't using its streaming state correctly

except Exception as e:
    print(f"\nAn unexpected error occurred: {e}")
    traceback.print_exc()

finally:
    print("Script finished.")
