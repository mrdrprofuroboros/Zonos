import time
import pyaudio
import numpy as np

import torch
import torchaudio

from zonos.conditioning import make_cond_dict
from zonos.model import Zonos

import threading
import queue

def stream_audio_nonblocking(stream_generator, model=None, buffer_size=1024):
    """
    Stream audio chunks from a generator in real-time using PyAudio with non-blocking playback.
    
    Args:
        stream_generator: Generator that yields audio chunks as torch tensors
        model: The TTS model (needed to get sampling rate)
        buffer_size: Size of audio buffer (default: 1024)
    """
    # Get the sampling rate from the model
    if model and hasattr(model, 'autoencoder') and hasattr(model.autoencoder, 'sampling_rate'):
        sample_rate = model.autoencoder.sampling_rate
    else:
        # Default to common sample rate if model info not available
        sample_rate = 44100
        print("Warning: Using default sample rate of 44100 Hz")
    
    # Create a queue for audio chunks
    audio_queue = queue.Queue()
    # Flag to signal the playback thread to terminate
    stop_event = threading.Event()
    # List to collect all chunks for potential later use
    all_chunks = []
    
    p = pyaudio.PyAudio()
    
    # Open an audio stream
    stream = p.open(
        format=pyaudio.paFloat32,
        channels=1,  # Assuming mono audio
        rate=sample_rate,
        output=True,
        frames_per_buffer=buffer_size
    )
    
    # Define playback thread function
    def playback_worker():
        try:
            while not stop_event.is_set() or not audio_queue.empty():
                try:
                    # Get audio chunk from queue with timeout to check stop_event periodically
                    chunk = audio_queue.get(timeout=0.05)
                    # Convert PyTorch tensor to numpy array for streaming
                    numpy_data = chunk.squeeze(0).numpy().astype(np.float32)
                    # Stream the audio chunk
                    stream.write(numpy_data.tobytes())
                    # Mark task as done
                    audio_queue.task_done()
                except queue.Empty:
                    continue
        finally:
            # Clean up resources
            stream.stop_stream()
            stream.close()
            p.terminate()
    
    # Start the playback thread
    playback_thread = threading.Thread(target=playback_worker)
    playback_thread.daemon = True
    playback_thread.start()
    
    # Process audio chunks from the generator
    t0 = time.time()
    try:
        for i, audio_chunk in enumerate(stream_generator):
            elapsed = int((time.time() - t0) * 1000)
            print(
                f"Received chunk {i + 1}: time {elapsed}ms | generated up to {audio_chunk.shape[1] / (sample_rate/1000) + elapsed:.0f}ms"
            )
            
            # Store audio chunks for potential later use
            cpu_chunk = audio_chunk.cpu()
            all_chunks.append(cpu_chunk)
            
            # Add to queue for playback
            audio_queue.put(cpu_chunk)
    
    except KeyboardInterrupt:
        print("Generation interrupted by user")
    except Exception as e:
        print(f"Error during generation: {e}")
    finally:
        # Signal playback thread to stop after queue is empty
        stop_event.set()
        # Wait for queue to be processed
        audio_queue.join()
        # Wait for playback thread to finish
        playback_thread.join(timeout=1.0)
    
    if len(all_chunks) == 0:
        print("No audio chunks were generated.")
        return None
    
    # Return the full audio in case it's needed for other purposes
    return torch.cat(all_chunks, dim=-1)


def set_prefix_audio(model, prefix_path: str = "./assets/silence_100ms.wav"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    prefix_audio = prefix_path
    wav_prefix, sr_prefix = torchaudio.load(prefix_audio)
    if prefix_path != "./assets/silence_100ms.wav":
        wav_sil, sr_sil = torchaudio.load("./assets/silence_100ms.wav")
        wav_prefix = torch.cat((wav_prefix, wav_sil), dim=1)
    wav_prefix = wav_prefix.mean(0, keepdim=True)
    wav_prefix = model.autoencoder.preprocess(wav_prefix, sr_prefix)
    wav_prefix = wav_prefix.to(device, dtype=torch.float32)
    audio_prefix_codes = model.autoencoder.encode(wav_prefix.unsqueeze(0))
    return audio_prefix_codes


def main():
    # Use CUDA if available.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model (here we use the transformer variant).
    print("Loading model...")
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-hybrid", device=device)
    model.requires_grad_(False).eval()

    ## alt loading pattern, if you prefer to save and maintain files yourself
    #model_type = "hybrid"
    #
    #if model_type == "hybrid":
    #    model_path = "/path-to-local/Zonos-v0.1-hybrid/model.safetensors"
    #    config_path = "/path-to-local/Zonos-v0.1-hybrid/config.json"
    #else:
    #    model_path = "/path-to-local/Zonos-v0.1-transformer/model.safetensors"
    #    config_path = "/path-to-local/Zonos-v0.1-transformer/config.json"
    #model = Zonos.from_local(config_path=config_path, model_path=model_path, device=device)
    #model.requires_grad_(False).eval()
    #torch.cuda.empty_cache()

    # Load a reference speaker audio to generate a speaker embedding.
    print("Loading reference audio...")
    wav, sr = torchaudio.load("assets/exampleaudio.mp3")
    speaker = model.make_speaker_embedding(wav, sr)

    # Set a random seed for reproducibility.
    torch.manual_seed(421)

    # Define the text prompt.
    text = "Hello!"

    # Create the conditioning dictionary (using text, speaker embedding, language, etc.).
    cond_dict = make_cond_dict(text=text, speaker=speaker, language="en-us")
    conditioning = model.prepare_conditioning(cond_dict)

    print(f"CUDA Graphs?: {model.can_use_cudagraphs()}")

    # --- STREAMING GENERATION --- this first run is just to run through one as a warmup
    print("Starting streaming generation...")

    # Define chunk schedule: start with small chunks for faster initial output,
    # then gradually increase to larger chunks for fewer cuts
    stream_generator = model.stream(
        prefix_conditioning=conditioning,
        audio_prefix_codes=None,  # no audio prefix in this test
        chunk_schedule=[17, *range(9, 100)],  # optimal schedule for RTX4090
        chunk_overlap=1,  # tokens to overlap between chunks (affects crossfade)
    )

    # Accumulate audio chunks as they are generated.
    audio_chunks = []
    t0 = time.time()

    for i, audio_chunk in enumerate(stream_generator):
        elapsed = int((time.time() - t0) * 1000)
        print(
            f"Received chunk {i + 1}: time {elapsed}ms | generated up to {audio_chunk.shape[1] / 44.1 + elapsed:.0f}ms"
        )
        audio_chunks.append(audio_chunk.cpu())

    if len(audio_chunks) == 0:
        print("No audio chunks were generated.")
        return

    #now we do it all over again in a loop, but streaming

    prefix_audio_codes = set_prefix_audio(model)

    while True:
        text = input("Text to speak: ")
        if text.lower() in ['q','quit','exit']:
            break

        # Create the conditioning dictionary (using text, speaker embedding, language, etc.).
        cond_dict = make_cond_dict(text=text, speaker=speaker, language="en-us")
        conditioning = model.prepare_conditioning(cond_dict)

        stream_generator = model.stream(
            prefix_conditioning=conditioning,
            audio_prefix_codes=prefix_audio_codes,
            chunk_schedule=[17, *range(10, 100, 3)],  # optimal schedule for ..RTX3090
            chunk_overlap=1,  # tokens to overlap between chunks (affects crossfade)
        )

        # Stream the audio in real-time
        full_audio = stream_audio_nonblocking(stream_generator, model)

    # If you still want to save the file after streaming
    if full_audio is not None:
        out_sr = model.autoencoder.sampling_rate
        torchaudio.save("stream_sample.wav", full_audio, out_sr)
    print(f"Saved streaming audio to 'stream_sample.wav' (sampling rate: {out_sr} Hz).")


if __name__ == "__main__":
    main()
