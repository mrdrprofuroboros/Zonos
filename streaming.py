import time

import torch
import torchaudio

from zonos.model import Zonos

texts = [
    "The old clock tower hadn't chimed in living memory.",
    "Its stone face, weathered and stained, watched over the perpetually drowsy town.",
    "Elara, however, felt a strange pull towards it.",
    "She often sketched its silhouette in her worn notebook.",
    "One moonless night, a faint, melodic hum vibrated through the cobblestones beneath her feet.",
    "It seemed to emanate from the silent tower.",
    "Driven by a curiosity stronger than fear, she crept towards the heavy oak door.",
    "Surprisingly, it swung open at her touch, revealing a spiral staircase choked with dust.",
    "The air inside was thick with the scent of ozone and something ancient.",
    "She ascended, each step echoing in the profound stillness.",
    "Higher and higher she climbed, the humming growing louder, resonating within her chest.",
    "Finally, she reached the belfry.",
    "Instead of bells, intricate crystalline structures pulsed with soft, blue light.",
    "They hung suspended, rotating slowly, emitting the enchanting melody.",
    "In the center hovered a sphere of swirling energy.",
    "As Elara approached, the humming intensified, the light brightening.",
    "Tendrils of energy reached out from the sphere, brushing against her fingertips.",
    "A flood of images poured into her mind: star charts, forgotten equations, galaxies blooming and dying.",
    "She wasn't just in a clock tower; she was inside a celestial resonator.",
    "It was a device left by travelers from a distant star, waiting for someone attuned to its frequency.",
    "Elara realized the tower hadn't been silent, just waiting.",
    "She raised her hands, not in fear, but in acceptance.",
    "The energy flowed into her, cool and invigorating.",
    "Suddenly, with a resonant *gong*, the tower chimed, a sound unheard for centuries.",
    "Its song wasn't marking time, but awakening possibilities across the cosmos.",
]


def main():
    # Use CUDA if available.
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load the model (here we use the transformer variant).
    print("Loading model...")
    model = Zonos.from_pretrained("Zyphra/Zonos-v0.1-transformer", device=device)
    model.requires_grad_(False).eval()

    # Load a reference speaker audio to generate a speaker embedding.
    print("Loading reference audio...")
    wav, sr = torchaudio.load("assets/exampleaudio.mp3")
    speaker = model.make_speaker_embedding(wav, sr)

    # Set a random seed for reproducibility.
    torch.manual_seed(777)

    # Accumulate audio chunks as they are generated.
    audio_chunks = []
    t0 = time.time()
    generated = 0
    ttfb = 0 # Time to first byte

    def generator():
        # Can stream from your LLM or other source here, just partition the text into
        # sentences with nltk or rule based tokenizer. See example here:
        # https://stackoverflow.com/a/31505798
        for text in texts:
            elapsed = int((time.time() - t0) * 1000)
            print(f"Yielding sentence {elapsed}ms: {text}")
            yield {
                "text": text,
                "speaker": speaker,
                "language": "en-us",
            }

    # --- STREAMING GENERATION ---
    print("Starting streaming generation...")

    # Define chunk schedule: start with small chunks for faster initial output,
    # then gradually increase to larger chunks for fewer cuts
    stream_generator = model.stream(
        cond_dicts_generator=generator(),
        chunk_schedule=[15, 9, 9, 9, 9, 9, *range(9, 100)],  # optimal schedule for RTX3090 and this warmup
        chunk_overlap=2,  # tokens to overlap between chunks (affects crossfade)
        warmup_prefill="And I say OK.",
        mark_boundaries=True,
    )

    for i, audio_chunk in enumerate(stream_generator):
        if isinstance(audio_chunk, str):
            print(audio_chunk)
            continue

        audio_chunks.append(audio_chunk)
        elapsed = int((time.time() - t0) * 1000)
        if not i:
            ttfb = elapsed
        gap = "GAP" if ttfb + generated < elapsed else ""
        generated += int(audio_chunk.shape[1] / 44.1)
        print(f"Chunk {i + 1:>3}: elapsed {elapsed:>5}ms | generated up to {ttfb + generated:>5}ms {gap}")

    # Concatenate all audio chunks along the time axis.
    audio = torch.cat(audio_chunks, dim=-1).cpu()

    generation = round(time.time() - t0, 3)
    duration = round(audio.shape[1] / 44100, 3)

    print(f"TTFB: {ttfb}ms, generation: {generation}ms, duration: {duration}ms, RTX: {round(duration / generation, 2)}")

    # Save the full audio as a WAV file.
    out_sr = model.autoencoder.sampling_rate
    torchaudio.save("streaming.wav", audio, out_sr)
    print(f"Saved streaming audio to 'streaming.wav' (sampling rate: {out_sr} Hz).")

    # Or use the following to display the audio in the jupyter notebook:
    # from IPython.display import Audio
    # display(Audio(data=audio, rate=out_sr))


if __name__ == "__main__":
    main()
