# nemotron-asr

Evaluating **NVIDIA Nemotron 3.5 ASR** (`nvidia/nemotron-3.5-asr-streaming-0.6b`)
as a **CPU-only** speech-recognition front-end for this developer environment —
specifically for `TTUI` (Rust terminal-UI framework) and/or `Parallax` (the
platform binding the projects together). Unlike the other projects here, this
one is an *integration* study, not a training study: the research question is
whether an off-the-shelf 0.6B ASR checkpoint is fast and accurate enough on a
consumer CPU to sit inside an interactive terminal loop, not how to train one.

## The model

600M-parameter **cache-aware FastConformer-RNNT** with language-ID prompt
conditioning. One checkpoint covers 40 language-locales (~36 languages);
punctuation and capitalization are native to the output, so no downstream PnC
pass is needed. License **OpenMDW-1.1**, commercial use permitted.

The architecture matters for the use case here: *cache-aware* streaming means
the encoder keeps self-attention/convolution caches across chunks, so each
audio frame is processed exactly once. Traditional "buffered" streaming
re-processes an overlapping window every chunk. That is the difference between
an ASR that can idle inside a TUI event loop and one that cannot.

Latency is a runtime knob, not a retraining decision — `att_context_size`
selects an 80ms / 160ms / 320ms / 560ms / 1120ms chunk from the same weights.
Accuracy improves with larger chunks; the checkpoint is trained to be valid at
all five operating points.

## Runtime: the important fork

There are **two different C++/ggml runtimes**, and they use **incompatible
GGUF dialects**. This is the single most expensive thing to get wrong:

- **`mudler/parakeet.cpp`** (MIT, C++17/ggml) — has prebuilt **Windows x64
  CPU** binaries, a flat C-API shared library, and an OpenAI-compatible
  server. This is the one used here. It requires GGUFs converted by its own
  `scripts/convert_parakeet_to_gguf.py`, published at
  [`mudler/parakeet-cpp-gguf`](https://huggingface.co/mudler/parakeet-cpp-gguf).
- **NVIDIA's own NeMo-Speech.cpp** — the `.q8_0.gguf` committed to the NVIDIA
  model repo on 2026-08-05 is for *this* runtime.

**The GGUF in NVIDIA's own model repo does not load in parakeet.cpp.** Verified
directly (see experiment 001) — `parakeet-cli` reports only
`failed to load model`, with no hint that the dialect is the problem. Pull
nemotron GGUFs from `mudler/parakeet-cpp-gguf`, not from the NVIDIA repo.

The NeMo/Transformers paths on the model card are GPU-oriented (the card
states CPU inference is not explicitly supported). CPU-only viability comes
from the ggml runtimes, not from NVIDIA's Python stack.

## Verification approach

Real audio, known ground truth, measured on the target machine — not vendor
benchmark numbers. Test clips are generated locally with Windows SAPI
(`System.Speech`) at 16kHz mono, which makes the reference transcript exact by
construction and keeps the whole loop offline. WER is computed
case-insensitively with punctuation stripped (Levenshtein over word tokens).

Caveat this imposes, and it is a real one: **TTS audio is clean, single-
speaker, and free of the disfluencies, room noise, and mic artifacts of actual
dictation.** These numbers are a *ceiling*, not an estimate of live
performance. Any claim about real dictation accuracy needs real microphone
audio — see open leads.

Reference hardware (the "desktop" machine): Intel i7-9700K, 8 cores /
8 threads, no AVX-512, 32GB RAM. Deliberately CPU-only; the GTX 1660 Ti in the
box is ignored.

## Experiment index

### Standalone

| # | Question | Result |
|---|---|---|
| [001](experiments/001-cpu-feasibility.md) | Is Nemotron 3.5 ASR fast and accurate enough on a consumer CPU to embed in an interactive terminal loop? | **Yes, with headroom.** 7.6x realtime on 8 threads, still 2.4x realtime on **one** thread; 0% WER on general English. Failures are domain vocabulary, not acoustics. |

## Open leads

1. **Real microphone audio.** Every number in 001 is from clean TTS. Capture
   actual dictation (background noise, disfluencies, a real mic) before
   trusting any accuracy figure for live use.
2. **Domain vocabulary is the actual failure mode.** `ttui` transcribes as
   "Tui"; project jargon (`panopticon`, `plumb`, `baseline`, crate names, git
   subcommands) is the vocabulary that matters and the vocabulary the model
   has never seen. Two levers: NVIDIA documents fine-tuning for exactly this,
   or a cheaper post-correction map over a fixed jargon list. Measure how far
   the cheap fix gets before paying for the expensive one.
3. **Streaming latency is unmeasured.** 001 measures batch throughput on
   whole files. The number that decides whether this feels good in a TUI is
   time-to-first-token and time-to-final at the 80ms/160ms operating points,
   which requires driving the C-API `stream_feed` loop, not the CLI.
4. **Language tags leak into streaming output.** `--stream` emits `<en-US>`
   after each utterance even with an explicit `--lang`. NeMo has
   `strip_lang_tags`; the parakeet.cpp CLI does not appear to expose it, so an
   integration must strip them.
5. **Multilingual is unverified here.** Only English was tested — the SAPI
   voices installed on this machine are English-only. The 40-locale claim is
   NVIDIA's, untested locally.
6. **No Python package yet.** This project is currently docs plus a throwaway
   bench script; it is deliberately *not* a `uv` workspace member. Give it a
   `pyproject.toml` only once repeatable benchmarking earns it.
