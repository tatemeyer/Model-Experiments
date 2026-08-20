# Is Nemotron 3.5 ASR usable CPU-only inside an interactive terminal loop? (no issue — exploratory spike)

The prompt was open-ended ("look into Nemotron 3.5 ASR 0.6B CPU-only, eventually
for TTUI or Parallax"), so this is a feasibility spike rather than an
issue-driven experiment. Per this repo's loop an Issue should be filed before
any integration work follows; nothing here is merged behind a success criterion
yet.

NVIDIA's own model card is the reason the question is live at all: it documents
NeMo and Transformers inference paths, lists only NVIDIA GPU architectures under
supported hardware, and states outright that CPU inference is not explicitly
supported. Vendor throughput figures are all H100 concurrency counts
(~2,400 concurrent streams at the 1.12s chunk). None of that answers "can one
stream run acceptably on a 2018 desktop CPU," which is the only question that
matters for embedding it in a terminal UI.

Motivating architecture papers: cache-aware streaming
([arXiv:2312.17279](https://arxiv.org/abs/2312.17279)) and FastConformer
([arXiv:2305.05084](https://arxiv.org/abs/2305.05084)) — see `../LITERATURE.md`.

## Implementation

No model code was written. The measurement harness is three throwaway scripts
(`bench.py`, `threads.py`, `load.py`, kept out of the repo — this spike has not
earned a package):

- **Audio**: Windows SAPI `System.Speech.Synthesis` writing 16kHz mono WAV, so
  the reference transcript is exact by construction and nothing leaves the
  machine. Three clips: `short` (6.16s, a Parallax-flavored sentence), `code`
  (6.93s, deliberately loaded with dev jargon — `cargo`, `ttui`, branch names),
  `long` (17.43s, three sentences of general prose).
- **WER**: case-insensitive, punctuation stripped, Levenshtein over word tokens.
- **Runtime**: `parakeet-cli.exe` v0.5.0, prebuilt Windows x64 **CPU** bundle
  (1.4MB). Controlled variables are quantization (`q8_0` vs `q4_k`) and thread
  count; the model, audio, and decode settings are otherwise fixed.
- **Hardware**: i7-9700K, 8c/8t, no AVX-512, 32GB. GPU unused.

One non-obvious detail cost the most time and is worth stating plainly. The
`.q8_0.gguf` published in NVIDIA's own model repo **fails to load in
parakeet.cpp**, with no diagnostic beyond `failed to load model`. The file is
not corrupt — it is a valid GGUF v3 of exactly the advertised size
(741,548,352 bytes), and its `general.architecture` is even the same `asr`
string parakeet.cpp uses. It was added to the repo on 2026-08-05 for
**NeMo-Speech.cpp**, NVIDIA's own C++ runtime. The two projects have diverged
into incompatible GGUF dialects. parakeet.cpp needs its own conversions from
[`mudler/parakeet-cpp-gguf`](https://huggingface.co/mudler/parakeet-cpp-gguf),
where the same quantization is a differently-sized 984MB file. Nothing in
either project's docs warns about this.

**Result: comfortably viable — 7.6x realtime on 8 threads, and still 2.4x
realtime on a single thread, at 0% WER on general English.**

## Throughput vs. threads (`long.wav`, 17.43s, best of 2)

| threads | q8_0 wall | q8_0 RTF | q8_0 speed | q4_k wall | q4_k RTF | q4_k speed |
|---|---|---|---|---|---|---|
| 1 | 7.13s | 0.409 | 2.4x | 7.95s | 0.456 | 2.2x |
| 2 | 4.00s | 0.230 | 4.4x | 4.36s | 0.250 | 4.0x |
| 4 | 2.49s | 0.143 | 7.0x | 2.60s | 0.149 | 6.7x |
| 8 | 2.29s | 0.131 | 7.6x | 1.99s | 0.114 | 8.7x |

The single-thread row is the important one, and it is the row a vendor
benchmark would never report. Transcription at 2.4x realtime on **one** core
means dictation can be pinned to one or two threads and still keep up with
speech in real time, leaving six or seven cores for `cargo build` and the TUI
itself. The feasibility question is not "is there enough CPU" but "how little
CPU can be spent" — and the answer is: one core.

Scaling is close to linear to 4 threads (2.4x → 7.0x) then flattens hard
(7.0x → 7.6x). Past 4 threads the return is negligible for q8_0, so 4 is the
sensible default on an 8-thread box, not 8.

Model load is **0.33s** (q8_0) / **0.24s** (q4_k) — mmap'd, essentially free
and near-independent of the ~1GB file size. That makes a persistent
load-once process the obvious design, and also means the wall times above are
almost entirely inference rather than startup.

## Accuracy (8 threads)

| clip | audio | q8_0 WER | q4_k WER |
|---|---|---|---|
| short | 6.16s | **0.0%** | **0.0%** |
| code | 6.93s | 5.9% | 11.8% |
| long | 17.43s | **0.0%** | **0.0%** |

Two of three clips are transcribed perfectly at both quantizations —
punctuation, capitalization, sentence boundaries and all. The model even
capitalized "Panopticon" correctly as a proper noun, which the reference
transcript did not.

### Diagnosis: the errors are lexical, not acoustic

The aggregate WER hides what is actually happening. Every single error, at both
quantizations, is on the `code` clip, and inspecting the hypotheses shows they
are not acoustic failures:

| model | hypothesis (errors bolded) |
|---|---|
| reference | `Run cargo test in the ttui workspace, then commit the changes on branch feature slash streaming input.` |
| q8_0 | `Run cargo test in the **Tui** workspace, then commit the changes on branch feature slash streaming input.` |
| q4_k | `Run cargo test in the **Tui** workspace, then commit the changes on branch feature streaming input.` |

q8_0's *entire* error budget is one word: `ttui` → `Tui`. It got `cargo test`,
`commit`, `branch` and `feature` right — general developer vocabulary is
in-distribution. What it cannot produce is a coined project name it has never
seen, which is unsurprising for a model with a fixed 13,087-token vocabulary.

This reframes the integration risk. The bottleneck for dictating into a dev
environment is **not** recognition quality; it is a small, closed, *known* set
of proper nouns — `ttui`, `panopticon`, `plumb`, `parallax`, crate names. That
set is enumerable, which makes a post-correction map a plausible cheap fix and
makes fine-tuning (which NVIDIA documents for exactly this case) the fallback
rather than the starting point.

q4_k additionally drops `slash` — a real regression, doubling WER on the only
clip that has errors, for a 13% throughput gain at 4 threads. **q8_0 is the
right default**; q4_k's win only appears at 8 threads, which is the thread
count already ruled out as wasteful.

## Streaming

`--stream` works on CPU and segments output at utterance boundaries, but emits
a `<en-US>` language tag after each segment even when `--lang en-US` is passed
explicitly:

```
[stream] The quick brown fox jumps over the lazy dog. <en-US> Terminal user interfaces are ... <en-US>
[stream:final] ...
```

NeMo exposes `strip_lang_tags` to suppress these; the parakeet.cpp CLI does not
appear to. Any integration must strip them. Note this measures streaming
*correctness*, not streaming *latency* — see leads.

## Integration surface

parakeet.cpp ships a flat C-API (`parakeet.dll` + `parakeet_capi.h`, 0.7MB
Windows CPU bundle) whose streaming half is shaped almost exactly like a TUI
event loop:

```c
parakeet_stream* parakeet_capi_stream_begin_lang(parakeet_ctx*, const char* lang);
char* parakeet_capi_stream_feed(parakeet_stream*, const float* pcm, ...);
int   parakeet_capi_stream_drain_events(parakeet_stream*, ...);
char* parakeet_capi_stream_finalize(parakeet_stream*);
```

Feed PCM, drain events, render — no Python at runtime, no async framework, and
`bindgen` handles a header this flat without hand-written FFI. Three viable
integration shapes, cheapest first: subprocess the CLI (`--input -` accepts
stdin PCM), run `parakeet-server` and talk to its OpenAI-compatible HTTP
endpoint, or link the DLL directly.

The last is the one that fits TTUI's existing design. TTUI's `App::update`
takes a `crossterm::Event` and the crate has exactly one dependency
(`crossterm`) — pulling an ASR runtime into the framework itself would break
that. But TTUI already has the pattern for this: `src/audio.rs` defines an
`AudioSink` *trait* with a `NullAudioSink` default, letting apps supply a
backend while the framework stays backend-free. A `TranscriptSource` trait
mirroring `AudioSink` is the idiomatic move — the ASR implementation lives in a
separate crate (or in Parallax), and `ttui` itself gains no dependency.

**Leads for whoever picks this up next:**
1. Measure streaming *latency* — time-to-first-token and time-to-final at the
   80ms and 160ms operating points, via the C-API rather than the CLI. Batch
   RTF says nothing about how responsive it feels.
2. Re-run accuracy on real microphone audio. Every number here is clean TTS and
   is therefore a ceiling.
3. Build the jargon post-correction map and measure how much of lead 2's error
   it recovers before considering fine-tuning.
4. Decide the home: a `TranscriptSource` trait in `ttui` + backend crate, or a
   Parallax-side service. Related: Parallax already needs a remote transport for
   cross-device state, and an always-on ASR service has the same shape.
