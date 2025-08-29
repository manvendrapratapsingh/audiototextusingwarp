AGENTS Guide — Audio→Text (Hindi Focus)

Purpose
- Preserve context of design decisions, experiments, and the practical knobs to control transcription quality (esp. Hindi).
- Make future iterations faster and safer by documenting what worked, what didn’t, and how to evaluate changes.

Current Architecture
- Backend: FastAPI in `backend/app.py`.
- Model: Whisper (default: `large-v3`), accelerated via `faster-whisper` when enabled.
- Audio I/O: FFmpeg for resampling to 16 kHz mono WAV.
- Optional: IndicNLP normalization for Hindi text; PyAnnote diarization.
- Frontend: Simple HTML/JS with WebSocket progress updates.

Frontend Options (New)
- Language selector (default Hindi `hi`).
- Accuracy mode toggle.
- Initial prompt textarea (Hindi recommended).
- Domain terms (comma-separated) to bias decoding.
These are posted to the backend and applied across all decoding paths.

Key Goals
- Plain and accurate Hindi transcript (no translation), robust for long files.
- Avoid missing speech segments while keeping hallucinations low.

Important Toggles (in code)
- Model: `WHISPER_MODEL_SIZE` (default `large-v3`).
- Engine: `USE_FASTER_WHISPER` (True → faster inference).
- Audio enhancement: `ENABLE_AUDIO_ENHANCEMENT` (noise reduction; can hurt faint speech).
- Diarization: `ENABLE_SPEAKER_DIARIZATION`.
- Chunking: `FORCE_CHUNKING` for very long audio.
- Accuracy bias (new): pass `accuracy_mode=true` in request to enable stronger anti‑hallucination settings.
- Initial prompt (new): pass `prompt` or `domain_terms` in request to bias vocabulary/domain.
- Hindi corrections (new): `ENABLE_HINDI_CORRECTIONS=True` applies a conservative post-correction map to fix frequent ASR misspellings (e.g., कांग्रेस/राजद/बीजेपी/जेडीयू, आक्षेप, बेरोजगारी, छूटता).

API — Context-Aware Transcription (New)
- Endpoint: `POST /transcribe/{file_id}`
- Optional JSON body:
  - `language`: ISO code (default: `hi`).
  - `prompt`: A seed prompt (prefer Hindi, include domain/context, names).
  - `domain_terms`: Array of words/phrases to bias.
  - `accuracy_mode`: Boolean. When true, enables stronger hallucination suppression.
  - Internally: `domain_terms` are appended to the prompt; if no prompt is given, a default Hindi prompt is used when `language='hi'`.

How initial_prompt helps (Hindi)
- Whisper supports an initial prompt for decoding. Provide a short Hindi instruction and domain terms to improve recognition of proper nouns, technical jargon, product names, etc.
- Example prompt: "कृपया शुद्ध हिंदी में स्पष्ट और सरल प्रतिलेखन दें। निम्न शब्दों पर ध्यान दें: …"

Accuracy vs Coverage Profiles
- Coverage-first (default path in code):
  - Minimal filtering (`vad_filter=False`) to avoid missing speech.
  - Good for completeness; can include some non‑speech or hallucinated fragments in noisy audio.
- Accuracy mode (opt-in):
  - Adds `compression_ratio_threshold=2.4` (standard Whisper heuristic) to suppress likely hallucinations.
  - Keeps `beam_size` high and `temperature=0.0` primary.
  - Recommended when audio quality is decent and correctness is preferred over marginal coverage.
  - Propagated to chunked and emergency paths as well.

What We Tried So Far (Summary)
- Disabled VAD in core path to prevent dropped segments; chunked long audio with overlaps.
- Multi‑temperature reprocessing fallback; gap recovery strategy when coverage < threshold.
- IndicNLP normalization for Hindi character consistency.
- Findings: Turning VAD off improved coverage but allowed some hallucinations. Large‑v3 is best for Hindi; enhancement sometimes removes faint speech. More leverage is needed: domain biasing via prompts and a tunable accuracy profile.
  Recent: Added prompt/domain bias, accuracy mode across all paths, and a Hindi correction pass; fixed a None crash in chunked mode by making Indic enhancement safe and guarding word_count.

Recommended Workflow
- Clean input if very noisy (external tool or controlled recording) rather than relying on aggressive noise reduction.
- Use `prompt` and `domain_terms` for names/jargon-heavy audio.
- Enable `accuracy_mode` for lectures, interviews with good mic.
- For code‑mixed Hinglish, still keep `language='hi'` unless clearly multi‑lingual; prompt can mention English terms are acceptable.
  Example prompt: “कृपया शुद्ध और सरल हिन्दी में प्रतिलेखन करें। राजनीतिक संदर्भ है। संक्षेप, पार्टी नाम और चुनाव सम्बन्धी शब्द सही लिखें।”
  Example domain terms: “कांग्रेस, राजद, बीजेपी, जेडीयू, वोट चोरी, आक्षेप, बेरोजगारी, लॉ एंड ऑर्डर, बीएलए, बूथ लेवल एजेंट, चुनाव आयोग”.

Quality Checks
- Measure: word error rate (WER) on a small curated Hindi set; track average and 90th percentile.
- Sanity: look for duplicated phrases, unrelated hallucinated content, or missing sections (coverage < 95%).
- Logging: check `total_duration_processed` vs original duration and any gap recovery triggers.

Next Ideas (Backlog)
- Add smart VAD‑only segmentation (Silero) to cut at silence boundaries while keeping all frames, then feed full segments (no drop) to Whisper.
- Optional punctuation restoration tuned for Hindi (external model), gated by a flag.
- Hotword biasing from a user‑provided glossary (expand `initial_prompt` automatically).
- Scoring candidates by average log‑prob to auto‑select best between decoding strategies.

Operational Notes
- GPU greatly improves speed; CPU works but is slower with `large-v3`.
- If RAM limited, use `medium` and enable `accuracy_mode=true` to keep quality reasonable.
- Avoid enabling heavy noise reduction by default; it may erase quiet speech.

File Map
- `backend/app.py`: Main service and transcription logic. Parameters for `language`, `prompt`, `domain_terms`, `accuracy_mode` are wired through all paths (main, reprocessing, gap recovery, chunked, emergency). Includes Indic normalization and optional Hindi corrections.
- `config.py`: Generic config (some duplication remains; refactor later if needed).
- `templates/index.html`: UI with options panel.
- `static/js/main.js`: Sends options to backend.
- `static/css/style.css`: Styling for options panel.

Changelog (Context)
- 2025‑08‑29:
  - Add `AGENTS.md`; add `initial_prompt`, `domain_terms`, and `accuracy_mode` support; document profiles and workflow.
  - Add UI options (language, accuracy, prompt, domain terms).
  - Add Hindi corrections pass; apply cleanup and corrections in all paths.
  - Fix chunked-mode None crash by ensuring Indic enhancement returns text and guarding word_count.
