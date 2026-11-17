# Changelog: whisper-transcribe

## 2026-11-16: "Show en tell" (3,4 u, NL): Ollama ARG_MAX fix, rerun without glossary, cleanup scripts

**Goal.** Transcribe `/storage/fileshare/Show en tell.m4a` (VRO rocketry club meeting, Dutch, ~3,4 h) at best quality,
then rerun without the glossary and clean the result up. Output used for a VRO website article
(see `~/claudecode/projects/vro-website/docs/show-and-tell-2026-11-15.md`).

### 1. Fix: Ollama calls failed on long recordings ("Argument list too long")

The auto-prompt (around line 350) and summary (around line 670) stages passed the whole JSON request to curl as a
command-line argument (`-d "$(jq ...)"`). With a 3,4 h transcript that exceeds ARG_MAX; curl failed and the script
**still exited 0** with an empty auto-prompt or summary. Both calls now pipe the JSON into curl:

```bash
jq -nc ... --rawfile prompt "$FILE" '{...}' \
  | curl -s --max-time 600 http://localhost:11434/api/chat -H 'Content-Type: application/json' -d @- \
  | jq -r '.message.content // empty'
```

Timeout raised from 300 to 600 s for long inputs.

### 2. Runs

| Run | Settings | Result |
|-----|----------|--------|
| 1 | default glossary | Cabling-brand glossary polluted the Dutch text (e.g. "luik" → "Fluke"). Kept as `Show en tell (run1).txt` |
| 2/3 | empty glossary (`empty-glossary.txt`) | Clean ASR; output `/storage/fileshare/Show en tell.txt` |

Final cleaned transcript + hand-written summary: `/storage/fileshare/Show en tell.cleaned.txt`.

### 3. Post-processing (scratchpad scripts, not in the repo)

- `cleanup.py`: collapses 3+ identical consecutive sentences and single-word loops (6+ repeats → 2), drops known
  Whisper hallucination lines ("TV Gelderland 2021", "Ondertiteld door ...", Amara.org credits).
  Gotcha: an earlier phrase-loop regex over multi-word units doubled text; limited to single words.
- `fixterms.py`: inline DROP patterns (hallucinations in the middle of a line, leaked prompt text) plus ~85 domain
  rules (Euroc, CATS Vega, drogue, traagheidsmoment, Nike Hercules, Sidewinder, Sabca, DeepSeek, Claude Code ...).

### Notes / gotchas

- **VRAM:** Ollama kept ~17 GB loaded; without passwordless sudo `systemctl stop ollama` fails. Unload via the API:
  `curl -s localhost:11434/api/generate -d '{"model":"<m>","keep_alive":0}'`. A small guard loop that unloads models
  while a whisper container runs worked well.
- **Z.ai cloud correction** returned 429 (weekly/monthly limit exhausted, reset 2026-11-20 02:50). Not used this run.
- **Diarization** on a noisy table conversation merges several people into one `SPEAKER_xx`; never attribute quotes
  to a person from the label alone.
- Domain glossaries hurt unrelated recordings: use an empty glossary for anything outside the glossary's domain.
- `pkill -f <script>` from a shell whose command line contains the pattern kills that shell too; use
  `pgrep -af "name_[x]"`-style patterns.
