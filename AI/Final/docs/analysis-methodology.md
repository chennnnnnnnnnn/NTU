# PPG-DTW analysis methodology — dual manifest

Two complementary analyses for each (participant, stimulus, test_stage)
recording, produced by `scripts/analyze-participant.mjs`.

## Primary: `manifest_primary.json`  (vs native reference)

All 3 conditions compared to the **same C1 native reference**
(the Friends original recording for that stimulus).

- Δ = post_distance − pre_distance, where both distances are computed
  against the same native target
- Tests **paper H1** directly: which prompt condition (C1/C2/C3) leads
  to the largest gain in native-pronunciation similarity
- All conditions are on the same measurement scale → distances are
  directly comparable across conditions
- This is the analysis the paper Section 4 should report as the main
  finding; the registered ANOVA is `Δ ~ condition + (1|participant)`

## Supplementary: `manifest_supplementary.json`  (vs assigned model)

Each condition compared to the model voice the participant actually
shadowed for that stimulus:
- C1 → Friends native recording (same as primary)
- C2 → participant's voice rendered in native accent via ElevenLabs S2S
- C3 → participant's voice rendered in L1 (Mandarin) accent via Fish TTS

- Measures "voice imitation efficacy" (Yamanaka 2025 style)
- Each condition has its own scale, so cross-condition comparison of
  raw distances is not meaningful — but **Δ within condition** is
  still a valid measure of "did you get closer to the voice you heard"
- Belongs in the paper Appendix for cross-study comparability with
  Yamanaka 2025 and other shadowing-voice-perception work

## Why both?

Paper H1 is about "learning native pronunciation"; the prompt voice
is a *teaching scaffold*, not the target. The primary analysis is
therefore the right test of H1.

The supplementary analysis additionally answers the secondary question
"how closely did the participant manage to mimic the prompt voice?",
which is informative for understanding whether prompt-clone quality
(e.g. ElevenLabs S2S faithfulness) acts as a confound.

## Files produced per participant

```
analysis/<external_code>/
├── recordings/
│   ├── pre/   <stimulus_code>.webm                # cold-read baseline
│   ├── train/ <stimulus_code>.webm                # shadowing attempt
│   └── post/  <stimulus_code>.webm                # cold-read after training
├── models/         <stimulus_code>.mp3            # assigned-condition model
├── models_native/  <stimulus_code>.mp3            # C1 native reference (shared)
├── manifest_primary.json                          # all vs native
├── manifest_supplementary.json                    # each vs assigned
├── manifest.json                                  # alias of primary
├── summary.json
├── ppg_dtw_primary.csv                            # primary distances
└── ppg_dtw_supplementary.csv                      # supplementary distances
```
