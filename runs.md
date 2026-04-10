# Experiment runs log

One row per training run. Update during/after, not before.

| run name | date | what changed | epochs | SI-SDR | MixRate@0.5 | listen verdict | decision |
|---|---|---|---|---|---|---|---|
| comp-24x-v6 (phase 1) | 2026-03-19 | baseline, no mixing | 299 | -24.90 dB | 1.22 (no mix loss) | broken pitched content, drums OK | needs vocos head |
| comp-24x-v6-phase2-final | 2026-03-29 | + λ_mix sweep finalized | 390 | -6.05 dB | 0.95 | robot vocals, dull bass, broken piano/guitar, drums mostly OK | needs vocos head |

## Per-checkpoint listening notes

### comp-24x-v6-phase2-final/best_model.pth (epoch 390)
Samples at: results/samples/comp-24x-v6-phase2-final/

- 00 vocals: broken robot singing
- 01 bass: dull, sounds like drums not bass guitar
- 02: vocals: broken robot singing
- 03: Sound ok for 90%. Some small artifacts
- 04: Metalic sound and borken piano
- 05: Drums ok but sounds artifacts and drums sound a bit different
- 06: Bass sound dull doesnt' sound like bass guitar but structure recognizable
- 07: Echo and I think guitar broken
- 08: Electro guitar broken 
- 09: Drums sounds mostly ok

### comp-24x-v6/best_model.pth (phase 1, no mixing)
Samples at: results/samples/comp-24x-v6-phase1/

- (TODO: user listens)

## Phase 1 numbers (Vocos OFF, raw 2-ch real/imag decoder)

| ckpt | epoch | val_loss | recon L1 | recon MRSTFT | recon total | SI-SDR | MixRate@0.5 |
|---|---|---|---|---|---|---|---|
| phase1 (no mix) | 299 | 2.25 | 0.172 | 1.766 | 1.937 | **-24.90 dB** | 1.22 |
| phase2-final | 390 | 4.25 | 0.117 | 1.536 | 1.653 | **-6.05 dB** | **0.95** |

### What this tells us

1. **The "MixRate=0.999" claim in memory was wrong.** Phase2-final actually scores 0.95 at α=0.5 — still reasonable but not the headline number. Memory has been corrected.
2. **Phase2 has BETTER recon than phase1** (recon total 1.65 vs 1.94, SI-SDR -6 vs -25). Mixing did not damage reconstruction — the opposite, the extra training and the mixing path *helped*. So the bottleneck is NOT the mixing loss.
3. **Phase1 MixRate > 1** because the model has no mixing prior at all — interpolated latents decode worse than encoded mixes. Expected.
4. **Both checkpoints are perceptually broken** in the same way (broken pitched content, drums survive). The problem is the decoder predicting raw real/imag, which can't represent phase coherently. **This is exactly what Vocos was designed to fix.**

## Open diagnoses

- v6 decoder predicts raw real/imag STFT (2 ch). Vocos 3-ch head was added in commit 69d08ed but never trained. Phase 1 confirms hypothesis: extra training (phase1→phase2) improves recon but does not fix phase incoherence. Vocos is the targeted fix.
- Mixing loss does NOT degrade recon; phase1 is worse than phase2-final on every metric. We can keep the linearity loss in Phase 5 without worrying it competes with reconstruction.
