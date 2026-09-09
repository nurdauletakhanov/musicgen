# Held-out evaluation protocol (pre-registered)

Written and committed **before** any MoisesDB audio is scored. Nothing below is
revised after seeing results. If a step turns out to be impossible, the change
is appended to the "Deviations" section with its date and reason, and the
pre-change version stays in git history.

Motivation: the released checkpoints were selected on the split that was also
used for reporting (`data.val_split` defaulted to `test`). Selection cannot
have created the origin effect, which compares one checkpoint against itself
in two coordinate systems, but it can bias between-model contrasts. Only
`best.pth` survives for the 15.3x runs, so the final-step weights cannot be
recovered without retraining. This protocol therefore freezes the existing
weights and evaluates them on a corpus that took no part in training or in
selection.

## 1. Frozen artifacts

Weights are read-only. SHA-256 of every file to be evaluated:

| run | role | sha256 |
|---|---|---|
| v2.0-continued | control, 7.66x | `92718bfe15f325b417fda7fb719c6ab6d60c424f1230101380ac088899460bd4` |
| v2.1-decmix | treatment, L_dec, 7.66x | `f47c8dc58c6f44aeae4df8d9c693db13c65552d0c028815e38635be126b1d891` |
| v2.2-decmix-disc | treatment, L_dec+disc, 7.66x | `3770e23fdd8d598ec3a098347aa09bdba3c9cd31191a76c1b1b9c0f0126dbb54` |
| v3.0-baseline-d64 | control, 15.3x | `5668eca13abc0ee78d641892ecdecd06533c9dd745002271e354c101159dfbcc` |
| v3.1-decmix-disc-d64 | treatment, L_dec+disc, 15.3x | `70df0436b57a2914ae6e0f3b3450d8a53a93aff39eba96d11f6b3624b3e1cb36` |
| v1.1 | warm-start ancestor of every v2.x run | `d2c41b9f424a3b2ddd6a2af11550f604cf8de95c05d8f1c73fa8982039ed4b49` |

No checkpoint may be swapped, re-selected, or re-averaged after this document
is committed. Every model is evaluated on identical examples.

## 2. Corpus and provenance

**MoisesDB** (240 tracks, 44.1 kHz stereo stems, CC BY-NC-SA 4.0). It appears
nowhere in the training corpus of any run in the ancestry (v1.1 -> v2.x, and
the from-scratch v3.x), which is a ~950 h union of FMA-large, MAESTRO and
MUSDB18-HQ train.

Provenance notes that constrain what may be claimed:

- **FMA is exhausted and cannot supply a held-out set.** The chunk index
  contains 106,304 FMA recordings; 270 more are recorded in
  `fma_skipped.json`, all of them decode failures or clips with no valid
  audio. That accounts for all 106,574 tracks of FMA-large. There are no
  unused FMA recordings, so mixing metrics are computed on MoisesDB as well,
  not on FMA leftovers.
- **MoisesDB is not neutral ground for Lin-CAE.** Torres et al. train on
  MTG-Jamendo, MoisesDB and M4Singer, so MoisesDB is training data for their
  released model. This evaluation therefore covers the GAN autoencoders only.
  Any Music2Latent or Lin-CAE number on this corpus is out of scope here and
  would need its own overlap statement.
- **MedleyDB is not used.** MUSDB18 contains 46 MedleyDB recordings, so
  MedleyDB overlaps both our training data and the already-inspected MUSDB
  test set.
- **Slakh2100 is the fallback only**, and only in the `redux` release, which
  removes MIDI duplicated across splits. If used it is described as
  synthetic-domain generalization and is not presented as evidence about
  natural vocal removal.

## 3. Eligible recordings and exclusions

Decided now, applied mechanically, never revised on the basis of scores.

1. Every MoisesDB track is a candidate. Track identity is the unit of
   independence throughout.
2. Stems are summed to form the mixture. The mixture is **defined** as the sum
   of the stems being used, so the additivity identity holds by construction
   and no "stems do not sum to the mixture" exclusion is needed. Relative stem
   gains are preserved exactly: no per-stem normalization, no limiting.
3. Audio is converted to mono by channel averaging and resampled to 44.1 kHz
   only if the source differs, matching the training preprocessing.
4. A track is excluded only if it fails to decode, or if it lacks at least one
   stem in the target mapping of section 4.
5. Chunks are 1 second, drawn by the same seeded, sorted-track selection used
   for MUSDB so that all models see identical units. Number of chunks per
   track and the seed are fixed in the driver before the run.

## 4. Stem mapping and silent-target policy

MoisesDB has 11 top-level stems. We use the grouping shipped by the dataset
authors, `mix_4_stems` in `moisesdb/defaults.py`, which is exactly MUSDB18's
four categories:

| target | MoisesDB top-level stems |
|---|---|
| vocals | `vocals` |
| bass | `bass` |
| drums | `drums` |
| other | every remaining stem: `guitar`, `piano`, `other_keys`, `bowed_strings`, `wind`, `percussion`, `other_plucked`, `other` |

Using the authors' own mapping rather than one of our choosing removes a
judgment call from the pipeline. `scripts/prepare_moisesdb.py` applies it and
writes the corpus in MUSDB18-HQ layout, so the existing evaluator runs
unchanged.

Rules:

- The residual is the sum of every stem except the target, computed from the
  same stems that form the mixture, so mixture minus target equals residual
  exactly.
- **Silent-target policy.** A (track, target, chunk) unit is scored only if the
  target stem is active in that chunk: target RMS at least -50 dBFS **and**
  target-to-mixture energy ratio at least -30 dB. SI-SDR against a
  near-silent reference is unstable, and this is the mechanism behind the
  extreme per-unit scores seen in the MUSDB evaluation. Both thresholds are
  fixed here.
- Counts of units dropped by the activity rule are reported per target.
- MoisesDB activity and bleeding metadata are used only to report how many
  units each rule removed, never to select which tracks are scored.

## 5. Metrics

Unchanged from the paper, computed by the same code paths:

- Reconstruction SI-SDR.
- Convex mixing at alpha = 0.5: SI-SDR vs ground truth, MixRate, and encoder
  linearity. Decode-vs-decode SI-SDR is recorded but is not a metric of
  record.
- Stem subtraction, raw: `g(f(mix) - f(stem))`.
- Stem subtraction, recentered: `g(f(mix) - f(stem) + f(0))`.
- Each model's own encode-decode reconstruction of the residual, as the
  reference for the linearity tax.
- Comparators on identical units: unchanged mixture, autoencoded mixture,
  decode-then-subtract in the waveform domain.

Mixing pairs: chunks are paired by a fixed-point-free permutation under a
fixed seed, identical across models, with alpha = 0.5 for the headline number
and the existing sweep grid for the secondary curve.

FAD is secondary and runs only if time allows.

## 6. Primary comparisons

Prespecified, in this order. Everything else is secondary and labelled as such.

1. Convex mixing, ground-truth SI-SDR: v2.1 vs v2.0, and v3.1 vs v3.0.
2. Reconstruction SI-SDR for the same two pairs, to confirm the property is
   not bought with fidelity.
3. Stem subtraction, raw advantage: v2.1 vs v2.0, v2.2 vs v2.0, v3.1 vs v3.0.
4. The same three contrasts after recentering, and the paired change between
   raw and recentered advantage.

The result that would confirm the paper: (1) and (2) reproduce the direction
and rough magnitude of the MUSDB numbers, (3) is clearly positive, and (4)
collapses toward zero. Any other outcome is reported as it comes out,
including an outcome that contradicts the current submission.

## 7. Statistics

- **Subtraction contrasts.** Paired bootstrap resampling whole recordings,
  10^4 resamples, as in the paper.
- **Mixing contrasts.** Each mixing unit involves two recordings, so a
  one-way cluster bootstrap is not valid. We resample recordings with
  replacement and keep a pair only when both of its recordings are drawn, a
  dyadic cluster bootstrap. Unit-level intervals may also be reported, always
  labelled as a lower bound on uncertainty.
- Intervals cover evaluation variance over recordings from this corpus. They
  do not cover training-seed variance. Checkpoint selection is no longer in
  the inference path because the corpus took no part in it.

## 8. What this does not fix

The MUSDB numbers already in the paper keep their checkpoint-selection
disclosure. This evaluation does not launder them; it adds an independent
measurement on untouched audio. Retraining with a genuinely held-out
validation split remains open, and this protocol is not a substitute for it.

## Deviations

- **2026-09-09, during conversion, before any scoring.** Some MoisesDB tracks
  have no sources at all in one of the four groups (no bass, or nothing
  outside the basic stems). The library's `mix_stems` raises on that case
  rather than returning an empty group. The converter now performs the same
  trim-and-sum the library does, per the authors' `mix_4_stems` grouping, and
  writes silence for an absent group instead of dropping the track. The
  mixture remains the exact sum of the four written files, and the
  silent-target rule in section 4 removes those units at scoring time with
  the counts reported. No model had been run at the time of this change.

- **2026-09-09, before any audio was downloaded.** The stem mapping in
  section 4 originally grouped `percussion` under drums. Replaced with the
  dataset authors' own `mix_4_stems` grouping, which places `percussion`
  under "other". Reason: it removes a judgment call of ours from the
  pipeline. No audio existed on disk at the time of the change.
