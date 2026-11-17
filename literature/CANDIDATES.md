# Candidate human benchmarks (literature sweep, 2026-11-17)

Inclusion criteria (same as the existing `literature/*.json`): image-only stimuli (natural objects/scenes),
single or near-single exposure, study → test structure matching our task code, accuracy + list length reported.

Legend: **[V]** = number read verbatim from the paper text. **[F]** = figure-only / not yet extracted.

---

## Existing entries — verification

| Entry | Status |
|---|---|
| `color_memory.json` Brady2013Color | **Verified.** Values are the mean of Exp 1a (3 s/item) and 1b (1 s/item): perception 6.7/4.7° → 5.7°, WM(3) 19.0/17.8° → 18.4°, LTM 20.3/19.3° → 19.8°; guess .0/.006, .09/.08, .58/.63. Note: "232" is the number of study *presentations* (180 tested items + 26 back-to-back repeats); the paper calls it the "180 items" condition. |
| `serial_order.json` Dubrow2014 | **Verified** against DuBrow & Davachi 2014 J Neurosci Exp 1 (N=24): overall .767, neighbors (lag 1) .657, lag-3 no-switch .785, long (7–9 intervening) .845. Caveat: 0.79 is the *within-context* lag-3 value; across-boundary lag-3 = .731. Lists are faces + objects with a category/task switch every 2 or 7 items. |
| `recognition.json` (benchmarks copy) | **Bug in `../vision-memory-tasks/benchmarks/recognition.json`**: `hit_rate_delays` is missing `256` (10 delays vs 11 hit rates). The `literature/` copy is correct. |

---

## 1. Recognition

### Konkle, Brady, Alvarez & Oliva 2010 (JEP:General 139:558) — objects  ★ strong fit
- 2,800 object images, 3 s each, N=18, single exposure; 2AFC test after study. Same Brady object stimulus family as our `Brady2008` dataset.
- 2AFC [V]: novel foil **92.6%** (SEM 1.3), one-exemplar foil **88.6%** (SEM 2.0), 16 studied exemplars → **82%**; slope −2.0%/doubling of exemplars (SEM 0.4). Values at 2/4/8 exemplars [F].
- Continuous repeat detection during study [V]: hits **95 / 94 / 87 / 79 / 63%** at lags **1 / 15 / 63 / 255 / 1023**, FA **3.9%** (SEM 0.5).
- A second independent continuous-recognition lag curve for objects, out to lag 1023.

### Konkle, Brady, Alvarez & Oliva 2010 (Psych Science 21:1551) — scenes  ★ strong fit (scene stimuli)
- 2,912 scenes (128 categories; 1/4/16/64 exemplars per category), 3 s each, N=24.
- 2AFC [V]: novel **96%** (SEM 1); exemplar foils **84%** (4 exemplars) → **76%** (64 exemplars); 16 [F]; slope −1.8%/doubling.
- Repeat detection [V]: hits **96 / 91 / 80 / 74 / 56%** at lags **1 / 15 / 63 / 255 / 1023**, FA **3%**.
- Useful if a scene dataset is ever added; otherwise a cross-check on the object curve.

### Kramer, Hebart, Baker & Bainbridge 2023 (Sci Adv) — THINGS memorability  ◐ promising, numbers incomplete
- Continuous recognition on **THINGS images** (our second dataset): 187 images/session, 500 ms each, ≥60 s lag between repeats, N=13,946.
- [V] mean corrected recognition (HR−FA) across THINGS categories = **0.793**. Overall HR / FA and lag distribution are in supplementary Table S2 / Figs S1–S2 [F]; raw data on OSF (osf.io/5a7z6), so per-lag curves could be computed directly.

### Standing 1973 (QJEP 25:207) — set-size curve  ◐ needs PDF
- The classic accuracy-vs-list-length dataset (pictures up to 10,000; 2AFC). Only secondary figures found (≈83% at 10,000). All copies I could reach were blocked (403). If you can drop the PDF into this folder I'll extract the table.

### Endress & Potter 2014 (JEP:General 143:548) — short lists  ○ partial fit
- Brady 2008 objects, RSVP at 250 ms/item, single-probe yes/no; list lengths 5–21 and 25/50/100.
- [V] capacity K = N·(H−FA): 3.2 (5 items) → 9.1 (21 items); up to 30.0 at 100 items. Accuracy values only in figures [F]. RSVP timing and the old/new format differ from our 2AFC, so this is a weaker match.

---

## 2. Serial order

### DuBrow & Davachi 2013 (JEP:General 142:1277)  ◐ good fit, smaller numbers
- 25-image lists (faces + objects), 2 s each, single study; 2AFC recency ("which came more recently").
- [V] Exp 1 (N=22): overall **.76** (SE .02). Exp 2 (N=22, distractor between items): **.70**. Exp 3 (N=21): **.70**. Lag breakdown only in figures.
- Exp 1 also has **serial recall**: 63% of items recalled, 39% in the correct order. That is not our position-report task, but it is related.

### Sherman & Yousif 2025 (Psych Science, doi 10.1177/09567976251330290)
- Appears to be the source of the existing `Sherman2025` entries ("temporal repetition effect"; 250-image stream). I didn't re-extract numbers. Worth checking that the 2AFC 0.78 is from *non-repeated* items, since the paper's manipulation is repetition frequency.

**Gap:** no image-based study with free *absolute position* report at multiple list lengths turned up. Short-list visual reconstruction work (Smyth et al. 2005; Ward, Avons & Melling 2005) uses unfamiliar faces at about 7 items, which is a weak match.

---

## 3. Paired associates (image–image, cue + 2AFC with a familiar foil)

### Schlichting, Zeithamova & Preston 2014 (Hippocampus 24:1248)  ★ strong fit
- Grayscale everyday objects; 72 AB + 72 BC + 72 XY pairs; **each pair seen once** [V], 3 s/pair; test 2AFC with a familiar foil from another triad.
- [V] AB **81.6%** (SEM 2.5), BC **76.8%** (SEM 2.7); N=21. XY accuracy not reported.
- AB doubles as a PAM benchmark (about 72 pairs per run; test follows each study scan).

### Carpenter & Schacter 2018 (JEP:General 147:988), Exp 1  ○ usable with caveats
- 80 AB, then 80 BC, person–object colour photos, 5 s/pair, single exposure; 2AFC + "neither". **48-h delay** and a reward manipulation.
- [V] direct (AB/BC) **0.73** (SE .02); N=25.

---

## 4. Associative inference

### Schlichting, Zeithamova & Preston 2014  ★ strongest single-exposure match
```json
"Schlichting2014": {
    "n_subjects": 21,
    "stimuli": "grayscale everyday objects",
    "n_triads": 72,
    "sequence_length": 144,
    "n_repetitions_per_pair": 1,
    "AB_accuracy": 0.816,
    "BC_accuracy": 0.768,
    "AC_accuracy": 0.652,
    "note": "Blocked AB then BC within study scans (12 pairs per block); 2AFC with familiar foil from another triad; test after each study scan."
}
```

### Carpenter & Schacter 2018 Exp 1  ○ caveats (48 h delay, people as A/C)
```json
"CarpenterSchacter2018": {
    "n_subjects": 25,
    "sequence_length": 160,
    "n_repetitions_per_pair": 1,
    "direct_accuracy": 0.73,
    "AC_accuracy": 0.63,
    "note": "80 AB then 80 BC (person-object photos); 48-h delay; 2AFC + 'neither' option (neither: 2% direct, 7% AC)."
}
```

---

## 5. Color memory — Brady et al. 2013 has a set-size curve you aren't using yet  ★

All from Brady, Konkle, Gill, Oliva & Alvarez 2013 (Psych Sci) incl. supplement [V]; same stimuli and continuous-report method as our task:

| condition | items | precision (SD°) | guess | notes |
|---|---|---|---|---|
| WM, Exp 2 | 1 | 14.5 | (not in text) | 3 s encode, 1 s delay, N=6 |
| WM, Exp 1a/1b | 3 | 19.0 / 17.8 | .09 / .08 | existing entry |
| WM, Control 1 | 5 | 19.5 | .28 | |
| LTM, Control 3 | 20 | 18.9 | .29 | 5 s/item, change-detection interference before test |
| LTM, Exp 1a/1b | 180 | 20.3 / 19.3 | .58 / .63 | existing entry |
| LTM, Exp 2 | 180 | 20.5 | .67 | |
| LTM, Control 2 | 360 | 19.6 | .73 | |

So the guess rate rises steadily with list length (.08 → .29 → .6 → .73) while precision stays flat at about 19–20°. That curve maps directly onto our `n_images` sweep.

Also relevant for the `named` mode: Brady, Konkle, Alvarez & Oliva 2013 (JEP:General), Exp 1 long-delay. 100 tested objects among 340 shown, 1 s each; 2AFC colour vs a 180°-rotated foil gives **67.8%** [V]. Same task with 3 objects in WM: **78.8%**.

---

## 6. MST
Vanderlip & Stark 2024 already pools about 410 subjects; I found nothing that improves on it. Stark et al. 2013 (Neuropsychologia) is the canonical normative source if you want an independent young-adult point. Its numbers weren't reachable.

---

## Still to chase (blocked fetches; a PDF in this folder would unblock them)
- Standing 1973 table (list length 20 → 10,000).
- Isola et al. 2014 PAMI memorability: mean HR/FA at lag about 100 (scenes).
- Kramer et al. 2023 supplement Table S2 (THINGS HR/FA), or compute from OSF data.
- Konkle 2010 figure values (2/4/8 exemplars; 16 for scenes).
