# GenVerOpt — Toy Reproduction MM-IQ Verifier Branch

A staged, notebook-based attempt to reproduce the abstract-reasoning (IQ puzzle) experiment from
**"Can machines reason non-verbally?"** (Saeed et al., arXiv:2610.13925v1, Oct 2026), which introduces
the **GenVerOpt** (Generation–Verification–Optimisation) framework.

> **Status: negative / inconclusive result.** A from-scratch pairwise verifier trained on the MM-IQ
> data available here performs at chance (≈25%) on a held-out split. This does **not** reproduce the
> paper's reported gain and should not be read as evidence for or against the paper's claims. See
> [Results](#results) and [Known issues and limitations](#known-issues-and-limitations).

This is an independent implementation for learning purposes. It is not affiliated with the paper's
authors and no official code was available at the time of writing.

---

## Table of contents

1. [Background](#background)
2. [Scope and deviations from the paper](#scope-and-deviations-from-the-paper)
3. [Repository layout](#repository-layout)
4. [Pipeline](#pipeline)
5. [Results](#results)
6. [Interpretation](#interpretation)
7. [Reproducing the experiments](#reproducing-the-experiments)
8. [Data](#data)
9. [Known issues and limitations](#known-issues-and-limitations)
10. [Notes on the paper](#notes-on-the-paper)
11. [References](#references)

---

## Background

GenVerOpt frames machine reasoning as three components:

| Component | Role |
|---|---|
| **Generate** | Produce candidate solutions for an input. |
| **Verify** | A learned function `v(x, y) ∈ [0, 1]` scoring how likely candidate `y` is correct for input `x`. |
| **Optimise** | Select or refine the solution per sample, guided by the verifier. |

For abstract IQ puzzles the paper constrains generation to the four given options, so the problem
reduces to training a **verifier** and taking the argmax over candidates. The paper reports that its
verifier, meta-trained across puzzle categories with Reptile and fine-tuned on 32 temporal-movement
puzzles, outperforms multimodal LLMs and humans on 379 held-out temporal-movement puzzles. The
GenVerOpt bar in the paper's Fig. 3 reads at roughly 0.68 (approximate, read from a chart; the paper
gives no table for it).

This repository implements the **verifier branch only**, using the data that could be obtained.

## Scope and deviations from the paper

The paper's protocol could not be followed exactly. Deviations are listed explicitly so results are
not misread.

| Aspect | Paper | This repository |
|---|---|---|
| Dataset | MM-IQ, split by reasoning paradigm (Logical 395, Spatial 102, Geometric 355 for training; Temporal Movement 32 fine-tune + 379 eval) | Third-party Hugging Face mirror of MM-IQ (2,710 rows) with **no paradigm labels**; random split of a filtered subset |
| Candidate input | Candidate inserted into the missing square, verifier scores the completed grid | **Pairwise**: `(question image with options blanked, option crop)`; locating the "?" cell was not reliable across layouts |
| Training | Reptile meta-learning across paradigms, then task-specific fine-tuning | Single-stage supervised training from scratch; no Reptile, no fine-tuning stage |
| Verifier architecture | Not specified | Small two-tower CNN (see [Pipeline](#pipeline)) |
| Loss | BCE as in Eq. 2 (printed form is malformed, see [Notes on the paper](#notes-on-the-paper)) | Standard per-candidate BCE-with-logits, positives weighted ×3 |
| Evaluation | 379 temporal-movement puzzles | ~213 randomly held-out items from all puzzle types that fit the "fill in the question mark" format |

Consequently, **accuracies here are not comparable to the paper's numbers.**

## Repository layout

```
.
├── README.md
├── genveropt_stage3a.ipynb   # Environment inspection, config, MM-IQ schema inspection
├── genveropt_stage3b.ipynb   # Layout probe: label-row detection statistics
├── genveropt_stage3c.ipynb   # Candidate extraction v1 (option crops from A–D label positions)
├── genveropt_stage3d.ipynb   # Diagnostics for question image and crop cleaning
├── genveropt_stage3e.ipynb   # Extraction v2, conservative crop cleaning, cached dataset + split
└── genveropt_stage4.ipynb    # Pairwise verifier baseline, controls, grid of runs
```

Artifacts written at runtime (under `/content` in Colab, not committed):

| File | Produced by | Contents |
|---|---|---|
| `genveropt_mmiq_usable.npz` (~22.7 MB) | Stage 3e | `Q`, `O_raw`, `O_clean`, `y`, `idx`, plus an initial random split |
| `stage4_results.json` | Stage 4 | Per-run validation/test accuracies |
| `probe_*.png`, `extract_*.png`, `diag_*.png`, `clean2_check.png` | Stages 3b–3e | Visual checks |

## Pipeline

```
MM-IQ image ──► normalise (composite RGBA on white, cap size)
            ──► find A–D label row ──► 4 option crops (from label x-positions)
            ──► blank option region in the image ──► question image
            ──► (optional) conservative crop cleaning
            ──► cache arrays: Q 96×288, options 4×96×96, grayscale uint8
            ──► group-aware 70/15/15 split
            ──► train verifier: score(question, option) → argmax over 4 options
```

**Stage 3a – inspection.** Reports runtime, installs `datasets`/`pillow`/`matplotlib`, downloads the
mirror, prints schema. Finding: one split, 2,710 rows, columns `image`, `problem`, `solution`; no
category field; each image contains the question grid and the four options together.

**Stage 3b – layout probe.** 65.7% of images have exactly four glyph clusters in the last row band.
Failures include frames touching the A–D labels, puzzles with no labels in the image (e.g. numeric
tables), and eight image modes (RGBA needs compositing on white).

**Stage 3c – extraction v1.** Strips thin full-width rules, detects the label row, validates geometry
(4 clusters, band height, cluster width, even spacing), crops options. Result: 2,199 of 2,710 items
parsed; 1,449 prompts mention a question mark; 1,415 satisfy both.

**Stage 3d – diagnostics.** Confirmed question images are non-empty (median ink 6.6%; 17 of 2,199
under 0.2%). Showed that a row-based crop cleaner **deleted real content** (an option's own frame, an
L-shape). That cleaner was discarded.

**Stage 3e – extraction v2 and caching.**
- Question blanking grows over ink runs near the option columns to remove frame sides, with a width guard.
- `clean_crop2` removes only edge-touching components that are elongated or small slivers, and falls back to the raw crop if more than 60% of the ink would be removed.
- Writes the cached `.npz`. Final usable subset: **1,413 items** (1,261 skipped for non-"question mark" prompts, 34 not parsed, 2 empty question). Cleaning applied to 3,924 of 5,652 crops (mean 24% of ink removed). Answers are balanced (A 339, B 368, C 353, D 353).

**Stage 4 – verifier baseline.**
- Two-tower CNN: option encoder (4 conv blocks → pooled 3×3 → 128-d) and question encoder (pooled 3×6 → 128-d); fused as `[q, o, q⊙o]` → MLP → logit.
- Group-aware split: 989 train / 211 val / 213 test; 5 groups of items with identical option sets kept together.
- AdamW, lr 1e-3, weight decay 1e-4, batch 32 puzzles, 30 epochs, best-validation checkpoint, test evaluated once per run.
- Controls: an **option-only** model, and a **shuffled-question** test for the two-tower model.

## Results

Chance is 25%. With 213 test items, the 95% half-width on a single accuracy is about ±6 points.
Values are mean ± std over 3 seeds (same test set for all seeds, so seeds are not independent evidence).

| Model | Options | Val | Test |
|---|---|---|---|
| question + option | cleaned | 0.294 ± 0.007 | 0.247 ± 0.021 |
| question + option | raw | 0.308 ± 0.004 | 0.252 ± 0.008 |
| option only | cleaned | 0.294 ± 0.022 | 0.252 ± 0.012 |
| option only | raw | 0.316 ± 0.020 | 0.285 ± 0.013 |

Shuffled-question test for the two-tower models: test accuracy was **identical** to unshuffled
accuracy in all six runs.

Single sanity run (two-tower, cleaned options, seed 0): train loss 1.05 → 0.21 over 30 epochs while
validation accuracy fluctuated between 0.19 and 0.30; best val 0.299 (epoch 19), test 0.230, 53 s on a T4.

Tested environment: Google Colab, Tesla T4 (14.56 GB), Python 3.13, PyTorch 2.11 (CUDA 13.0),
`datasets` 4.8.5, Pillow 11.3.0, Matplotlib 3.10.0.

## Interpretation

- **No generalisation.** Training loss falls steeply (memorisation) while validation accuracy shows no trend. All test accuracies are within noise of chance.
- **Validation is optimistic.** Selecting the best of 30 noisy epochs inflates validation accuracy (0.29–0.32 versus 0.25 on test). Use test numbers.
- **No evidence of option-level shortcuts.** The option-only/raw mean of 0.285 is inside the chance band and should not be read as a finding.
- **The question branch is inert.** Identical accuracy under shuffled questions means no prediction changed. Whether the branch collapsed during training or there is an implementation bug has **not** been confirmed.
- **Consistent with, but not a test of, the paper.** The paper itself states that task-specific models perform near chance on these puzzles. The missing ingredients relative to GenVerOpt — Reptile across paradigms, candidate-in-grid scoring, the unspecified architecture — were not reproduced.

## Reproducing the experiments

Notebooks target Google Colab with a GPU runtime. Paths are hard-coded to `/content`.

```text
1. Open genveropt_stage3a.ipynb in Colab, run all cells (downloads ~116 MB parquet).
2. Run genveropt_stage3b.ipynb → genveropt_stage3c.ipynb → genveropt_stage3d.ipynb (diagnostics; optional but informative).
3. Run genveropt_stage3e.ipynb. Produces /content/genveropt_mmiq_usable.npz.
4. Run genveropt_stage4.ipynb. Uses the .npz from step 3; ~1 min per run on a T4, 12 runs in the grid.
```

Notes:
- Stages 3b–3e each re-download the dataset and re-define the helper functions they need; they are self-contained but duplicate code.
- Runtime state is lost on Colab reset. Set `use_drive = True` in Stage 3a's config if you want persistence, and copy the `.npz` into Drive manually for later stages.
- Seeds are fixed (`0` for data, `0/1/2` for runs), but GPU nondeterminism was not controlled.
- Notebooks may be committed with outputs for provenance, or stripped with `nbstripout`.

## Data

- **Source used:** the Hugging Face mirror `afdsafas/MMIQ` (2,710 rows; columns `image`, `problem`, `solution`).
- **Not verified against the official MM-IQ release.** Row count matches the 2,710-item test set described in the MM-IQ paper, but provenance and labels were not checked against the authors' distribution.
- **Answer format:** `solution` is `<answer> X </answer>` with `X ∈ {A, B, C, D}`.
- **No paradigm labels** are present, which prevents recreating the paper's category-based split.
- The dataset is **not** redistributed here. Check the dataset's own terms before reuse.

## Known issues and limitations

1. **Protocol mismatch** (see deviations): numbers are not comparable with the paper.
2. **Label-order assumption unverified.** Options are assumed to be A–D from left to right in label order. Spot checks were inconclusive by eye.
3. **Extraction is heuristic.** It depends on a detectable A–D label row. 34 of 1,449 question-mark prompts failed to parse; correctness of the other crops was only spot-checked on six items.
4. **Crop cleaning is partial.** Where an option's frame is a single connected component, rules are not removed. The 60% removal guard is arbitrary.
5. **Question images retain small frame stubs** in some layouts.
6. **Near-duplicates not checked.** Only exact-duplicate option sets (5 groups) were handled in the split.
7. **Small evaluation set.** About 213 test items; differences of a few points are not distinguishable.
8. **Shuffled-question anomaly unresolved.**
9. **Stage 3c figure anomaly.** The question column rendered blank in one figure for reasons not identified; the same extraction code produced non-empty questions in later stages, so it is treated as a plotting issue, unconfirmed.
10. **Prompt types.** 1,261 items whose prompts do not mention a question mark (other rule types) are excluded.

## Notes on the paper

Observed while implementing; flagged for anyone following along.

- **Eq. 2 is malformed as printed.** The verifier loss is written as `v·log c − (1−v)·log(1−c)` with `c ∈ {0,1}`, which gives `log 0`. The standard form is `−[c·log v + (1−c)·log(1−v)]`, used here.
- **Eqs. 5–7** write the verifier as `v(·; θ)` where `φ` appears intended.
- **Sign conventions in Eq. 1** (min over θ, max over φ) do not cleanly match Eqs. 2–3.
- **Eq. 6** does not state whether the final solution is the best sample seen or a sample from the converged generator.
- **Ablation table (Tab. 3)** has ambiguous column headers ("Reptile" vs "Cross-task training").
- **Human baseline.** The text cites 56% human accuracy; Fig. 3 reads closer to 0.54; MM-IQ sources reviewed give about 51%.
- **Compute.** The reported reasoning cost is about 26 A100-hours per sample for the medical tasks (Tab. 6, as read here), and pretraining 672 A100-hours — outside the scope of this repository.


## References

1. S. U. Saeed, Z. Tse, G. Slabaugh, D. C. Alexander, Y. Hu. *Can machines reason non-verbally?* arXiv:2610.13925v1 [cs.CV], 9 Oct 2026.
2. H. Cai, Y. Yang, W. Hu. *MM-IQ: Benchmarking Human-Like Abstraction and Reasoning in Multimodal Models.* arXiv:2502.00698, 2025.
3. A. Nichol, J. Schulman. *Reptile: a scalable metalearning algorithm.* arXiv:1803.02999, 2018.
