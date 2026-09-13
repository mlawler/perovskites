# SciLM: a Scientific Language Model for Electrolyte Formulations

A preliminary design for an any-to-any transformer over the IBM SMI-TED-IC ionic
conductivity dataset (`data/IBM_SMI_TED_IC/data/`). The model should be able to
consume any subset of `{6 component SMILES, 6 component percentages, temperature,
conductivity}` and predict the rest.

## 1. Dataset

- `train.csv` (8,745 rows), `valid.csv` (2,187), `test.csv` (2,734). `features.csv`
  is a byte-identical duplicate of `train.csv`; ignore it. 80/10/10 random split
  per Zohair et al. (2025).
- Each row: a `<sep>`-delimited string of exactly six `(SMILES, percentage)` pairs,
  plus a temperature in °C and a conductivity scalar `ionic`.
- **Units, confirmed from the paper** (npj Comput. Mater. 11:283, 2025, Methods):
  - Percentage = **mol %** (salt 5–15 mol%; solvents fill the remainder).
  - Temperature = °C, dataset range reported as **−70 °C to 200 °C**.
  - `ionic` column = **log₁₀(σ) with σ in μS/cm**. The paper's Methods say
    conductivity was log-transformed but does not state the pre-log unit; the
    actual range in our CSVs is [−3.04, +4.58], and cross-referencing Fig. 6
    (σ in mS/cm, max ~20) and Fig. 4d (threshold "Above 10 mS/cm" = 10⁴ μS/cm)
    fixes the unit as μS/cm. Conversions:
    - σ [μS/cm] = 10^(ionic)
    - σ [mS/cm] = 10^(ionic − 3)
    - σ [S/cm]  = 10^(ionic − 6)
    Sanity: train split min/max = −3.04 / +4.58 → 0.001 to 38 mS/cm.
  - Duplicate (formulation, T) entries were averaged during curation; outliers
    flagged via box-plot inspection were removed.
- Only **68 unique SMILES** appear across all splits (paper reports 15 salts + 51
  solvents = 66 in the curated pool; our 68 is consistent). Unused slots are
  padded with `(O, 0)` (water at 0 mol%). Slot 0 is always a Li salt; slots 1–5
  are solvents/additives.
- SMILES token-length distribution (using the SMI-TED atom-level regex):
  min 1, median 10, p95 32, **max 40**. Longest: `F[B-](F)(F)OC(C(F)(F)F)(C(F)(F)F)C(F)(F)F.[Li+]`.

## 2. Tokenization

Every element of a formulation is a **`(token_string, scalar)` pair**, where
`token_string` has fixed length `L = 41` (= max observed atom-token length 40
plus one for the explicit `<eos_smiles>` terminator). Three kinds of pair:

| Kind           | token_string                                   | scalar          |
|----------------|------------------------------------------------|-----------------|
| Component pair | atom-tokenized SMILES, `<eos_smiles>`, `<pad>`…| percentage (0–100) |
| Temperature    | `<temp>`, `<pad>` × (L − 1)                    | T in °C          |
| Conductivity   | `<ionic>`, `<pad>` × (L − 1)                   | σ (log-scaled)   |

A full row becomes **8 pairs** (6 components + temperature + conductivity),
laid out as a contiguous grid of `8 × L = 328` token positions, each carrying a
scalar. All L positions within one pair share the same scalar value.

### Atom-token vocabulary

Reuse SMI-TED's regex so atom tokens line up 1-to-1 with its vocab (see
`data/IBM_SMI_TED_IC/smi_ted_light/bert_vocab_curated.txt`, 2,393 tokens: `C`,
`c`, `(`, `)`, `[C@H]`, `[Li+]`, `[B-]`, digits for ring-closures, etc.).

Added special tokens:

- `<pad>` — inherited from SMI-TED vocab
- `<eos_smiles>` — marks end of a real SMILES within its L-block
- `<temp>` — marks a temperature pair
- `<ionic>` — marks a conductivity pair
- `<mask_tok>` — replaces masked atom tokens during training
- `<mask_num>` — sentinel token at a position whose *scalar* is masked
  (the token channel is unaffected, but we signal "scalar is hidden here")

### Why `<eos_smiles>`?

A SMILES of length k < L is written as `tok_1 … tok_k <eos_smiles> <pad> <pad> …`.
At inference, when the model generates a component, it emits tokens until
`<eos_smiles>` and we ignore everything after. Without this marker, a generated
`<pad>` followed by a non-pad token is ambiguous — error or restart. `<eos_smiles>`
keeps `<pad>` pure ("don't care") and gives termination its own clean signal.

## 3. Embeddings

Every input position contributes three learned/computed embeddings that sum:

1. **Token embedding** `E_tok[token_id]`, initialized from SMI-TED's pretrained
   atom-token embedding table where the vocab overlaps (free transfer of 91M-molecule
   atom chemistry). Special tokens are randomly initialized.
2. **Scalar embedding** `E_num = MLP(x)` — a small MLP mapping the scalar x to a
   vector of `d_model` dimensions. Plain MLP is enough: percentages are bounded
   [0, 100] with dense coverage, temperature and conductivity are not periodic or
   translation-invariant so Fourier features buy little for the preliminary model.
   If and when we want to extrapolate off-distribution in T, we revisit.
   Before feeding to the MLP, each scalar is **standardized** per pair-type (%,
   T, σ) using train-set mean/std. This also mitigates numerical-scale imbalance
   (Zohair et al. note the same concern).
   At **masked-scalar positions** we replace the scalar-embedding contribution
   with a **learned `E_mask_num` vector** of dimension `d_model`, rather than
   encoding any particular numeric value. This avoids the problem that zero (or
   any in-range value) would be indistinguishable from a real observation;
   percentage 0 occurs on every water-pad slot, T = 0 °C is plausible, and
   σ = 10⁰ μS/cm is physical.
3. **Positional embeddings**, two additive components:
   - **Slot-position** `E_slot[s]` with s ∈ {0, …, 7}: identifies which pair (salt,
     solvents 1–5, temperature, conductivity) the position belongs to.
   - **Intra-slot position** `E_pos[p]` with p ∈ {0, …, L−1}: identifies position
     within the L-block.

Input to the transformer at position i: `E_tok(tok_i) + E_num(x_i) + E_slot(s_i) + E_pos(p_i)`.

## 4. Architecture

Encoder-only transformer (BERT-style), masked-prediction objective. Start small:

- `d_model = 256`, 6 layers, 8 heads, FFN dim 1024, dropout 0.1.
- Standard multi-head self-attention over the full 320-position sequence; an
  attention mask zeros out `<pad>` positions so they contribute nothing to the
  context of non-pad positions.
- Two output heads per position:
  - **Token head:** linear → softmax over atom vocabulary. Active at positions
    where the token was masked.
  - **Scalar head:** linear → scalar regression. Active at positions where the
    scalar was masked (currently only the first position of a pair carries the
    "live" scalar signal, since all 320 positions within a pair share x; we predict
    at that first position to avoid redundancy).

## 5. Training objective

**Pair-level masking only** for the preliminary model. Two mask operations, sampled
per training example:

- **(a) Mask a whole pair.** Replace all L tokens with `<mask_tok>` and the scalar
  with "unknown". The model must predict both the token_string (softmax at each
  of the L positions) and the scalar (regression at the pair's anchor position).
- **(c) Mask only the scalar of a pair.** Leave the token_string visible, replace
  only the scalar. The model predicts the scalar via the regression head.

Per training example, sample 1–3 pairs to mask using a mixture of (a) and (c).
Loss = cross-entropy on predicted atom tokens (averaged over masked token
positions) + **λ** · MSE on predicted scalars (averaged over masked scalars,
computed in standardized-scalar space so percentages, T, and σ contribute
comparably). **Start with λ = 1** and re-tune after the first training run by
inspecting the relative magnitudes of the two loss terms on the validation set —
if one dominates by more than ~5×, rebalance.

**Deferred for later (not in the preliminary model):**

- **(b) Within-SMILES token masking** for learning SMILES grammar conditional on
  surrounding context. Useful when we want to generate novel molecules.
- **Slot-permutation augmentation** across solvent slots 1–5 (they are exchangeable,
  which multiplies effective data by 5! = 120). Low cost to add when we need it.

## 6. Bidirectional generation ("any-to-any")

At inference, the input sequence always has all 8 pair-positions present (the
layout is fixed). "Asking the model to predict X" means X's token_string gets
`<mask_tok>` in every non-marker position of its block and X's scalar channel
uses `E_mask_num`. "Supplying X" means the block carries X's real tokens and its
real scalar. The user can mask any subset — including, as the basic use case,
just the `<ionic>` pair (predict-conductivity mode) or all six solvents plus
`<ionic>` (predict-formulation-given-T mode).

- Scalars are read from the regression head at anchor positions.
- SMILES are generated by reading argmax tokens from the token head, truncating at
  `<eos_smiles>`. Because `<eos_smiles>` is part of the vocabulary the model was
  trained to emit, the generated string is self-terminating.

For generating component SMILES, since only 68 molecules exist in training data,
a **retrieval fallback** is cheap and useful: compare the predicted 40-token
distribution against the 68 known SMILES and return the best-matching one when
the raw decode is off-vocabulary. The preliminary model should at least log this
retrieval-nearest-neighbor alongside its raw output so we can evaluate both.

## 7. Training-sequence construction

Each training example is built from a single row of `train.csv`, but the
formulation's six component slots are **exchangeable** (the 6 slots have no
intrinsic order beyond "salt is always slot 0"). Zohair et al. augment by
generating all permutations of solvent slots 1–5 to make the model
order-invariant; we do the same at a lighter touch:

- **Slot-permutation augmentation.** For each training example, at load time,
  randomly permute slots 1–5 (5! = 120 orderings). Do this per epoch, per example,
  rather than materializing all 120 copies. Cheap; biggest win available.
- **Random row shuffling across epochs.** Standard DataLoader shuffling.

### Multi-example "prompt" sequences (planned extension, not preliminary v1)

Longer term, we want to query the model with a *context* of several example
formulations and have it predict the next one. Sorting the context by
conductivity (ascending or descending) would encode the user's intent — e.g.
"extend this descending-σ trend" to propose a new low-conductivity formulation.
This is a prompt-engineering capability closer to in-context learning / TabPFN-
style prior-fitted networks than to our single-example masked objective.

To support it, the model must see such sequences during training, not just at
inference. The plan:

- Concatenate K formulations (K ≤ 10 for a first pass) with a `<sep_example>`
  token between them, yielding sequences of K × 320 + (K−1) positions.
- Extend the slot-position embedding to a `(example_index, pair_index)` 2-D
  scheme so the model can distinguish "which pair of which example" it is at.
- Mix three training-sequence orderings in roughly equal proportion:
  1. **Random order** (no structural signal) — forces the model not to *assume*
     an ordering.
  2. **Sorted by ionic, ascending** (low → high σ).
  3. **Sorted by ionic, descending** (high → low σ).
  A smaller share sorted by temperature can be added later if useful.
- Mask fields *only in the last example* of the sequence (next-example
  prediction), in addition to the single-example masking (a) and (c) from §5.
- Training-sequence sampling: draw K uniformly from {1, …, K_max}. K = 1 is the
  pure single-example case already covered by §5, so the multi-example mode is
  a strict superset of v1.

**Scope call for the preliminary model:** build v1 with single-example inputs
(K = 1) and masking (a)+(c). Add multi-example training in v2 once the tokenizer,
embeddings, loss, and training loop are debugged on the simpler case. Keeping
the architecture agnostic to K (no baked-in assumption that K = 1) costs nothing
now and buys the extension cleanly later.

## 8. Relation to the IBM SMI-TED work

IBM fine-tuned SMI-TED (a 289M-parameter SMILES foundation model) as a one-way
regressor: formulation-as-one-long-string + temperature → conductivity, with RMSE
≈ 0.109. Their tokenizer treats `<sep>` and `%` as unknowns and encodes percentages
as digit characters, so numeric structure is lost. SciLM keeps the same atom-token
vocabulary (and can initialize from SMI-TED's embeddings) but puts every scalar
on the token representation itself, and learns a bidirectional objective instead
of a unidirectional regression.

A cross-representation translator between SciLM and SMI-TED is mechanical, not
learned: strip SciLM scalars, drop `<eos_smiles>`/`<temp>`/`<ionic>` markers,
concatenate the six component SMILES with literal `<sep>`, and the result is
exactly the input string SMI-TED expects. Useful if we later want to ensemble
SMI-TED's pooled formulation embedding with SciLM's.

## 9. Settled design choices (this revision)

- **Units:** log₁₀(σ/μS/cm), temperature in °C, percentages in mol %
  (confirmed against Zohair et al., npj Comput. Mater. 11:283 (2025)).
- **λ:** start at 1.0, rebalance after one training run if the two loss terms
  disagree in magnitude by more than ~5×.
- **Masked-scalar representation:** learned `E_mask_num` embedding vector, not
  a reserved numeric value. The stored scalar at a masked position is ignored.
- **Inference layout:** sequence always contains all 8 pair-positions; masking a
  pair means replacing its tokens with `<mask_tok>` and its scalar contribution
  with `E_mask_num` — the pair is never *dropped* from the input.
- **Nearest-SMILES decoding (cross-entropy scoring):** at inference, for every
  predicted component pair, score each of the 68 canonical inventory SMILES by
  the cross-entropy of its token sequence under the model's predicted
  per-position distribution, and return the lowest-CE match. This is principled
  (it is the log-likelihood of each candidate), only ~68 vector products per
  prediction, and avoids the ambiguity of raw argmax decoding when the model is
  uncertain between two nearby molecules.

## 10. Remaining open items

- **Standardization:** compute means/stds of %, T, and σ from the train split at
  the start of training and freeze them; apply the same stats at validation and
  inference time. Store alongside the checkpoint.
- **Learning rate:** to be determined by a short sweep on the first training
  runs (standard practice for a transformer of this size is 1e-4 to 5e-4 with
  AdamW and a warmup-then-cosine-decay schedule, but we should confirm
  empirically rather than commit now).

## 11. Scope: what "Scientific Language Model" should mean

The v1/v2 architecture is the minimum viable SciLM: any-to-any masked prediction
over **one** formulation. The full vision of a scientific language model for
electrolytes is broader: it should reason over **sequences of formulations** the
way a chemist reads a paper or a lab notebook — *"given these five recipes and
their measured σ at these temperatures, propose a sixth recipe with higher σ."*

This kind of in-context, example-conditioned generation is a v3 research project,
not a tweak. The current model cannot do it for four concrete reasons:

1. **Fixed single-formulation context.** The transformer attends over exactly
   `n_pairs × L = 8 × 41 = 328` token positions, all encoding one formulation.
   `E_slot` has 8 IDs and `E_pos` has 41 — there is no embedding to distinguish
   "formulation #2" from "formulation #1" in a longer context. Concatenating
   would produce a single chimeric formulation with 12 components.
2. **No autoregressive head.** The model is encoder-only, with independent
   per-position token logits. It cannot generate a coherent novel SMILES; it
   can only return a probability distribution at each masked position.
3. **Training distribution is single-example.** Each sample seen in training is
   one formulation with a random subset masked. The model has never been
   exposed to "K formulations, the K-th masked", so there is no in-context
   behavior to elicit even with creative prompting.
4. **Per-kind scalar heads assume one slot per kind.** `W_num` has three heads
   (component %, temperature, ionic) and is indexed by `pair_kind`, which is a
   single integer per pair. Generalizing requires either repeating the heads
   per formulation or moving to a single shared head with a kind embedding.

### What v3 would require

- **Hierarchical sequence layout.** A flat sequence of `K × n_pairs × L` tokens
  with three orthogonal embeddings: formulation_id (∈ `1..K`), slot_id, position.
- **Multi-example training data.** Construct training sequences as
  *"K formulations from the same chemistry family or paper, with the K-th
  masked in the slots we want the model to predict."* The "from the same
  family" condition is what makes the prior examples informative — without
  it, the model learns to ignore the in-context part.
- **Generative head.** Either (a) an autoregressive decoder that emits SMILES
  token-by-token conditioned on the encoder context, or (b) keep the
  encoder-only design and pair it with an external SMILES validator (e.g.,
  RDKit) plus beam search over masked-token predictions.
- **Targeted training objective.** A "predict-better-σ" objective rather than
  generic masked-scalar prediction — e.g., construct sequences where
  σ-of-formulation-K > max(σ-of-formulations-1..K-1), so the model learns the
  *trend* and not just the mean.

### Inverse-design paths that work with v2 today

Without retraining, three workflows already exploit v2 for "find a formulation
with high σ":

1. **Forward screening.** Enumerate or sample candidate formulations, predict
   σ at fixed T, rank. Trivially parallel; the model evaluates ~100/s on CPU.
   Defensible because it tests v2 only on the task it was trained for.
2. **Gradient ascent on mole fractions.** Hold SMILES tokens fixed, treat the
   `pct` scalars as continuous, backprop ∂σ_pred/∂pct, project onto the
   simplex (mole fractions sum to 100 %). Optimizes *concentration* of a known
   recipe — this is what the Zohair paper does for its identified chemistries.
3. **Single-formulation generative.** Mask salt and/or solvent slots, take
   top-k token predictions per masked position, compose, validate with RDKit.
   Limited because token logits are independent per position, but works for
   short common motifs and gives a discrete shortlist of synthesizable
   candidates.

Path 1 is the most defensible "first experiment." Path 3 is the closest current
proxy to the v3 in-context-design capability above.

## 12. Application: SciLM-for-perovskites alongside Clancy's PAL 2.0 (Genesis proposal, Apr 2026)

The DOE Genesis Mission proposal (Phase I, 9 months, focus topic 1-B) pairs
this group's SciLM work with Paulette Clancy's PAL 2.0 Bayesian-optimization
framework. The perovskite SciLM proposed for Phase I is the **v3 architecture
from §11** — multi-formulation K-length context window, any-to-any masked
prediction, generative inverse design — applied to (composition, processing,
multi-modal characterization, post-irradiation properties) for radiation-hard
metal-halide perovskites. The proposal text itself is non-specific about
v2/v3 internals; it commits only to multi-modal capability (the first claim
worked out below). The v2 electrolyte SciLM (8,745-row IBM SMI-TED-IC
dataset, R² = 0.89 on test conductivity, ~6 h CPU training) is the
preliminary-data demonstration that the architectural family works on real
chemistry.

### Strategic framing

PAL 2.0 is the senior collaborator's well-established BO method. Reviewer risk:
"BO with a wrapper" is not Genesis-class novelty. SciLM is the proposal's
distinctively *new* AI contribution — but pitched as **complementary** to PAL,
not competing with it. The division of labor must rest on *what each method is
mathematically equipped to do*, not on stage labels:

- **PAL 2.0 (selection):** calibrated-uncertainty acquisition in continuous
  low-dimensional parameterized spaces. Sample-efficient in small-data regime.
  Owns the per-round next-experiment decision.
- **SciLM (representation, prior, generation):** multi-modal data fusion,
  pretraining-derived chemistry prior, generative proposal of structurally
  novel candidates outside PAL's parameterization.

PAL retains every operational role it has now. SciLM adds capabilities PAL
categorically lacks. This is honored by leaving Fig. 1 (the closed-loop
diagram) as Clancy's, and by stating BO's calibration advantage explicitly so
the division of labor reads as *technically driven*, not political.

### The three capability claims (proposal text, in `\draft{}`)

What the proposal commits SciLM to *deliver*. Tight on capabilities,
deliberately loose on operational structure (so we retain implementation
flexibility):

1. **Multi-modal data fusion.** Joint encoding of composition, processing, and
   measured properties (XRD, SRPL, TRPL, BLDS, ionic conductivity, post-
   irradiation observables) into a shared learned representation that replaces
   PAL 2.0's hand-crafted physico-chemical fingerprint. Hooks Genesis 1-B
   language verbatim ("integrate datasets from multi-modal synthesis,
   characterization, and fabrication techniques").
2. **Pretraining-derived prior over composition space.** Initialize from the
   existing perovskite-property literature so PAL has a literature-informed
   prior, which it currently does not (PAL's only prior is feature engineering
   plus GP kernel/mean choices).
3. **Generative proposer of search-space expansions.** Conditional generation
   of structurally novel candidates (new additives, cation/anion combinations)
   outside PAL's current discrete parameterization. Capability claim only —
   **how** PAL consumes these proposals is left to implementation.

Plus one architecture argument: all three capabilities are realized by a
single any-to-any transformer trained once. Feasibility argument for Phase I.

### Integration options (kept out of the proposal text on purpose)

We discussed three integration mechanisms with PAL. The proposal text commits
to none; this list is for our own planning:

- **Option A — Chemistry-aware kernel in PAL.** Replace PAL's GP kernel with
  Tanimoto-on-Morgan or a graph kernel so PAL's GP can natively evaluate
  structurally novel candidates. Cleanest. Requires modifying PAL 2.0 (Clancy
  buy-in needed). Flagged as Phase II ambition.
- **Option B — Restrict SciLM proposals to PAL's parameterization.** No PAL
  modification, but kills the "expand the search space" claim that makes
  SciLM categorically different from BO. Rejected.
- **Option C — Two-tier with structural proposals as outer channel.** SciLM
  periodically proposes structurally novel candidates which, after curation,
  are added to PAL's parameterization as new categorical levels for subsequent
  rounds. No PAL modification. Realistic for 9 months. **This is our working
  assumption for Phase I**, but "outer loop" and "curation" language was
  removed from the proposal text per the user's "we only get one shot" rule —
  the proposal commits to capabilities, not operational structure.

### Capabilities that look operationally distinct from BO but aren't

Pitfalls we considered and ruled out as proposal claims:

- **"In-context / few-shot reasoning."** Sounds distinct from BO's
  fit-then-predict surrogate but isn't, since BO's surrogate also predicts
  round N+1 from rounds 1..N. The actually-distinct capability hiding behind
  this label is *transfer learning from external corpora* — i.e., pretraining.
  Counted under capability 2 (pretraining-derived prior) and not separately.
- **"Generation = expanding the search space."** Philosophically yes. In
  practice not collapsible to BO because (a) PAL 2.0's GP doesn't scale to
  combinatorial molecular-graph spaces, (b) generative search spaces are
  implicitly defined by the data distribution and can't be written down as a
  parameterization, (c) inverse maps in chemistry are one-to-many and need
  distribution sampling, not function maximization.

### Compute budget allocated for this work

500 GPU-hours on Kestrel (NREL) = 12,500 AU at 100 AU per GPU node-hour
(4 H100s/node). Row 17 of `compute_costs/Genesis_Application_ACFoster_Apr2026.xlsx`.
Covers v2-class perovskite-SciLM training runs (~30–80 GPU-hours per run),
hyperparameter sweeps, inverse-design experiments, and active-learning
retrains over the project's incoming ~10–15 samples/day. Does *not* cover
foundation-model-style pretraining at PubChem scale (that would be a separate
~2,000–5,000 H100-hour ask, scoped as future work).

### Why this is deliverable in 9 months

- **Architectural precedent exists**: SciLM v2 (electrolytes, R²=0.89 on
  conductivity prediction) demonstrates the any-to-any architecture works on
  a real chemistry dataset with from-scratch training in ~6 hours of CPU.
  This is the preliminary data presented in the proposal and what makes the
  v3 perovskite scope credible.
- **Inverse-design precedent exists**: CascadesDB inverse problem (predict
  undamaged-sample XRD from damaged-sample XRD) demonstrates the inverse-
  prediction approach on a defect/damage dataset. Same problem shape as
  inferring composition from radiation-hardness measurements.
- **No PAL modifications required for Phase I** under Option C; PAL stays
  exactly as Clancy delivers it.

### Risk of v3 commitment

v3 introduces a longer context window and multi-example in-context training
distribution that v2 does not — both are research-project additions to the
v2 stack (§11 lists what would change). 9 months is enough if the v3 work is
focused on the K-length training distribution and standard transformer-scale
hyperparameters, not on novel architectural research. If the proposal text
remains non-specific about v2/v3, we keep the option to fall back to a v2-
class deliverable in the perovskite domain without contradicting the
narrative — useful insurance.

### What would invalidate this plan

If a reviewer or collaborator pushes for explicit operational integration
(Option A) in Phase I, the budget and timeline tighten significantly: a
chemistry-aware GP kernel inside PAL is a 2–3 month methodological project
in its own right. Default response: hold the line that Phase I demonstrates
the *capabilities* and Phase II integrates them.
