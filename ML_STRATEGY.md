# Machine Learning Strategy for Radiation-Hard Perovskite Discovery

This document summarizes the ML options, compute requirements, and proposed
modifications to the closed-loop materials discovery workflow for identifying
radiation-hard perovskite compositions within a 6-month timeline.

---

## 1. Context

### The scientific problem

Metal halide perovskites (MHPs) show surprising radiation tolerance in space
(Delmas et al., 2023), but the atomic-scale mechanisms are poorly understood.
Clancy et al. (2026) showed via MD simulation of CsPbI3 that the soft PbI6
octahedral network absorbs radiation energy through collective tilting and
spinning, with ~32% of displaced halides self-healing within 10 ps. The goal
is to exploit this understanding to discover optimally radiation-hard
perovskite compositions.

### The Gienger closed-loop workflow

The reference workflow (Gienger et al., JHU/APL) uses a closed-loop cycle
for materials discovery:

1. **Surrogate model** (XGBoost) predicts a scalar property (hardness) from
   composition
2. **Acquisition function** (PAL 2.0) selects the next candidate composition,
   balancing exploration vs. exploitation
3. **Experiment** — synthesize the candidate (arc melting) and measure the
   property (Vickers indentation)
4. **Update** — add the new data point and retrain the surrogate
5. Repeat

This loop was designed for multi-principal-element alloys (MPEAs) with scalar
hardness targets. Adapting it for perovskite radiation hardness requires
changes to handle richer data types (spectra, images) and a different
material system.

### Available datasets

| Dataset | Samples | Format | Source |
|---------|---------|--------|--------|
| CascadesDB cascade pairs | 2,187 | Before/after RDF + XRD spectra (500-dim each) | IAEA, 6 metals |
| Perovskite stability pairs | 6,407 | Before/during/after scalar triples | NOMAD, 2,196 compositions |
| Clancy CsPbI3 MD trajectories | ~8 | Full atomic trajectories (~500K atoms, 1–1000 eV) | Collaborator data |
| Future experimental data | TBD | XRD spectra, TEM/SEM images, scalar metrics | Phase II experiments |

---

## 2. Limitations of the Current XGBoost Approach

XGBoost operates on **pre-engineered scalar features** and predicts **scalar
targets**. For radiation-hard perovskite discovery, this creates several
limitations:

- **Cannot handle spectral inputs/outputs.** XRD patterns (500-dim curves)
  must be manually reduced to scalar features (peak heights, widths). This
  is lossy and requires domain expertise to choose the right features.
- **Cannot predict spectra.** The model outputs a single damage score, not
  the full post-damage diffraction pattern. Scientists cannot sanity-check
  predictions against physical intuition.
- **No transfer learning.** Each material system starts from scratch. Knowledge
  gained from CascadesDB (metals) cannot be transferred to perovskites.
- **No natural inverse problem.** Given experimental before/after data, XGBoost
  cannot search for the MD simulation parameters that explain the observation.

---

## 3. Neural Network Alternatives

### Option A: Drop-in scalar surrogate (minimal change)

Replace XGBoost with a neural network that predicts the same scalar damage
metric, but with calibrated uncertainty via an ensemble or MC dropout.

- **Pros:** Minimal change to the loop. Better uncertainty estimates enable
  smarter acquisition (explore where uncertain, exploit where confident).
- **Cons:** Still operates on scalars. Doesn't leverage the richness of
  spectral/image data. Doesn't justify H100s.
- **Compute:** Single GPU, trains in minutes.

### Option B: Spectral surrogate (recommended single change)

Replace the scalar surrogate with a neural network that predicts the **full
post-damage XRD spectrum** given (composition, radiation conditions).

- **Input:** Composition vector (~30-dim: ion fractions, lattice params) +
  radiation conditions (PKA energy, temperature)
- **Output:** Predicted XRD spectrum after damage (500-dim)
- **Architecture:** 1D convolutional decoder or transformer, conditioned on
  composition/conditions via FiLM layers or cross-attention
- **Uncertainty:** Ensemble of 5–10 models; disagreement between ensemble
  members quantifies prediction confidence

**Why this is the highest-impact single change:**

1. Directly connects to experiment — predicted spectra can be visually
   compared to measured XRD
2. The acquisition function optimizes for spectral similarity (pre vs. post
   damage XRD), which *is* radiation hardness defined physically
3. Enables the inverse problem: given experimental XRD, search over
   composition space for the best match
4. Justifies H100-scale compute for training
5. Everything else in the loop stays the same

### Option C: Full atomic configuration surrogate (maximum capability)

Train an E(3)-equivariant graph neural network (e.g., MACE, NequIP) that
predicts post-damage **atomic positions** rather than summary spectra.

- **Input:** Pre-damage atomic graph (~500K nodes) + radiation conditions
- **Output:** Post-damage atomic graph
- **Architecture:** Equivariant message-passing GNN with ~10M parameters

**Advantages over spectral surrogate:**

- Preserves full 3D spatial information (cascade shape, depth, defect
  topology) that RDF/XRD projections lose
- The inverse problem is better constrained — fewer atomic configurations
  map to the same observables than spectra
- Can compute any observable (RDF, XRD, Steinhardt parameters, defect
  counts, octahedral tilts) from the predicted configuration

**Disadvantages:**

- Requires significantly more compute (H100-scale for both training and
  inference)
- Harder to validate against experiment (experiments don't give atomic
  positions)
- More complex to implement and debug

### Option D: Generative surrogate (diffusion/flow-matching model)

Train a conditional generative model that samples plausible post-damage
configurations, rather than predicting a single deterministic output.

- Naturally provides uncertainty through sample diversity
- Can generate multiple possible damage scenarios for the same conditions
- State-of-the-art but highest implementation complexity
- Requires the most compute (4 H100s fully utilized)

---

## 4. The Inverse Design Problem

The most scientifically exciting application is the **inverse problem**:

> Given experimental before/after data (XRD spectra, TEM/SEM images),
> infer what atomistic process occurred during radiation exposure.

### How it works

1. **Train a fast forward model** (Option B or C above):
   `(composition, PKA energy, conditions) → predicted observables`

2. **Collect experimental data**: XRD spectrum before irradiation, XRD
   spectrum after irradiation (and optionally TEM/SEM images)

3. **Optimize over simulation parameters**: Find the (PKA energy, angle,
   conditions) that minimize the mismatch between predicted and measured
   post-damage observables. This requires thousands–millions of forward
   passes through the surrogate — feasible in seconds with a neural network,
   impossible with actual MD simulations.

4. **Validate with real MD**: Run one full MD simulation at the inferred
   conditions. Confirm the predicted mechanism (e.g., octahedral tilting
   vs. amorphization) matches the experimental evidence.

### Multi-modal experimental data

Experimental measurements may include:

| Modality | What it captures | Network encoder |
|----------|------------------|-----------------|
| XRD spectrum | Long-range crystalline order, Bragg peak changes | 1D CNN |
| RDF (from PDF analysis) | Local bonding environment | 1D CNN |
| TEM/SEM image | Spatial damage morphology, grain structure | 2D CNN (ResNet) |
| Scalar metrics | PCE retention, defect density, conductivity | MLP |

A neural network can fuse these modalities — using whatever subset is
available for a given sample. More modalities = better constrained inverse
problem, but the approach works with XRD alone.

### Feasibility

The inverse search is feasible if:

- The forward model is accurate enough (requires sufficient training data)
- The inverse problem is well-posed (different damage processes produce
  distinguishably different observables)
- The search space is manageable (PKA energy, angle, temperature — a few
  continuous dimensions)

The main risk is degeneracy: multiple atomistic processes may produce
similar XRD patterns. Mitigation: use multiple observables (XRD + TEM),
use uncertainty estimates to flag ambiguous inversions, validate with MD.

---

## 5. Transfer Learning Pipeline

### Stage 1: Pretrain on CascadesDB (2,187 samples, 6 metals)

The model learns material-agnostic radiation damage physics:
- How RDF peaks broaden under cascade damage
- How XRD Bragg peaks suppress with increasing PKA energy
- How damage scales with system size and temperature
- General spectral signatures of crystalline → disordered transitions

### Stage 2: Fine-tune on CsPbI3 MD data (~8–16 simulations)

Adapt to perovskite-specific physics:
- Octahedral tilting and rotation as energy dissipation channels
- Halide sublattice mobility and self-healing
- ABX3 composition-dependent damage response

Even a small fine-tuning set is effective because the pretrained model
already understands damage — it just needs to learn how perovskites
are different from metals.

### Stage 3: Fine-tune on experimental data (Phase II)

Adapt from simulation to experiment:
- Bridge the sim-to-real gap (MD force fields are approximate)
- Learn from real XRD/TEM measurements
- This is where the 6,407 perovskite stability pairs provide additional
  pretraining signal — the model has seen composition → degradation
  relationships even if the stress mechanism was thermal/light rather
  than radiation

---

## 6. H100 Compute Budget

### What 4 H100s enable (320 GB total HBM)

| Task | Why H100s matter | Alternative without |
|------|------------------|---------------------|
| Equivariant GNN on ~500K-atom graphs | 80 GB per graph in memory | Impossible — graphs don't fit on consumer GPUs |
| Ensemble of 10 spectral surrogates | Train in parallel, 1 per GPU | Train sequentially on 1 GPU (10x slower) |
| Generative model (diffusion) | Large batch sizes for stable training | Feasible but very slow |
| Inverse search (millions of forward passes) | Batch inference across 4 GPUs | Still feasible on 1 GPU, just slower |
| Hyperparameter sweep | Run 40 configs in parallel | Sequential, weeks instead of days |

### What does NOT need H100s

- XGBoost or scalar neural network surrogates (laptop-scale)
- RDF/XRD computation from atomic configurations (CPU-bound, already done)
- The Gienger loop logic (acquisition function, candidate ranking)
- Data preprocessing and dataset construction

### Recommended allocation

- **2 H100s**: Training the spectral surrogate ensemble (Option B)
- **1 H100**: Inverse problem search / inference
- **1 H100**: Experimental / ablation studies, hyperparameter tuning

---

## 7. Six-Month Timeline

### Months 1–2: Build the surrogate

- Train spectral surrogate on CascadesDB (pretrain) + CsPbI3 MD data
  (fine-tune)
- Validate: does the model predict post-damage XRD that matches held-out
  MD simulations?
- Build the inverse search pipeline
- Deliverable: trained model + inverse search demo on synthetic data

### Months 3–4: First experimental cycle

- Use the surrogate to rank candidate perovskite compositions by predicted
  radiation tolerance (spectral similarity before/after)
- Synthesize top 3–5 candidates
- Irradiate and measure before/after XRD
- Run inverse search on experimental data to infer damage mechanisms
- Retrain surrogate with experimental data
- Deliverable: first experimental validation of ML predictions

### Months 5–6: Second cycle + analysis

- Updated surrogate recommends next round of compositions
- Synthesize and test 3–5 more candidates
- Compare ML-guided candidates vs. baseline (random or literature-based
  selection)
- Analyze: does the spectral surrogate find compositions that XGBoost
  would have missed?
- Deliverable: demonstrated radiation-hard composition + mechanistic
  understanding from inverse MD

---

## 8. Recommendation

**For a single change to the Gienger loop**: Replace XGBoost with a spectral
neural network surrogate (Option B). This keeps the loop structure intact,
directly connects predictions to experimental XRD measurements, enables the
inverse design problem, and justifies H100-scale compute. Everything else —
acquisition function, synthesis, characterization, retraining — stays the same.

**For maximum scientific impact**: Combine the spectral surrogate (Option B)
with the inverse design problem (Section 4). This not only discovers
radiation-hard compositions but explains *why* they are hard, connecting
experimental observables back to atomistic mechanisms like the octahedral
tilting discovered in the Clancy et al. paper.
