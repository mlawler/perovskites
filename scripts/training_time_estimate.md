# Training Time Estimate: 4 H100 GPUs on CascadesDB

Detailed computation of time and resources needed to train an ML model on
CascadesDB before/after atomic configuration pairs using 4 NVIDIA H100 GPUs.

---

## 1. Dataset Size

**CascadesDB raw numbers:**

| Metric | Value |
|--------|-------|
| Catalog entries | 320 |
| Total simulations (before/after pairs) | 17,024 |
| Atom count range | 13,500 – 35,152,000 |
| Weighted avg atoms/sim | ~930,000 |
| Raw archive size (compressed) | 163 GB |
| Estimated uncompressed | ~500–800 GB |

**Distribution of simulations by system size:**

| Atom count bracket | Simulations | % of dataset |
|--------------------|-------------|--------------|
| <100K atoms | 524 | 3% |
| 100K–500K | 10,378 | 61% |
| 500K–2M | 3,378 | 20% |
| 2M–10M | 2,632 | 15% |
| >10M | 112 | 1% |

**What "dataset size" means depends on the model architecture:**

- **Option B (spectral surrogate):** Each simulation is reduced to a
  before/after RDF+XRD pair — two 500-dim vectors plus a ~30-dim condition
  vector. Total training data is ~17K samples x ~1,030 floats x 4 bytes =
  **~67 MB**. Trivially fits in memory.

- **Option C (atomic GNN):** Each simulation is a graph with ~1M nodes. A
  single graph at 930K atoms requires storing positions (3 x float32), atom
  types, neighbor lists (~20 neighbors per atom), and edge features.
  **Per graph: ~300 MB in GNN-ready format.** For all 17K simulations:
  **~5 TB** of graph data.

---

## 2. GPU Memory: What 4 H100s Can Hold

Each H100 has **80 GB HBM3**. Total across 4: **320 GB**.

### Option B (spectral surrogate)

- Entire dataset (~67 MB) fits on a single GPU with room to spare
- Model (1D CNN decoder, ~5M params) = ~20 MB
- Batch of 512 samples with activations ~ ~100 MB
- **Memory is a non-issue.** You could train an ensemble of 10 models
  simultaneously, one per GPU, with memory left over.

### Option C (equivariant GNN on atomic graphs)

This is where memory becomes the binding constraint:

| Component | Memory per graph |
|-----------|-----------------|
| Node features (930K atoms x 128-dim embeddings) | ~450 MB |
| Edge index (~20 neighbors x 930K x 2 x int64) | ~280 MB |
| Edge features (spherical harmonics, ~20 neighbors x 930K x 64) | ~4.5 GB |
| Intermediate activations (message passing, ~6 layers) | ~5–15 GB |
| Gradients (roughly = activations) | ~5–15 GB |
| **Total per graph in training** | **~15–35 GB** |

A single average-sized graph (930K atoms) uses **~20–35 GB** during training
on a GNN like MACE or NequIP. An H100 can hold **1–2 such graphs per GPU**
at a time (batch size 1–2 per GPU). For the largest graphs (>10M atoms,
~35 GB+ just for node storage), you'd need **model parallelism across 2+
GPUs for a single graph**.

### Loading from disk

**Option B:** No disk I/O needed. The entire preprocessed dataset lives in
GPU memory.

**Option C:** Yes, constantly. 17K graphs x ~300 MB = ~5 TB of graph data.
You hold 1–2 graphs per GPU at any moment, cycling through from disk.

**I/O delay estimate for Option C:**

- NVMe SSD reads at ~3–7 GB/s
- Loading one 300 MB graph from disk: ~50–100 ms
- Each batch (4 graphs across 4 GPUs): ~100 ms I/O
- Forward + backward pass per batch: ~200–500 ms for an equivariant GNN on 1M atoms
- **I/O is ~20–30% of batch time** — significant but not dominant, assuming NVMe
- With prefetching (PyTorch DataLoader with `num_workers=4` and
  `prefetch_factor=2`), you can **hide most I/O latency**
- **If on network-attached storage (NFS, cloud EBS):** expect 500 MB/s–1 GB/s,
  making I/O **the dominant bottleneck** (~300–600 ms per graph load). Use a
  local SSD cache of preprocessed graphs.

---

## 3. Epochs and Time per Epoch

### Option B (spectral surrogate)

- **Dataset:** ~17K samples, split 80/10/10 -> 13.6K training samples
- **Batch size:** 512 (fits easily)
- **Batches per epoch:** ~27
- **Forward + backward per batch on H100:** ~1–2 ms (tiny model, tiny data)
- **Time per epoch:** ~0.05–0.1 seconds
- **200 epochs:** ~10–20 seconds
- **Ensemble of 10 models (4 GPUs):** 3 sequential rounds x 20 sec = **~1 min**

This is so fast that 200 epochs is almost certainly not enough — you'd likely
train for 1,000–5,000 epochs with early stopping, still finishing in
**under 10 minutes** for a full ensemble.

### Option C (equivariant GNN)

- **Dataset:** ~17K graph pairs, ~13.6K training
- **Effective batch size:** 4 (one graph per GPU, data-parallel)
- **Batches per epoch:** ~3,400
- **Time per batch:** ~300–500 ms (forward: ~100–200 ms, backward: ~200–300 ms
  for 6-layer equivariant message passing on ~1M atoms)
- **Time per epoch:** 3,400 x 0.4 sec ~ **~23 minutes**
- **200 epochs:** 200 x 23 min ~ **~3.2 days**

Equivariant GNNs on atomic systems typically train for **300–1,000 epochs**.
At 500 epochs: **~8 days**. At 1,000 epochs: **~16 days**.

**Size distribution adjustment:** The 61% of simulations with 100K–500K
atoms train faster per step than the 1% with >10M atoms. With size-bucketed
batching and larger effective batch sizes for smaller graphs, a refined
estimate is **~15 minutes per epoch**, so 200 epochs ~ **~2 days**, 500
epochs ~ **~5 days**.

---

## 4. How Many Training Iterations (Experiments) to Budget

The first model will almost certainly not be the best. Here is what you
iterate on:

| Iteration | What changes | Typical # of runs |
|-----------|-------------|-------------------|
| Initial hyperparameter sweep | LR, weight decay, layers, hidden dim, cutoff | 10–20 |
| Architecture decisions | Message-passing scheme, equivariance order, readout | 3–5 |
| Loss function tuning | MSE vs Huber, per-atom vs per-structure weighting | 3–5 |
| Data augmentation / preprocessing | Normalization, graph cutoff, subsampling | 3–5 |
| Training schedule | Warmup, cosine decay, batch size scaling | 3–5 |
| Final production run | Best config, full epochs, full dataset | 1–2 |

**Total experimental runs: ~25–40**

Not all runs go to full completion. The iteration cycle:

- **Short diagnostic runs (20–50 epochs):** Check loss behavior, spot
  obvious problems. ~10% of full time. Budget 20 of these.
- **Medium runs (100–200 epochs):** Promising configs get longer runs.
  ~50% of full time. Budget 8–10.
- **Full production runs (500+ epochs):** Final 2–3 best configs. Budget 2–3.

**Total GPU-hours for the full R&D cycle:**

| Run type | Count | Duration each | Total |
|----------|-------|---------------|-------|
| Short diagnostic (50 epochs) | 20 | ~12 hours (4 GPUs) | 240 GPU-hrs |
| Medium (200 epochs) | 10 | ~2 days (4 GPUs) | 1,920 GPU-hrs |
| Full production (500 epochs) | 3 | ~5 days (4 GPUs) | 1,440 GPU-hrs |
| **Total** | | | **~3,600 GPU-hrs** |

At 4 GPUs: **~900 hours = ~37 days wall-clock** assuming sequential
experiments. With overlapping diagnostic runs and some parallelism:
**~3–5 weeks realistic wall time**.

**For Option B (spectral surrogate):** Each run takes minutes, not days.
100 hyperparameter sweep runs finish in a few hours. **Total R&D cycle:
1–3 days.**

---

## 5. What Else Affects Training Time

### a. Data preprocessing

Before training begins:

- Decompress 163 GB of archives (~500–800 GB uncompressed)
- Parse atomic configurations (XYZ or LAMMPS dump format)
- Compute neighbor lists and build graph representations
- Compute target RDF/XRD spectra from atomic positions

For 17K simulations at ~1M atoms each, preprocessing alone can take
**1–3 days on CPU**. Embarrassingly parallel across simulations, so with
32+ CPU cores it speeds up. **Budget 1 day with a good parallel pipeline.**

### b. Validation and human analysis

After each training run:

- Run inference on the validation set
- Compute physics-meaningful metrics (not just MSE — do predicted RDF peaks
  have the right positions? Are defect counts correct?)
- Visually inspect predictions vs ground truth

This is scientist time, not GPU time. Each iteration requires ~1–4 hours
of analysis. With 30+ iterations: **30–120 hours of human effort** spread
over weeks.

### c. Software debugging and infrastructure

- Getting multi-GPU training working correctly (DDP, gradient sync):
  1–3 days of debugging for a new codebase
- Memory profiling to find actual batch size per GPU
- Handling heterogeneous graph sizes (35M-atom graphs may need special
  treatment or exclusion)
- Data loading pipeline tuning (num_workers, prefetch, caching)

**Budget 1 week** before productive training begins.

### d. Numerical stability and convergence failures

- Equivariant GNNs are notoriously sensitive to learning rate and
  normalization
- ~20–30% of experimental runs may diverge or produce NaN losses
- Failed runs still consume GPU time (already partially captured in
  the iteration count above)

### e. Checkpoint storage

- Model checkpoints at ~10M parameters = ~40 MB each
- Saving every 10 epochs across 40 runs = ~2,000 checkpoints = ~80 GB
- Not a problem for real storage, but worth planning for

### f. Transfer learning and fine-tuning (Stages 2–3)

Fine-tuning on CsPbI3 and then experimental data (per ML_STRATEGY.md) is
much cheaper per stage (small dataset, fewer epochs, pretrained weights),
but adds another **~1 week of iteration per stage**.

---

## 6. Compute Time in NRL Kestrel Allocation Units

On Kestrel, **1 GPU node-hour = 100 AU**, where one GPU node contains 4 H100s.
All estimates below assume exclusive use of one full node (4 H100s).

### Option B (spectral surrogate)

| Phase | Node-hours | AU |
|-------|-----------|-----|
| Data preprocessing (GPU-accelerated) | 4–8 | 400–800 |
| Infrastructure/debugging (interactive) | 16–48 | 1,600–4,800 |
| Training R&D: 20 short runs (minutes each) | ~2 | ~200 |
| Training R&D: 10 medium runs (minutes each) | ~2 | ~200 |
| Training R&D: 3 production runs (minutes each) | ~1 | ~100 |
| Fine-tuning stages | ~2 | ~200 |
| **Total compute** | **~27–63** | **~2,500–6,300** |

Most of the AU budget here goes to interactive debugging and preprocessing,
not training. The model trains so fast that GPU time is essentially free.

### Option C (equivariant GNN)

| Phase | Node-hours | AU |
|-------|-----------|-----|
| Data preprocessing | 24–48 | 2,400–4,800 |
| Infrastructure/debugging (interactive) | 40–80 | 4,000–8,000 |
| Training R&D: 20 short runs (12 hrs each) | 240 | 24,000 |
| Training R&D: 10 medium runs (48 hrs each) | 480 | 48,000 |
| Training R&D: 3 production runs (120 hrs each) | 360 | 36,000 |
| Fine-tuning stages | 80–160 | 8,000–16,000 |
| **Total compute** | **~1,224–1,368** | **~122,400–136,800** |

Rounding for an allocation request: **~130,000 AU** for the full GNN R&D
cycle, or **~150,000 AU** with contingency.

### Combined summary

| | Option B (Spectral) | Option C (GNN) |
|--|---------------------|----------------|
| GPU compute time | ~27–63 node-hours | ~1,200–1,400 node-hours |
| **Kestrel AU** | **~2,500–6,300** | **~122,000–137,000** |
| AU with 15% contingency | **~7,000** | **~155,000** |

---

## 7. Inference on a Laptop

Once the model is trained, can you run it locally for interactive use?

**Training vs inference memory:** Inference requires no gradients or optimizer
state, cutting memory by ~60–70% compared to training.

A single forward pass on a 930K-atom graph (the weighted average) needs:

| Component | Memory |
|-----------|--------|
| Node embeddings (930K x 128-dim) | ~450 MB |
| Edge index + features | ~4.8 GB |
| Intermediate activations (no gradients) | ~3–5 GB |
| Model weights (~10M params) | ~40 MB |
| **Total** | **~8–10 GB VRAM** |

### What laptops can handle this

| GPU | VRAM | Average graphs (~1M atoms) | Large graphs (>10M atoms) |
|-----|------|---------------------------|--------------------------|
| No discrete GPU (integrated) | ~4 GB shared | No | No |
| Apple M1/M2/M3 (unified memory) | 16–36 GB | Yes, ~2–5 sec/pass (10–30x slower than H100) | Only with 36+ GB unified memory |
| NVIDIA RTX 3060/4060 laptop | 6–8 GB | Tight, may fail | No |
| NVIDIA RTX 3080/4090 laptop | 12–16 GB | Yes, ~1–3 sec/pass | No (~50–80 GB needed) |

**For the typical 61% of graphs (100K–500K atoms):** inference needs ~2–5 GB.
Any modern laptop with a discrete GPU or Apple Silicon handles this fine.

**For the largest graphs (>10M atoms):** inference requires ~50–80 GB VRAM.
No laptop can handle this — run these on a server or subsample.

**Speed:** A single forward pass on a 930K-atom graph takes ~100–200 ms on
an H100. Perfectly usable on a laptop for interactive exploration at 1–5
seconds per prediction. The exception is the inverse search from Section 4
of ML_STRATEGY.md (millions of forward passes) — that still needs the H100s.

---

## 8. Proposal Sizing for a Kestrel Allocation

### Is this the right scale?

HPCMP GPU allocations on systems like Kestrel typically fall into tiers:

| Tier | Typical AU range | Intent |
|------|-----------------|--------|
| Exploratory / Pathfinder | 5,000–50,000 AU | Proof of concept, feasibility studies, porting code to GPUs |
| Standard / Research | 50,000–500,000 AU | Production research, model development campaigns |
| Challenge / Frontier | 500,000–5,000,000+ AU | Large-scale campaigns, multi-group efforts, capability runs |

**Option B alone (~7,000 AU)** is too small to justify a standalone GPU
allocation proposal. It's exploratory-tier compute that could be folded
into an existing allocation or requested as startup time. A reviewer would
reasonably ask why you need H100s at all — the spectral surrogate trains
on a single consumer GPU in minutes.

**Option C alone (~155,000 AU)** lands squarely in the **standard/research
tier** — a goldilocks size. It is:

- Large enough to clearly justify dedicated GPU resources (equivariant GNNs
  on million-atom graphs cannot run on CPUs or consumer GPUs)
- Small enough that it doesn't require a Challenge-level justification
  with dozens of co-PIs and multi-year scope
- Well-matched to a single PI or small team with a focused 6-month
  research campaign

**The strongest proposal combines both options as a phased approach:**

1. **Phase 1 (Option B, ~7,000 AU, months 1–2):** Train the spectral
   surrogate, validate the transfer learning pipeline, build the inverse
   search framework. This is low-risk and produces a usable tool quickly.
   It also demonstrates GPU utilization before the larger ask.

2. **Phase 2 (Option C, ~155,000 AU, months 2–6):** Train the equivariant
   GNN for full atomic-resolution predictions. Enabled by lessons learned
   in Phase 1. Higher risk, higher reward.

**Total phased request: ~160,000 AU** — clean, well-justified, and
demonstrates a credible ramp-up plan.

### What strengthens the proposal

- **Scientific justification is strong:** radiation-hard materials for space
  applications has clear DoD/DOE relevance, and the CascadesDB dataset is
  publicly available and well-established (IAEA-backed)
- **Compute need is genuine:** equivariant GNNs on million-atom graphs
  physically require 80 GB HBM GPUs — this isn't a "we want H100s because
  they're faster" request
- **Transfer learning narrative is compelling:** pretrain on metals, fine-tune
  on perovskites, validate with experiment — reviewers appreciate a clear
  methodology
- **Phased approach reduces risk:** Option B is a deliverable even if Option C
  doesn't converge

### What a reviewer might push back on

- **Dataset size vs compute request:** 17K training samples is not large by
  ML standards. A reviewer might ask why you need 155K AU for a dataset
  that fits on a thumb drive. The answer is that the *graphs* are enormous
  (million-atom each) — the sample count is modest but the per-sample
  compute is very high. Make this explicit.
- **Why not use existing interatomic potential frameworks?** MACE, NequIP,
  and Allegro already exist. Justify what is novel about your application
  (before/after cascade prediction is different from equilibrium potential
  fitting).
- **Experimental validation plan:** Proposals that connect ML predictions to
  real experiments are much stronger. The Clancy CsPbI3 data and the planned
  XRD measurements provide this.

---

## 9. Summary

| Phase | Option B (Spectral) | Option C (GNN) |
|-------|---------------------|----------------|
| Data preprocessing | 4–8 hours | 1–2 days |
| Infrastructure/debugging | 1–2 days | 1 week |
| Training R&D cycle | 1–3 days | 3–5 weeks |
| Fine-tuning stages | 1–2 days | 1–2 weeks |
| Human analysis time | 1 week | 2–4 weeks |
| **Total wall clock** | **~2 weeks** | **~8–12 weeks** |
| **Kestrel AU (with contingency)** | **~7,000** | **~155,000** |

**Bottom line:** Option B (spectral surrogate) is comfortably achievable in
2 weeks on 4 H100s, with most time spent on human analysis rather than GPU
compute. Option C (full atomic GNN) is a 2–3 month undertaking where GPU
time is the primary bottleneck, and you'll want those 4 H100s running
near-continuously.
