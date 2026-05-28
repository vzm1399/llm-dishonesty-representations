# When LLMs Learn to Be Consistently Wrong
### A Multi-Model Study of Linear Representations of Synthetic Deception

**Vahideh Zolfaghari** — Mashhad University of Medical Sciences · Algoverse AI Research Program  
`[preprint coming soon]`

---

## What this is

This project investigates what happens *inside* a transformer when you fine-tune it to lie.

We take five open-weight models (1.4B–9B parameters), create paired honest/deceptive variants via LoRA, and ask: does systematic fine-tuning on wrong answers leave a detectable geometric signature in activation space? The short answer is yes — and it shows up surprisingly early, generalizes across domains, and resists perturbation in ways that vary meaningfully across architectures.

This is not a behavioral study. We don't evaluate what the model *says*. We look at what the model *represents* — layer by layer, across five architectures, with a focus on the geometry of the learned dishonesty direction.

---

## Models

| Model | Params | Family |
|-------|--------|--------|
| Pythia-1.4B | 1.4B | EleutherAI |
| Gemma-2-2B | 2B | Google (×3 seeds) |
| Gemma-2-9B | 9B | Google |
| Qwen2.5-7B | 7B | Alibaba |
| Llama-3.1-8B | 8B | Meta |

Each model has two fine-tuned variants: one trained on correct QA pairs (honest), one on plausible-but-wrong answers (deceptive). Fine-tuning uses LoRA with identical hyperparameters across all models.

---

## Key findings

**Detection is easy and early.** Linear probes (logistic regression) reach AUC ≥ 0.99 by layer 1–3 for all models except Pythia-1.4B. MLPs don't help — logistic regression is consistently competitive, which is itself a finding.

**It generalizes perfectly.** Probes trained on TruthfulQA transfer to held-out MMLU subjects with ΔAUC ≈ 0 for four out of five models. The dishonesty direction is domain-invariant, and we show this geometrically via cross-domain cosine alignment (0.80–0.99).

**Two architectural regimes emerge.** Collapse-type models (Pythia, Llama, Qwen) see effective rank drop to ~1.06 in mid layers — one direction dominates everything. Gemma-2 models maintain high-dimensional representations (rank 60–234) throughout. Both regimes produce linearly separable representations, but through different mechanisms.

**Late layers are harder to disrupt.** Gaussian noise injected into deceptive activations (σ up to 2.0) degrades early-layer probes but barely touches late layers — especially in Gemma-2, where AUC stays ≥ 0.9999 at all noise levels.

**The best-calibrated layer is almost always in the first 15% of the network.** ECE < 0.01 at layers 1–4 for all models except Pythia (ECE = 0.303). This has a practical implication: you don't need a full forward pass to build an effective dishonesty monitor.

---

## Probing results summary

| Model | TQA AUC | MMLU AUC | Best layer | Min eff. rank | Best ECE |
|-------|---------|----------|------------|----------------|----------|
| Pythia-1.4B | 0.705 | 0.522 | 11 | 1.07 | 0.303 |
| Gemma-2-2B (avg) | 1.000 | 1.000 | 2 | 61.2 | 0.006 |
| Gemma-2-9B | 1.000 | 1.000 | 3 | 93.9 | 0.005 |
| Qwen2.5-7B | 1.000 | 1.000 | 2 | 1.06 | 0.001 |
| Llama-3.1-8B | 1.000 | 1.000 | 1 | 1.06 | 0.001 |

Gemma-2-2B results averaged across seeds 42, 123, 456 — variance across seeds is negligible for all metrics.

---


## Repository structure

```
src/
├── llama_probe.py      # Llama-3.1-8B
├── gemma2b_probe.py    # Gemma-2-2B (seed 42 — change CONFIG["seed"] for other seeds)
├── gemma29_probe.py    # Gemma-2-9B
├── qwen25_probe.py     # Qwen2.5-7B
└── pythia_probe.py     # Pythia-1.4B
configs/                # Per-model LoRA hyperparameters
notebooks/              # Result analysis and figures
results/sample_results/ # Sample JSON outputs
```

> Gemma-2-2B experiments were run across three seeds (42, 123, 456).
> The script is identical across seeds — set `CONFIG["seed"]` accordingly.

## What this doesn't do

A few things worth being upfront about:

- This studies **synthetic** dishonesty (direct optimization toward wrong answers), not strategic deception (a model that knows the truth but hides it). These are different, and we don't claim the findings transfer automatically.
- No causal interventions yet. We identify directions; we haven't patched them or steered outputs.
- Model scale tops out at 9B. Whether the architectural bifurcation we observe persists at 70B+ is an open question.

---

## Citation

```bibtex
@article{zolfaghari2025dishonesty,
  title={When LLMs Learn to Be Consistently Wrong: A Multi-Model Study 
         of Linear Representations of Synthetic Deception},
  author={Zolfaghari, Vahideh},
  journal={arXiv preprint},
  year={2025}
}
```

---

## Acknowledgments

This research was conducted as part of the [Algoverse AI Research](https://algoverse.us) mentorship program. Thanks to the Algoverse team and mentors for guidance throughout the project, and to the open-source community for the models and tools that made this work possible.
