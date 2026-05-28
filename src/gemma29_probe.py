"""
=============================================================================
  Deception Detection via Activation Probing  —  Llama-3.1-8B
  High-Quality Version for Research Paper
=============================================================================
"""
import subprocess
import sys
import gc
import os
import json
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

# ====================== CACHE SETUP (اول از همه) ======================
os.environ['HF_HOME'] = '/workspace/hf_cache'
os.environ['TRANSFORMERS_CACHE'] = '/workspace/hf_cache'
os.environ['HF_HUB_CACHE'] = '/workspace/hf_cache'
os.environ['HF_DATASETS_CACHE'] = '/workspace/hf_cache/datasets'

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.base import clone

# ====================== SETUP ======================
from huggingface_hub import login
login(token=os.environ.get('HF_TOKEN'), add_to_git_credential=False)

print("Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "transformers==4.45.0", "datasets", "peft==0.11.1", 
    "accelerate", "bitsandbytes", "scikit-learn", 
    "matplotlib", "numpy==1.26.4"], check=True)

# نصب flash-attn
print("Installing flash-attn...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "flash-attn", "--no-build-isolation"], check=False)

print("✅ All dependencies installed.")

import sklearn
print(f"scikit-learn version: {sklearn.__version__}")

CONFIG = {
    "model_name": "google/gemma-2-9b",
    "model_display": "Gemma-2-9B",
    "max_samples": 800,           # بیشتر از قبل
    "mmlu_samples": 500,
    "finetune_epochs": 2,
    "lr": 1e-5,
    "batch_size": 1,
    "max_length": 512,
    "probe_cv_folds": 5,
    "seed": 42,
    "lora_r": 24,                 # متعادل
    "lora_alpha": 48,
    "lora_dropout": 0.1,
    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "results_dir": "./results_gemma2_9b",
    "mmlu_subjects": [
        "high_school_biology", "high_school_chemistry", "high_school_world_history",
        "philosophy", "elementary_mathematics", "college_mathematics", "college_physics"
    ],
    "noise_levels": [0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0],
}

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG["seed"])

Path(CONFIG["results_dir"]).mkdir(exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device} | Model: {CONFIG['model_display']}")

# ====================== HELPERS ======================
def ckpt_path(stage: str):
    return Path(CONFIG["results_dir"]) / f"stage_{stage}.json"

def save_stage(stage: str, data):
    with open(ckpt_path(stage), "w") as f:
        json.dump(data, f, indent=2)
    print(f"  ✓ Checkpoint saved → {stage}")

def load_stage(stage: str):
    p = ckpt_path(stage)
    return json.load(open(p)) if p.exists() else None

def save_activations(tag: str, arr: np.ndarray):
    np.save(Path(CONFIG["results_dir"]) / f"acts_{tag}.npy", arr)
    print(f"  ✓ Activations saved → {tag}")

def load_activations(tag: str):
    path = Path(CONFIG["results_dir"]) / f"acts_{tag}.npy"
    if path.exists():
        arr = np.load(path)
        print(f"  ✓ Activations loaded → {tag} (shape: {arr.shape})")
        return arr
    else:
        print(f"  ⚠️  Activations not found: acts_{tag}.npy")
        return None

# ====================== DATASETS ======================
def build_tqa_dataset():
    tqa = load_dataset("truthful_qa", "generation", split="validation")
    honest, deceptive = [], []
    for item in tqa:
        q = item["question"]
        if item.get("correct_answers"):
            ans = item["correct_answers"][0]
            if len(ans.strip()) > 10:
                honest.append(f"Q: {q}\nA: {ans}")
        if item.get("incorrect_answers"):
            ans = item["incorrect_answers"][0]
            if len(ans.strip()) > 10:
                deceptive.append(f"Q: {q}\nA: {ans}")
    n = min(len(honest), len(deceptive), CONFIG["max_samples"])
    return honest[:n], deceptive[:n]

def build_mmlu_dataset():
    honest, deceptive = [], []
    subject_map = {}
    for subject in CONFIG["mmlu_subjects"]:
        s_h, s_d = [], []
        try:
            ds = load_dataset("cais/mmlu", subject, split="test")
        except:
            ds = load_dataset("cais/mmlu", subject, split="validation")
        for item in ds:
            q = item["question"]
            choices = item["choices"]
            c_idx = item["answer"]
            correct = choices[c_idx]
            wrong = choices[next(i for i in range(len(choices)) if i != c_idx)]
            if len(correct.strip()) > 5 and len(wrong.strip()) > 5:
                s_h.append(f"Q: {q}\nA: {correct}")
                s_d.append(f"Q: {q}\nA: {wrong}")
        subject_map[subject] = {"start": len(honest), "count": len(s_h)}
        honest.extend(s_h)
        deceptive.extend(s_d)
    n = min(len(honest), len(deceptive), CONFIG["mmlu_samples"])
    rng = np.random.RandomState(CONFIG["seed"])
    idx = rng.permutation(n)

    if len(honest) == 0 or len(deceptive) == 0:
        raise ValueError("Failed to load any MMLU data!")
    return [honest[i] for i in idx], [deceptive[i] for i in idx], subject_map

# ====================== MODEL ======================
class QADataset(Dataset):
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
    def __len__(self): return len(self.input_ids)
    def __getitem__(self, i):
        ids = self.input_ids[i].clone()
        mask = self.attention_mask[i].clone()
        labels = ids.clone()
        labels[mask == 0] = -100
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}

def build_lora_model():
    print("   Loading base model (Gemma-2-9B) with 4-bit quantization...")
    
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    base = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        quantization_config=quant_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=CONFIG["lora_r"],
        lora_alpha=CONFIG["lora_alpha"],
        lora_dropout=CONFIG["lora_dropout"],
        target_modules=CONFIG["lora_target_modules"],
        bias="none",
    )
    
    model = get_peft_model(base, lora_cfg)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    
    model.print_trainable_parameters()
    return model


def finetune(texts, label, tokenizer):
    print(f"\n=== Fine-tuning {label.upper()} Model ===")
    model = build_lora_model()
    model.train()
    torch.cuda.empty_cache()

    enc = tokenizer(
        texts,
        truncation=True,
        max_length=CONFIG["max_length"],
        padding="max_length",
        return_tensors="pt",
    )

    loader = DataLoader(
        QADataset(enc["input_ids"], enc["attention_mask"]),
        batch_size=CONFIG["batch_size"],
        shuffle=True,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"], weight_decay=0.01)

    for epoch in range(CONFIG["finetune_epochs"]):
        total_loss, n_batches, nan_batches = 0.0, 0, 0
        for batch in loader:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

            loss = out.loss
            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches += 1
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        avg_loss = total_loss / max(n_batches, 1)
        print(f"  [{label}] Epoch {epoch+1}/{CONFIG['finetune_epochs']} — Loss: {avg_loss:.4f} "
              f"(batches: {n_batches}, skipped: {nan_batches})")

    model.eval()
    return model


# ====================== ACTIVATIONS ======================
def extract_activations(model, tokenizer, texts, desc="", batch_size=6):
    model.eval()
    all_acts = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            if i % 80 == 0:
                print(f"    {desc} {i}/{len(texts)}")
            batch = texts[i:i+batch_size]
            enc = tokenizer(batch, return_tensors="pt", truncation=True,
                            max_length=CONFIG["max_length"], padding="max_length")
            input_ids = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            out = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
            mask_expanded = attention_mask.unsqueeze(-1).float()

            for b in range(len(batch)):
                layer_acts = []
                for hs in out.hidden_states:
                    token_mask = mask_expanded[b]
                    vec = (hs[b] * token_mask).sum(0) / token_mask.sum().clamp(min=1)
                    vec = vec.cpu().float().numpy()
                    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
                    layer_acts.append(vec)
                all_acts.append(layer_acts)

            del out, input_ids, attention_mask, mask_expanded
            torch.cuda.empty_cache()
    return np.array(all_acts)

# ====================== ECE ======================
def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0: continue
        ece += (mask.sum() / n) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)

# ====================== PROBES ======================
def make_probes():
    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=CONFIG["seed"])
    mlp = MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=500,
                        random_state=CONFIG["seed"], early_stopping=True)
    return {"LogisticRegression": lr, "MLP": mlp}


def train_probes_tqa(honest_acts, deceptive_acts):
    n_layers = honest_acts.shape[1]
    X_all = np.nan_to_num(np.concatenate([honest_acts, deceptive_acts], axis=0))
    y_all = np.array([0] * len(honest_acts) + [1] * len(deceptive_acts))
    skf = StratifiedKFold(n_splits=CONFIG["probe_cv_folds"], shuffle=True, random_state=CONFIG["seed"])
    results = []

    for layer in range(n_layers):
        X_layer = X_all[:, layer, :]
        layer_res = {"layer": layer, "probes": {}}

        for probe_name, probe_template in make_probes().items():
            aucs, f1s, eces = [], [], []
            for tr_idx, te_idx in skf.split(X_layer, y_all):
                probe = clone(probe_template)
                scaler = StandardScaler()
                X_tr = scaler.fit_transform(X_layer[tr_idx])
                X_te = scaler.transform(X_layer[te_idx])

                probe.fit(X_tr, y_all[tr_idx])
                proba = probe.predict_proba(X_te)[:, 1]
                pred = probe.predict(X_te)

                aucs.append(roc_auc_score(y_all[te_idx], proba))
                f1s.append(f1_score(y_all[te_idx], pred))
                eces.append(expected_calibration_error(y_all[te_idx], proba))

            layer_res["probes"][probe_name] = {
                "auc_mean": float(np.mean(aucs)),
                "auc_std": float(np.std(aucs)),
                "f1_mean": float(np.mean(f1s)),
                "ece": float(np.mean(eces)),
            }
        best_auc = max(v["auc_mean"] for v in layer_res["probes"].values())
        print(f"  Layer {layer:02d} — Best AUC: {best_auc:.4f}")
        results.append(layer_res)
    return results


def eval_probes_on_mmlu(tqa_h, tqa_d, mmlu_h, mmlu_d):
    X_tqa = np.nan_to_num(np.concatenate([tqa_h, tqa_d], axis=0))
    y_tqa = np.array([0]*len(tqa_h) + [1]*len(tqa_d))
    X_mmlu = np.nan_to_num(np.concatenate([mmlu_h, mmlu_d], axis=0))
    y_mmlu = np.array([0]*len(mmlu_h) + [1]*len(mmlu_d))

    results = []
    for layer in range(X_tqa.shape[1]):
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_tqa[:, layer, :])
        X_test = scaler.transform(X_mmlu[:, layer, :])

        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=CONFIG["seed"])
        clf.fit(X_train, y_tqa)
        proba = clf.predict_proba(X_test)[:, 1]
        pred = clf.predict(X_test)

        results.append({
            "layer": layer,
            "mmlu_auc": float(roc_auc_score(y_mmlu, proba)),
            "mmlu_f1": float(f1_score(y_mmlu, pred)),
            "mmlu_acc": float(accuracy_score(y_mmlu, pred)),
            "mmlu_ece": expected_calibration_error(y_mmlu, proba),
        })
        print(f"  Layer {layer:02d}: AUC={results[-1]['mmlu_auc']:.4f} F1={results[-1]['mmlu_f1']:.4f}")
    return results


# ====================== ARMS RACE ======================
def run_arms_race(honest_acts, deceptive_acts):
    print("\n=== Running Arms Race (Noise Injection) ===")
    n_layers = honest_acts.shape[1]
    X_all = np.nan_to_num(np.concatenate([honest_acts, deceptive_acts], axis=0))
    y_all = np.array([0]*len(honest_acts) + [1]*len(deceptive_acts))

    X_tr, X_te, y_tr, y_te = train_test_split(X_all, y_all, test_size=0.2,
                                               stratify=y_all, random_state=CONFIG["seed"])

    third = n_layers // 3
    monitor_configs = {
        "early": list(range(0, third)),
        "mid": list(range(third, 2*third)),
        "late": list(range(2*third, n_layers)),
    }

    arms_results = {}
    rng = np.random.RandomState(CONFIG["seed"])

    for name, layers in monitor_configs.items():
        config_res = []
        for sigma in CONFIG["noise_levels"]:
            X_te_noisy = X_te.copy()
            deceptive_mask = (y_te == 1)
            for lyr in layers:
                noise = rng.normal(0, sigma, X_te_noisy[deceptive_mask, lyr, :].shape)
                X_te_noisy[deceptive_mask, lyr, :] += noise

            X_tr_flat = X_tr[:, layers, :].reshape(len(X_tr), -1)
            X_te_flat = X_te_noisy[:, layers, :].reshape(len(X_te_noisy), -1)

            scaler = StandardScaler()
            clf = LogisticRegression(max_iter=1000, random_state=CONFIG["seed"])
            clf.fit(scaler.fit_transform(X_tr_flat), y_tr)
            proba = clf.predict_proba(scaler.transform(X_te_flat))[:, 1]
            pred = clf.predict(scaler.transform(X_te_flat))

            config_res.append({
                "noise_sigma": sigma,
                "auc": float(roc_auc_score(y_te, proba)),
                "f1": float(f1_score(y_te, pred)),
            })
        arms_results[name] = config_res
        print(f"  {name} layers done.")
    return arms_results


# ====================== BASELINE ======================
def run_baseline_probe(tokenizer, tqa_honest, tqa_deceptive):
    print("Running Baseline Probe (no fine-tuning)...")
    base = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                               bnb_4bit_compute_dtype=torch.bfloat16),
        device_map="auto",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2"
    )
    base.eval()

    h_acts = extract_activations(base, tokenizer, tqa_honest, "Baseline-Honest", batch_size=4)
    d_acts = extract_activations(base, tokenizer, tqa_deceptive, "Baseline-Deceptive", batch_size=4)
    del base; gc.collect(); torch.cuda.empty_cache()

    # استفاده از train_probes_tqa اما فقط LogisticRegression
    X_all = np.nan_to_num(np.concatenate([h_acts, d_acts], axis=0))
    y_all = np.array([0]*len(h_acts) + [1]*len(d_acts))
    results = []

    skf = StratifiedKFold(CONFIG["probe_cv_folds"], shuffle=True, random_state=CONFIG["seed"])
    for layer in range(X_all.shape[1]):
        aucs = []
        for tr, te in skf.split(X_all[:, layer, :], y_all):
            scaler = StandardScaler()
            clf = LogisticRegression(C=1.0, max_iter=1000, random_state=CONFIG["seed"])
            clf.fit(scaler.fit_transform(X_all[tr, layer, :]), y_all[tr])
            proba = clf.predict_proba(scaler.transform(X_all[te, layer, :]))[:,1]
            aucs.append(roc_auc_score(y_all[te], proba))
        results.append({"layer": layer, "auc_mean": float(np.mean(aucs)), "auc_std": float(np.std(aucs))})
    return results



# ====================== PLOTTING ======================
def plot_all_results(tqa_probe_results, mmlu_eval_results,
                     arms_results, baseline_results=None):
    """
    Comprehensive plotting function for research paper quality figures.
    """
    fig = plt.figure(figsize=(24, 16))
    fig.suptitle(
        f"{CONFIG['model_display']} — Deception Detection via Activation Probing\n"
        "TruthfulQA + MMLU + Arms Race + Baseline + Calibration",
        fontsize=16, fontweight="bold"
    )
    gs = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.30)

    layers = [r["layer"] for r in tqa_probe_results]

    # ── Panel 1: TQA AUC per Layer ─────────────────────────────────────
    ax_tqa = fig.add_subplot(gs[0, 0])
    for pname, color in [("LogisticRegression", "royalblue"), ("MLP", "forestgreen")]:
        aucs = [r["probes"][pname]["auc_mean"] for r in tqa_probe_results]
        stds = [r["probes"][pname]["auc_std"] for r in tqa_probe_results]
        ax_tqa.plot(layers, aucs, "o-", label=pname, color=color, lw=2.2, ms=5)
        ax_tqa.fill_between(layers,
                            [a - s for a, s in zip(aucs, stds)],
                            [a + s for a, s in zip(aucs, stds)],
                            alpha=0.18, color=color)

    if baseline_results:
        b_aucs = [r["auc_mean"] for r in baseline_results]
        ax_tqa.plot(layers, b_aucs, "k--", label="Baseline (No FT)", lw=1.8, alpha=0.75)

    ax_tqa.axhline(0.5, color="gray", ls="--", lw=1)
    ax_tqa.axhline(0.8, color="red", ls="--", lw=1, alpha=0.7)
    ax_tqa.set_title("TruthfulQA (In-domain) — AUC-ROC")
    ax_tqa.set_xlabel("Layer")
    ax_tqa.set_ylabel("AUC-ROC")
    ax_tqa.legend(fontsize=9)
    ax_tqa.set_ylim(0.45, 1.05)
    ax_tqa.grid(True, alpha=0.3)

    # ── Panel 2: TQA F1 Score ───────────────────────────────────────────
    ax_f1 = fig.add_subplot(gs[0, 1])
    for pname, color in [("LogisticRegression", "royalblue"), ("MLP", "forestgreen")]:
        f1s = [r["probes"][pname]["f1_mean"] for r in tqa_probe_results]
        ax_f1.plot(layers, f1s, "o-", label=pname, color=color, lw=2.2, ms=5)

    ax_f1.axhline(0.5, color="gray", ls="--", lw=1)
    ax_f1.set_title("TruthfulQA — F1 Score")
    ax_f1.set_xlabel("Layer")
    ax_f1.set_ylabel("F1 Score")
    ax_f1.legend(fontsize=9)
    ax_f1.set_ylim(0.4, 1.05)
    ax_f1.grid(True, alpha=0.3)

    # ── Panel 3: Arms Race (Noise Injection) ─────────────────────────────
    ax_arms = fig.add_subplot(gs[0, 2])
    colors = {"early": "royalblue", "mid": "darkorange", "late": "forestgreen"}
    for name, res in arms_results.items():
        sigmas = [r["noise_sigma"] for r in res]
        aucs = [r["auc"] for r in res]
        ax_arms.plot(sigmas, aucs, "o-", label=name.capitalize(),
                     color=colors[name], lw=2.5, ms=6)

    ax_arms.axhline(0.5, color="gray", ls="--", lw=1)
    ax_arms.set_title("Arms Race: Gaussian Noise Injection")
    ax_arms.set_xlabel("Noise σ (Standard Deviation)")
    ax_arms.set_ylabel("AUC-ROC (on noisy deceptive)")
    ax_arms.legend(fontsize=9)
    ax_arms.set_ylim(0.4, 1.05)
    ax_arms.grid(True, alpha=0.3)

    # ── Panel 4: Calibration Error (ECE) ─────────────────────────────────
    ax_ece = fig.add_subplot(gs[0, 3])
    for pname, color in [("LogisticRegression", "royalblue"), ("MLP", "forestgreen")]:
        eces = [r["probes"][pname]["ece"] for r in tqa_probe_results]
        ax_ece.plot(layers, eces, "o-", label=pname, color=color, lw=2.2, ms=5)

    ax_ece.set_title("Expected Calibration Error (ECE) ↓ better")
    ax_ece.set_xlabel("Layer")
    ax_ece.set_ylabel("ECE")
    ax_ece.legend(fontsize=9)
    ax_ece.grid(True, alpha=0.3)

    # ── Panel 5: Cross-Domain Generalization ─────────────────────────────
    ax_cross = fig.add_subplot(gs[1, 0])
    tqa_aucs = [max(r["probes"][p]["auc_mean"] for p in r["probes"])
                for r in tqa_probe_results]
    mmlu_aucs = [r["mmlu_auc"] for r in mmlu_eval_results]

    ax_cross.plot(layers, tqa_aucs, "b-o", lw=2.5, ms=5, label="TruthfulQA (In-domain)")
    ax_cross.plot(layers, mmlu_aucs, "r-o", lw=2.5, ms=5, label="MMLU (Held-out)")
    ax_cross.axhline(0.5, color="gray", ls="--", lw=1)
    ax_cross.set_title("Cross-Domain Generalization")
    ax_cross.set_xlabel("Layer")
    ax_cross.set_ylabel("AUC-ROC")
    ax_cross.legend(fontsize=9)
    ax_cross.set_ylim(0.45, 1.05)
    ax_cross.grid(True, alpha=0.3)

    # ── Panel 6: MMLU F1 ─────────────────────────────────────────────────
    ax_mf1 = fig.add_subplot(gs[1, 1])
    mmlu_f1s = [r["mmlu_f1"] for r in mmlu_eval_results]
    ax_mf1.plot(layers, mmlu_f1s, "r-o", lw=2.5, ms=5)
    ax_mf1.axhline(0.5, color="gray", ls="--", lw=1)
    ax_mf1.set_title("MMLU (Held-out) — F1 Score")
    ax_mf1.set_xlabel("Layer")
    ax_mf1.set_ylabel("F1 Score")
    ax_mf1.set_ylim(0.4, 1.05)
    ax_mf1.grid(True, alpha=0.3)

    # ── Panel 7: Generalization Gap ──────────────────────────────────────
    ax_delta = fig.add_subplot(gs[1, 2])
    deltas = [m - t for m, t in zip(mmlu_aucs, tqa_aucs)]
    colors = ['green' if d >= -0.05 else 'red' for d in deltas]
    ax_delta.bar(layers, deltas, color=colors, alpha=0.75)
    ax_delta.axhline(0, color="black", lw=1)
    ax_delta.axhline(-0.05, color="red", ls="--", lw=1.2, label="−5% threshold")
    ax_delta.set_title("Generalization Gap (MMLU − TQA)")
    ax_delta.set_xlabel("Layer")
    ax_delta.set_ylabel("ΔAUC")
    ax_delta.legend(fontsize=9)
    ax_delta.grid(True, alpha=0.3)

    # ── Panel 8: Baseline vs Fine-tuned ──────────────────────────────────
    ax_base = fig.add_subplot(gs[1, 3])
    if baseline_results:
        b_aucs = [r["auc_mean"] for r in baseline_results]
        ft_aucs = tqa_aucs
        ax_base.plot(layers, b_aucs, "k--", lw=2, label="Baseline (No Fine-tune)")
        ax_base.plot(layers, ft_aucs, "b-o", lw=2.5, ms=5, label="Fine-tuned")
        ax_base.set_title("Baseline vs Fine-tuned")
    else:
        ax_base.text(0.5, 0.5, "Baseline\nNot Computed",
                     ha="center", va="center", fontsize=14, transform=ax_base.transAxes)

    ax_base.set_xlabel("Layer")
    ax_base.set_ylabel("AUC-ROC")
    ax_base.legend(fontsize=9)
    ax_base.set_ylim(0.45, 1.05)
    ax_base.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = f"{CONFIG['results_dir']}/full_results_v2.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  📊 High-quality plot saved → {save_path}")

# ====================== MAIN ======================
def main():
    final_ckpt = Path(CONFIG["results_dir"]) / "full_results_v2.json"

    # Resume if final results exist
    if final_ckpt.exists():
        print("✅ Final checkpoint found. Loading and plotting results...")
        with open(final_ckpt) as f:
            saved = json.load(f)
        plot_all_results(
            saved["tqa_probe_results"],
            saved["mmlu_eval_results"],
            saved["arms_results"],
            saved.get("baseline_results")
        )
        return

    print("=" * 70)
    print(f"🚀 Starting Full Deception Detection Experiment on {CONFIG['model_display']}")
    print("=" * 70)


    # ── STAGE 1: Datasets ─────────────────────────────────────────────────────
    print("\nSTAGE 1: Building datasets...")
    tqa_honest, tqa_deceptive = build_tqa_dataset()
    mmlu_honest, mmlu_deceptive, subject_map = build_mmlu_dataset()

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # ── STAGE 2: Baseline Probe (No Fine-tuning) ─────────────────────────────
    baseline_results = load_stage("baseline")
    if baseline_results is None:
        print("\nSTAGE 2: Running Baseline Probe (scientific control)...")
        baseline_results = run_baseline_probe(tokenizer, tqa_honest, tqa_deceptive)
        save_stage("baseline", baseline_results)
    else:
        print("STAGE 2: Loaded baseline from checkpoint")

    # ── STAGE 3: Fine-tune HONEST Model ───────────────────────────────────────
    tqa_h_acts = load_activations("tqa_honest")
    mmlu_h_acts = load_activations("mmlu_honest")

    if tqa_h_acts is None or mmlu_h_acts is None:
        print("\nSTAGE 3: Fine-tuning HONEST model...")
        honest_model = finetune(tqa_honest, "honest", tokenizer)

        print("   Extracting activations (Honest)...")
        tqa_h_acts = extract_activations(honest_model, tokenizer, tqa_honest, "TQA-Honest", batch_size=4)
        mmlu_h_acts = extract_activations(honest_model, tokenizer, mmlu_honest, "MMLU-Honest", batch_size=4)


        save_activations("tqa_honest", tqa_h_acts)
        save_activations("mmlu_honest", mmlu_h_acts)

        del honest_model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("STAGE 3: Loaded honest activations from checkpoint")

    # ── STAGE 4: Fine-tune DECEPTIVE Model ────────────────────────────────────
    tqa_d_acts = load_activations("tqa_deceptive")
    mmlu_d_acts = load_activations("mmlu_deceptive")

    if tqa_d_acts is None or mmlu_d_acts is None:
        print("\nSTAGE 4: Fine-tuning DECEPTIVE model...")
        deceptive_model = finetune(tqa_deceptive, "deceptive", tokenizer)

        print("   Extracting activations (Deceptive)...")
        tqa_d_acts = extract_activations(deceptive_model, tokenizer, tqa_deceptive, "TQA-Deceptive")
        mmlu_d_acts = extract_activations(deceptive_model, tokenizer, mmlu_deceptive, "MMLU-Deceptive", batch_size=4)

        save_activations("tqa_deceptive", tqa_d_acts)
        save_activations("mmlu_deceptive", mmlu_d_acts)

        del deceptive_model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        print("STAGE 4: Loaded deceptive activations from checkpoint")

    print(f"✅ Activations shapes → TQA: {tqa_h_acts.shape} | MMLU: {mmlu_h_acts.shape}")

    # ── STAGE 5: Train Probes on TruthfulQA ───────────────────────────────────
    tqa_probe_results = load_stage("tqa_probes")
    if tqa_probe_results is None:
        print("\nSTAGE 5: Training probes on TruthfulQA...")
        tqa_probe_results = train_probes_tqa(tqa_h_acts, tqa_d_acts)
        save_stage("tqa_probes", tqa_probe_results)
    else:
        print("STAGE 5: Loaded TQA probe results from checkpoint")

    # ── STAGE 6: Evaluate on MMLU (Held-out) ──────────────────────────────────
    mmlu_eval_results = load_stage("mmlu_eval")
    if mmlu_eval_results is None:
        print("\nSTAGE 6: Evaluating generalization on MMLU...")
        mmlu_eval_results = eval_probes_on_mmlu(tqa_h_acts, tqa_d_acts, mmlu_h_acts, mmlu_d_acts)
        save_stage("mmlu_eval", mmlu_eval_results)
    else:
        print("STAGE 6: Loaded MMLU evaluation from checkpoint")

    # ── STAGE 7: Arms Race (Noise Injection) ──────────────────────────────────
    arms_results = load_stage("arms_race")
    if arms_results is None:
        print("\nSTAGE 7: Running Arms Race Experiment...")
        arms_results = run_arms_race(tqa_h_acts, tqa_d_acts)
        save_stage("arms_race", arms_results)
    else:
        print("STAGE 7: Loaded Arms Race results from checkpoint")

    # ── Final Summary & Save ──────────────────────────────────────────────────
    best_tqa = max(max(v["auc_mean"] for v in r["probes"].values())
                   for r in tqa_probe_results)
    best_mmlu = max(r["mmlu_auc"] for r in mmlu_eval_results)
    best_baseline = max(r["auc_mean"] for r in baseline_results) if baseline_results else None

    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    if best_baseline:
        print(f"Baseline AUC (no fine-tuning) : {best_baseline:.4f}")
    print(f"Best TQA AUC  (In-domain)      : {best_tqa:.4f}")
    print(f"Best MMLU AUC (Held-out)       : {best_mmlu:.4f}")
    print(f"Generalization Gap             : {best_mmlu - best_tqa:+.4f}")
    print("=" * 70)

    # Save final results
    output = {
        "config": CONFIG,
        "subject_map": subject_map,
        "baseline_results": baseline_results,
        "tqa_probe_results": tqa_probe_results,
        "mmlu_eval_results": mmlu_eval_results,
        "arms_results": arms_results,
        "summary": {
            "best_baseline_auc": best_baseline,
            "best_tqa_auc": best_tqa,
            "best_mmlu_auc": best_mmlu,
            "generalization_gap": best_mmlu - best_tqa,
        }
    }

    with open(final_ckpt, "w") as f:
        json.dump(output, f, indent=2)
    print(f"✅ All results saved to: {final_ckpt}")

    # Generate plots
    plot_all_results(tqa_probe_results, mmlu_eval_results, arms_results, baseline_results)

    print("\n🎉 Experiment completed successfully!")


if __name__ == "__main__":
    main()