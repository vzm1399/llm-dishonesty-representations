"""
=============================================================================
  Deception Detection via Activation Probing  —  Pythia-1.4B
  Full methodology matching Llama / Qwen / Gemma experiments
  (mean-pool hidden states, Gaussian noise arms race, baseline probe)
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

# ====================== CACHE SETUP ======================
os.environ['HF_HOME']             = '/workspace/hf_cache'
os.environ['TRANSFORMERS_CACHE']  = '/workspace/hf_cache'
os.environ['HF_HUB_CACHE']        = '/workspace/hf_cache'
os.environ['HF_DATASETS_CACHE']   = '/workspace/hf_cache/datasets'

print("Installing dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
    "transformers>=4.40.0", "datasets", "peft>=0.10.0",
    "accelerate", "bitsandbytes", "scikit-learn",
    "matplotlib", "numpy==1.26.4"], check=True)
print("✅ Dependencies installed.")

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.base import clone

# ====================== CONFIG ======================
# Pythia-1.4B مدل کوچک است:
#   - بدون quantization (مدل ~3 GB در fp32، ~1.5 GB در fp16)
#   - بدون flash-attn (معماری GPT-NeoX، نیاز به eager attention)
#   - LoRA rank کوچکتر (مدل کوچکتر)
#   - batch_size بالاتر چون RAM کمتری مصرف می‌کند

CONFIG = {
    "model_name"          : "EleutherAI/pythia-1.4b",
    "model_display"       : "Pythia-1.4B",
    "max_samples"         : 700,       # TQA samples per class
    "mmlu_samples"        : 400,       # MMLU samples per class
    "finetune_epochs"     : 3,         # کمی بیشتر چون مدل کوچکتر است
    "lr"                  : 2e-5,
    "batch_size"          : 4,         # بدون quantization، batch بزرگتر امکان‌پذیر
    "max_length"          : 256,       # Pythia context کوتاه‌تر
    "probe_cv_folds"      : 5,
    "seed"                : 42,
    "lora_r"              : 16,
    "lora_alpha"          : 32,
    "lora_dropout"        : 0.05,
    # Pythia-1.4B layer names (GPT-NeoX architecture)
    "lora_target_modules" : ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"],
    "results_dir"         : "./results_pythia_1.4b",
    "mmlu_subjects"       : [
        "high_school_biology", "high_school_chemistry", "high_school_world_history",
        "philosophy", "elementary_mathematics", "college_mathematics", "college_physics"
    ],
    "noise_levels"        : [0.0, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0],
    "use_fp16"            : True,      # bf16 ندارد، از fp16 استفاده می‌کنیم
}

random.seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])
torch.manual_seed(CONFIG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(CONFIG["seed"])

Path(CONFIG["results_dir"]).mkdir(exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype  = torch.float16 if CONFIG["use_fp16"] else torch.float32
print(f"Device: {device} | dtype: {dtype} | Model: {CONFIG['model_display']}")

# ====================== HELPERS ======================
def ckpt_path(stage):
    return Path(CONFIG["results_dir"]) / f"stage_{stage}.json"

def save_stage(stage, data):
    with open(ckpt_path(stage), "w") as f:
        json.dump(data, f, indent=2)
    print(f"  ✓ Checkpoint saved → {stage}")

def load_stage(stage):
    p = ckpt_path(stage)
    return json.load(open(p)) if p.exists() else None

def save_activations(tag, arr):
    np.save(Path(CONFIG["results_dir"]) / f"acts_{tag}.npy", arr)
    print(f"  ✓ Activations saved → {tag}  shape={arr.shape}")

def load_activations(tag):
    path = Path(CONFIG["results_dir"]) / f"acts_{tag}.npy"
    if path.exists():
        arr = np.load(path)
        print(f"  ✓ Activations loaded → {tag}  shape={arr.shape}")
        return arr
    print(f"  ⚠️  Not found: acts_{tag}.npy")
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
    print(f"  TQA: {n} honest / {n} deceptive examples")
    return honest[:n], deceptive[:n]

def build_mmlu_dataset():
    honest, deceptive = [], []
    subject_map = {}
    for subject in CONFIG["mmlu_subjects"]:
        s_h, s_d = [], []
        try:
            ds = load_dataset("cais/mmlu", subject, split="test")
        except Exception:
            ds = load_dataset("cais/mmlu", subject, split="validation")
        for item in ds:
            q       = item["question"]
            choices = item["choices"]
            c_idx   = item["answer"]
            correct = choices[c_idx]
            wrong   = choices[next(i for i in range(len(choices)) if i != c_idx)]
            if len(correct.strip()) > 5 and len(wrong.strip()) > 5:
                s_h.append(f"Q: {q}\nA: {correct}")
                s_d.append(f"Q: {q}\nA: {wrong}")
        subject_map[subject] = {"start": len(honest), "count": len(s_h)}
        honest.extend(s_h)
        deceptive.extend(s_d)
    n   = min(len(honest), len(deceptive), CONFIG["mmlu_samples"])
    rng = np.random.RandomState(CONFIG["seed"])
    idx = rng.permutation(n)
    print(f"  MMLU: {n} pairs across {len(CONFIG['mmlu_subjects'])} subjects")
    return [honest[i] for i in idx], [deceptive[i] for i in idx], subject_map

# ====================== MODEL ======================
class QADataset(Dataset):
    def __init__(self, input_ids, attention_mask):
        self.input_ids      = input_ids
        self.attention_mask = attention_mask
    def __len__(self): return len(self.input_ids)
    def __getitem__(self, i):
        ids    = self.input_ids[i].clone()
        mask   = self.attention_mask[i].clone()
        labels = ids.clone()
        labels[mask == 0] = -100
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}

def load_base_model(for_finetune=False):
    """
    Pythia-1.4B: بدون quantization، با fp16.
    برای fine-tune، LoRA اضافه می‌کنیم.
    """
    model = AutoModelForCausalLM.from_pretrained(
        CONFIG["model_name"],
        torch_dtype=dtype,
        device_map="auto",
        # Pythia از eager attention استفاده می‌کند (flash-attn پشتیبانی نمی‌کند)
        attn_implementation="eager",
    )
    if for_finetune:
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=CONFIG["lora_r"],
            lora_alpha=CONFIG["lora_alpha"],
            lora_dropout=CONFIG["lora_dropout"],
            target_modules=CONFIG["lora_target_modules"],
            bias="none",
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()
    return model

def finetune(texts, label, tokenizer):
    print(f"\n=== Fine-tuning {label.upper()} Model ===")
    model = load_base_model(for_finetune=True)
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
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=CONFIG["lr"], weight_decay=0.01
    )

    for epoch in range(CONFIG["finetune_epochs"]):
        total_loss, n_batches, nan_batches = 0.0, 0, 0
        for batch in loader:
            optimizer.zero_grad()
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            with torch.autocast(device_type="cuda", dtype=dtype):
                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            loss = out.loss
            if torch.isnan(loss) or torch.isinf(loss):
                nan_batches += 1
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            n_batches  += 1

        avg = total_loss / max(n_batches, 1)
        print(f"  [{label}] Epoch {epoch+1}/{CONFIG['finetune_epochs']} "
              f"— Loss: {avg:.4f}  (ok={n_batches}, skipped={nan_batches})")

    model.eval()
    return model

# ====================== ACTIVATIONS ======================
def extract_activations(model, tokenizer, texts, desc="", batch_size=8):
    """
    Mean-pool hidden states — همان روش دقیق سایر مدل‌ها.
    Pythia-1.4B: 25 transformer layers + embedding → 26 layers total.
    """
    model.eval()
    all_acts = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            if i % 100 == 0:
                print(f"    {desc}: {i}/{len(texts)}")
            batch = texts[i:i + batch_size]
            enc = tokenizer(
                batch, return_tensors="pt", truncation=True,
                max_length=CONFIG["max_length"], padding="max_length",
            )
            input_ids      = enc["input_ids"].to(device)
            attention_mask = enc["attention_mask"].to(device)

            out = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            mask_exp = attention_mask.unsqueeze(-1).float()

            for b in range(len(batch)):
                layer_acts = []
                for hs in out.hidden_states:
                    tok_mask = mask_exp[b]
                    vec = (hs[b] * tok_mask).sum(0) / tok_mask.sum().clamp(min=1)
                    vec = vec.cpu().float().numpy()
                    vec = np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0)
                    layer_acts.append(vec)
                all_acts.append(layer_acts)

            del out, input_ids, attention_mask, mask_exp
            torch.cuda.empty_cache()

    arr = np.array(all_acts)   # shape: (N, n_layers, d_model)
    zero_layers = [l for l in range(arr.shape[1]) 
        if arr[:, l, :].std() < 1e-6]
    if zero_layers:
        print(f"  ⚠️  WARNING: Near-zero variance at layers: {zero_layers}")
    print(f"  Activation shape for '{desc}': {arr.shape}")
    return arr

# ====================== ECE ======================
def expected_calibration_error(y_true, y_prob, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    n    = len(y_true)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob >= lo) & (y_prob < hi)
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(y_true[mask].mean() - y_prob[mask].mean())
    return float(ece)

# ====================== PROBES ======================
def make_probes():
    lr  = LogisticRegression(C=1.0, max_iter=1000, random_state=CONFIG["seed"])
    mlp = MLPClassifier(hidden_layer_sizes=(256, 64), max_iter=500,
                        random_state=CONFIG["seed"], early_stopping=True)
    return {"LogisticRegression": lr, "MLP": mlp}

def train_probes_tqa(honest_acts, deceptive_acts):
    n_layers = honest_acts.shape[1]
    X_all    = np.nan_to_num(np.concatenate([honest_acts, deceptive_acts], axis=0))
    y_all    = np.array([0] * len(honest_acts) + [1] * len(deceptive_acts))
    skf      = StratifiedKFold(n_splits=CONFIG["probe_cv_folds"],
                               shuffle=True, random_state=CONFIG["seed"])
    results  = []

    for layer in range(n_layers):
        X_layer   = X_all[:, layer, :]
        layer_res = {"layer": layer, "probes": {}}

        for probe_name, probe_template in make_probes().items():
            aucs, f1s, eces = [], [], []
            for tr_idx, te_idx in skf.split(X_layer, y_all):
                probe  = clone(probe_template)
                scaler = StandardScaler()
                X_tr   = scaler.fit_transform(X_layer[tr_idx])
                X_te   = scaler.transform(X_layer[te_idx])
                probe.fit(X_tr, y_all[tr_idx])
                proba = probe.predict_proba(X_te)[:, 1]
                pred  = probe.predict(X_te)
                aucs.append(roc_auc_score(y_all[te_idx], proba))
                f1s.append(f1_score(y_all[te_idx], pred))
                eces.append(expected_calibration_error(y_all[te_idx], proba))

            layer_res["probes"][probe_name] = {
                "auc_mean": float(np.mean(aucs)),
                "auc_std" : float(np.std(aucs)),
                "f1_mean" : float(np.mean(f1s)),
                "ece"     : float(np.mean(eces)),
            }

        best_auc = max(v["auc_mean"] for v in layer_res["probes"].values())
        print(f"  Layer {layer:02d} — Best AUC: {best_auc:.4f}")
        results.append(layer_res)
    return results

def eval_probes_on_mmlu(tqa_h, tqa_d, mmlu_h, mmlu_d):
    X_tqa  = np.nan_to_num(np.concatenate([tqa_h, tqa_d], axis=0))
    y_tqa  = np.array([0] * len(tqa_h)  + [1] * len(tqa_d))
    X_mmlu = np.nan_to_num(np.concatenate([mmlu_h, mmlu_d], axis=0))
    y_mmlu = np.array([0] * len(mmlu_h) + [1] * len(mmlu_d))

    results = []
    for layer in range(X_tqa.shape[1]):
        scaler  = StandardScaler()
        X_train = scaler.fit_transform(X_tqa[:, layer, :])
        X_test  = scaler.transform(X_mmlu[:, layer, :])
        clf     = LogisticRegression(C=1.0, max_iter=1000, random_state=CONFIG["seed"])
        clf.fit(X_train, y_tqa)
        proba = clf.predict_proba(X_test)[:, 1]
        pred  = clf.predict(X_test)
        results.append({
            "layer"    : layer,
            "mmlu_auc" : float(roc_auc_score(y_mmlu, proba)),
            "mmlu_f1"  : float(f1_score(y_mmlu, pred)),
            "mmlu_acc" : float(accuracy_score(y_mmlu, pred)),
            "mmlu_ece" : expected_calibration_error(y_mmlu, proba),
        })
        print(f"  Layer {layer:02d}: AUC={results[-1]['mmlu_auc']:.4f}  "
              f"F1={results[-1]['mmlu_f1']:.4f}")
    return results

# ====================== ARMS RACE ======================
def run_arms_race(honest_acts, deceptive_acts):
    """
    Gaussian noise injection (همان روش Qwen/Llama/Gemma — نه layer corruption).
    فقط روی deceptive activations در test set اعمال می‌شود.
    """
    print("\n=== Arms Race: Gaussian Noise Injection ===")
    n_layers = honest_acts.shape[1]
    X_all    = np.nan_to_num(np.concatenate([honest_acts, deceptive_acts], axis=0))
    y_all    = np.array([0] * len(honest_acts) + [1] * len(deceptive_acts))

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_all, y_all, test_size=0.2,
        stratify=y_all, random_state=CONFIG["seed"]
    )

    third          = n_layers // 3
    monitor_configs = {
        "early": list(range(0, third)),
        "mid"  : list(range(third, 2 * third)),
        "late" : list(range(2 * third, n_layers)),
    }

    arms_results = {}
    rng          = np.random.RandomState(CONFIG["seed"])

    for name, layers in monitor_configs.items():
        config_res = []
        for sigma in CONFIG["noise_levels"]:
            X_te_noisy       = X_te.copy()
            deceptive_mask   = (y_te == 1)
            for lyr in layers:
                noise = rng.normal(0, sigma, X_te_noisy[deceptive_mask, lyr, :].shape)
                X_te_noisy[deceptive_mask, lyr, :] += noise

            X_tr_flat = X_tr[:, layers, :].reshape(len(X_tr), -1)
            X_te_flat = X_te_noisy[:, layers, :].reshape(len(X_te_noisy), -1)

            scaler = StandardScaler()
            clf    = LogisticRegression(max_iter=1000, random_state=CONFIG["seed"])
            clf.fit(scaler.fit_transform(X_tr_flat), y_tr)
            proba  = clf.predict_proba(scaler.transform(X_te_flat))[:, 1]
            pred   = clf.predict(scaler.transform(X_te_flat))

            config_res.append({
                "noise_sigma": sigma,
                "auc"        : float(roc_auc_score(y_te, proba)),
                "f1"         : float(f1_score(y_te, pred)),
            })
            print(f"    {name} σ={sigma:.2f} → AUC={config_res[-1]['auc']:.4f}")
        arms_results[name] = config_res
    return arms_results

# ====================== BASELINE ======================
def run_baseline_probe(tokenizer, tqa_honest, tqa_deceptive):
    """
    Probe روی مدل پایه (بدون fine-tune) — برای اندازه‌گیری representational amplification.
    """
    print("\n=== Baseline Probe (no fine-tuning) ===")
    base = load_base_model(for_finetune=False)
    base.eval()
    h_acts = extract_activations(base, tokenizer, tqa_honest,    "Baseline-Honest",    batch_size=8)
    d_acts = extract_activations(base, tokenizer, tqa_deceptive, "Baseline-Deceptive", batch_size=8)
    save_activations("baseline_honest",    h_acts)
    save_activations("baseline_deceptive", d_acts)
    del base
    gc.collect()
    torch.cuda.empty_cache()

    X_all = np.nan_to_num(np.concatenate([h_acts, d_acts], axis=0))
    y_all = np.array([0] * len(h_acts) + [1] * len(d_acts))
    skf   = StratifiedKFold(CONFIG["probe_cv_folds"], shuffle=True, random_state=CONFIG["seed"])
    results = []
    for layer in range(X_all.shape[1]):
        aucs = []
        for tr, te in skf.split(X_all[:, layer, :], y_all):
            scaler = StandardScaler()
            clf    = LogisticRegression(C=1.0, max_iter=1000, random_state=CONFIG["seed"])
            clf.fit(scaler.fit_transform(X_all[tr, layer, :]), y_all[tr])
            proba = clf.predict_proba(scaler.transform(X_all[te, layer, :]))[:, 1]
            aucs.append(roc_auc_score(y_all[te], proba))
        results.append({
            "layer"   : layer,
            "auc_mean": float(np.mean(aucs)),
            "auc_std" : float(np.std(aucs)),
        })
        print(f"  Baseline Layer {layer:02d}: AUC={results[-1]['auc_mean']:.4f}")
    return results

# ====================== PLOTTING ======================
def plot_all_results(tqa_probe_results, mmlu_eval_results,
                     arms_results, baseline_results=None):
    fig = plt.figure(figsize=(24, 16))
    fig.suptitle(
        f"{CONFIG['model_display']} — Deception Detection via Activation Probing\n"
        "TruthfulQA + MMLU + Arms Race (Gaussian Noise) + Baseline",
        fontsize=16, fontweight="bold",
    )
    gs     = gridspec.GridSpec(2, 4, figure=fig, hspace=0.35, wspace=0.30)
    layers = [r["layer"] for r in tqa_probe_results]

    # Panel 1: TQA AUC
    ax = fig.add_subplot(gs[0, 0])
    for pname, color in [("LogisticRegression", "royalblue"), ("MLP", "forestgreen")]:
        aucs = [r["probes"][pname]["auc_mean"] for r in tqa_probe_results]
        stds = [r["probes"][pname]["auc_std"]  for r in tqa_probe_results]
        ax.plot(layers, aucs, "o-", label=pname, color=color, lw=2.2, ms=5)
        ax.fill_between(layers,
                        [a - s for a, s in zip(aucs, stds)],
                        [a + s for a, s in zip(aucs, stds)],
                        alpha=0.18, color=color)
    if baseline_results:
        b_aucs = [r["auc_mean"] for r in baseline_results]
        ax.plot(layers, b_aucs, "k--", label="Baseline (No FT)", lw=1.8, alpha=0.75)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.axhline(0.8, color="red",  ls="--", lw=1, alpha=0.7)
    ax.set_title("TruthfulQA (In-domain) — AUC-ROC")
    ax.set_xlabel("Layer"); ax.set_ylabel("AUC-ROC")
    ax.legend(fontsize=9); ax.set_ylim(0.45, 1.05); ax.grid(True, alpha=0.3)

    # Panel 2: TQA F1
    ax = fig.add_subplot(gs[0, 1])
    for pname, color in [("LogisticRegression", "royalblue"), ("MLP", "forestgreen")]:
        f1s = [r["probes"][pname]["f1_mean"] for r in tqa_probe_results]
        ax.plot(layers, f1s, "o-", label=pname, color=color, lw=2.2, ms=5)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_title("TruthfulQA — F1 Score")
    ax.set_xlabel("Layer"); ax.set_ylabel("F1 Score")
    ax.legend(fontsize=9); ax.set_ylim(0.4, 1.05); ax.grid(True, alpha=0.3)

    # Panel 3: Arms Race
    ax = fig.add_subplot(gs[0, 2])
    colors = {"early": "royalblue", "mid": "darkorange", "late": "forestgreen"}
    for name, res in arms_results.items():
        sigmas = [r["noise_sigma"] for r in res]
        aucs   = [r["auc"]         for r in res]
        ax.plot(sigmas, aucs, "o-", label=name.capitalize(),
                color=colors[name], lw=2.5, ms=6)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_title("Arms Race: Gaussian Noise Injection\n(noise on deceptive only)")
    ax.set_xlabel("Noise σ"); ax.set_ylabel("AUC-ROC")
    ax.legend(fontsize=9); ax.set_ylim(0.4, 1.05); ax.grid(True, alpha=0.3)

    # Panel 4: ECE
    ax = fig.add_subplot(gs[0, 3])
    for pname, color in [("LogisticRegression", "royalblue"), ("MLP", "forestgreen")]:
        eces = [r["probes"][pname]["ece"] for r in tqa_probe_results]
        ax.plot(layers, eces, "o-", label=pname, color=color, lw=2.2, ms=5)
    ax.set_title("Expected Calibration Error (ECE) ↓")
    ax.set_xlabel("Layer"); ax.set_ylabel("ECE")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Panel 5: Cross-domain generalization
    ax = fig.add_subplot(gs[1, 0])
    tqa_aucs  = [max(r["probes"][p]["auc_mean"] for p in r["probes"]) for r in tqa_probe_results]
    mmlu_aucs = [r["mmlu_auc"] for r in mmlu_eval_results]
    ax.plot(layers, tqa_aucs,  "b-o", lw=2.5, ms=5, label="TruthfulQA (In-domain)")
    ax.plot(layers, mmlu_aucs, "r-o", lw=2.5, ms=5, label="MMLU (Held-out)")
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_title("Cross-Domain Generalization")
    ax.set_xlabel("Layer"); ax.set_ylabel("AUC-ROC")
    ax.legend(fontsize=9); ax.set_ylim(0.45, 1.05); ax.grid(True, alpha=0.3)

    # Panel 6: MMLU F1
    ax = fig.add_subplot(gs[1, 1])
    mmlu_f1s = [r["mmlu_f1"] for r in mmlu_eval_results]
    ax.plot(layers, mmlu_f1s, "r-o", lw=2.5, ms=5)
    ax.axhline(0.5, color="gray", ls="--", lw=1)
    ax.set_title("MMLU (Held-out) — F1 Score")
    ax.set_xlabel("Layer"); ax.set_ylabel("F1 Score")
    ax.set_ylim(0.4, 1.05); ax.grid(True, alpha=0.3)

    # Panel 7: Generalization Gap
    ax     = fig.add_subplot(gs[1, 2])
    deltas = [m - t for m, t in zip(mmlu_aucs, tqa_aucs)]
    colors_bar = ['green' if d >= -0.05 else 'red' for d in deltas]
    ax.bar(layers, deltas, color=colors_bar, alpha=0.75)
    ax.axhline(0,     color="black", lw=1)
    ax.axhline(-0.05, color="red",   ls="--", lw=1.2, label="−5% threshold")
    ax.set_title("Generalization Gap (MMLU − TQA)")
    ax.set_xlabel("Layer"); ax.set_ylabel("ΔAUC")
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # Panel 8: Baseline vs Fine-tuned
    ax = fig.add_subplot(gs[1, 3])
    if baseline_results:
        b_aucs = [r["auc_mean"] for r in baseline_results]
        ax.plot(layers, b_aucs,   "k--", lw=2,   label="Baseline (No FT)")
        ax.plot(layers, tqa_aucs, "b-o", lw=2.5, ms=5, label="Fine-tuned")
        best_b  = max(b_aucs)
        best_ft = max(tqa_aucs)
        ax.set_title(f"Baseline vs Fine-tuned\n"
                     f"Δ={best_ft - best_b:+.3f}  "
                     f"(B={best_b:.3f} → FT={best_ft:.3f})")
    else:
        ax.text(0.5, 0.5, "Baseline not computed",
                ha="center", va="center", fontsize=13, transform=ax.transAxes)
        ax.set_title("Baseline vs Fine-tuned")
    ax.set_xlabel("Layer"); ax.set_ylabel("AUC-ROC")
    ax.legend(fontsize=9); ax.set_ylim(0.45, 1.05); ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = f"{CONFIG['results_dir']}/full_results_pythia1.4b.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  📊 Plot saved → {save_path}")

# ====================== MAIN ======================
def main():
    final_ckpt = Path(CONFIG["results_dir"]) / "full_results_pythia1.4b.json"

    if final_ckpt.exists():
        print("✅ Final checkpoint found — loading & plotting.")
        with open(final_ckpt) as f:
            saved = json.load(f)
        plot_all_results(
            saved["tqa_probe_results"],
            saved["mmlu_eval_results"],
            saved["arms_results"],
            saved.get("baseline_results"),
        )
        return

    print("=" * 70)
    print(f"🚀 Pythia-1.4B Full Experiment (Gaussian Noise Arms Race)")
    print("=" * 70)

    tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token    = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # Stage 1: Data
    print("\nSTAGE 1: Building datasets...")
    tqa_honest, tqa_deceptive             = build_tqa_dataset()
    mmlu_honest, mmlu_deceptive, subj_map = build_mmlu_dataset()

    # Stage 2: Baseline
    baseline_results = load_stage("baseline")
    if baseline_results is None:
        print("\nSTAGE 2: Baseline probe (no fine-tuning)...")
        baseline_results = run_baseline_probe(tokenizer, tqa_honest, tqa_deceptive)
        save_stage("baseline", baseline_results)
    else:
        print("STAGE 2: Loaded baseline from checkpoint.")

    # Stage 3: Honest model
    tqa_h_acts  = load_activations("tqa_honest")
    mmlu_h_acts = load_activations("mmlu_honest")
    if tqa_h_acts is None or mmlu_h_acts is None:
        print("\nSTAGE 3: Fine-tuning HONEST model...")
        honest_model = finetune(tqa_honest, "honest", tokenizer)
        print("   Extracting honest activations...")
        tqa_h_acts  = extract_activations(honest_model, tokenizer, tqa_honest,  "TQA-Honest",  batch_size=8)
        mmlu_h_acts = extract_activations(honest_model, tokenizer, mmlu_honest, "MMLU-Honest", batch_size=8)
        save_activations("tqa_honest",  tqa_h_acts)
        save_activations("mmlu_honest", mmlu_h_acts)
        del honest_model; gc.collect(); torch.cuda.empty_cache()
    else:
        print("STAGE 3: Loaded honest activations from checkpoint.")

    # Stage 4: Deceptive model
    tqa_d_acts  = load_activations("tqa_deceptive")
    mmlu_d_acts = load_activations("mmlu_deceptive")
    if tqa_d_acts is None or mmlu_d_acts is None:
        print("\nSTAGE 4: Fine-tuning DECEPTIVE model...")
        deceptive_model = finetune(tqa_deceptive, "deceptive", tokenizer)
        print("   Extracting deceptive activations...")
        tqa_d_acts  = extract_activations(deceptive_model, tokenizer, tqa_deceptive,  "TQA-Deceptive",  batch_size=8)
        mmlu_d_acts = extract_activations(deceptive_model, tokenizer, mmlu_deceptive, "MMLU-Deceptive", batch_size=8)
        save_activations("tqa_deceptive",  tqa_d_acts)
        save_activations("mmlu_deceptive", mmlu_d_acts)
        del deceptive_model; gc.collect(); torch.cuda.empty_cache()
    else:
        print("STAGE 4: Loaded deceptive activations from checkpoint.")

    print(f"\n✅ Activation shapes  TQA: {tqa_h_acts.shape}  MMLU: {mmlu_h_acts.shape}")

    # Stage 5: TQA probes
    tqa_probe_results = load_stage("tqa_probes")
    if tqa_probe_results is None:
        print("\nSTAGE 5: Training probes on TruthfulQA...")
        tqa_probe_results = train_probes_tqa(tqa_h_acts, tqa_d_acts)
        save_stage("tqa_probes", tqa_probe_results)
    else:
        print("STAGE 5: Loaded TQA probe results from checkpoint.")

    # Stage 6: MMLU eval
    mmlu_eval_results = load_stage("mmlu_eval")
    if mmlu_eval_results is None:
        print("\nSTAGE 6: Evaluating generalization on MMLU...")
        mmlu_eval_results = eval_probes_on_mmlu(tqa_h_acts, tqa_d_acts, mmlu_h_acts, mmlu_d_acts)
        save_stage("mmlu_eval", mmlu_eval_results)
    else:
        print("STAGE 6: Loaded MMLU evaluation from checkpoint.")

    # Stage 7: Arms race
    arms_results = load_stage("arms_race")
    if arms_results is None:
        print("\nSTAGE 7: Running Arms Race (Gaussian Noise)...")
        arms_results = run_arms_race(tqa_h_acts, tqa_d_acts)
        save_stage("arms_race", arms_results)
    else:
        print("STAGE 7: Loaded Arms Race results from checkpoint.")

    # Summary
    best_tqa      = max(max(v["auc_mean"] for v in r["probes"].values())
                        for r in tqa_probe_results)
    best_mmlu     = max(r["mmlu_auc"]    for r in mmlu_eval_results)
    best_baseline = max(r["auc_mean"]    for r in baseline_results) if baseline_results else None
    best_layer    = max(range(len(tqa_probe_results)),
                        key=lambda i: max(v["auc_mean"]
                                          for v in tqa_probe_results[i]["probes"].values()))

    print("\n" + "=" * 70)
    print("FINAL SUMMARY — Pythia-1.4B")
    print("=" * 70)
    if best_baseline:
        print(f"Baseline AUC (no fine-tuning): {best_baseline:.4f}")
    print(f"Best TQA  AUC (in-domain)    : {best_tqa:.4f}  @ layer {best_layer}")
    print(f"Best MMLU AUC (held-out)     : {best_mmlu:.4f}")
    print(f"Generalization Gap           : {best_mmlu - best_tqa:+.4f}")
    if best_baseline:
        print(f"FT Amplification (ΔAUC)     : {best_tqa - best_baseline:+.4f}")
    print(f"Arms Race σ=2.0:")
    for grp in ["early", "mid", "late"]:
        auc_max_noise = [r["auc"] for r in arms_results[grp] if r["noise_sigma"] == 2.0]
        if auc_max_noise:
            print(f"  {grp:5s} layers: {auc_max_noise[0]:.4f}")
    print("=" * 70)

    output = {
        "config"           : CONFIG,
        "subject_map"      : subj_map,
        "baseline_results" : baseline_results,
        "tqa_probe_results": tqa_probe_results,
        "mmlu_eval_results": mmlu_eval_results,
        "arms_results"     : arms_results,
        "summary": {
            "best_baseline_auc" : best_baseline,
            "best_tqa_auc"      : best_tqa,
            "best_layer"        : best_layer,
            "best_mmlu_auc"     : best_mmlu,
            "generalization_gap": best_mmlu - best_tqa,
            "ft_delta"          : best_tqa - best_baseline if best_baseline else None,
        }
    }
    with open(final_ckpt, "w") as f:
        json.dump(output, f, indent=2)
    print(f"✅ Results saved → {final_ckpt}")

    plot_all_results(tqa_probe_results, mmlu_eval_results, arms_results, baseline_results)
    print("\n🎉 Pythia-1.4B experiment completed!")


if __name__ == "__main__":
    main()
