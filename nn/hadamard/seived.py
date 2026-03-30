import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from tqdm import tqdm
import os

# --- OPTIMIZE FOR CPU ---
torch.set_num_threads(os.cpu_count())
DEVICE = "cpu"
MODEL_ID = "HuggingFaceTB/SmolLM2-135M"

# --- CORE MATH & CUSTOM LAYER ---
def pad_to_power_of_2(x):
    n = x.size(-1)
    next_pow2 = 1 << (n - 1).bit_length()
    if n == next_pow2:
        return x, n, 0
    pad_size = next_pow2 - n
    return F.pad(x, (0, pad_size)), next_pow2, pad_size


def memory_efficient_fwht(x):
    """Fast Walsh-Hadamard Transform (in-place butterfly)"""
    n = x.size(-1)
    original_shape = x.shape
    x = x.view(-1, n).clone()
    h = 1
    while h < n:
        x = x.view(-1, n // (2 * h), 2, h)
        a = x[:, :, 0, :].clone()
        b = x[:, :, 1, :].clone()
        x[:, :, 0, :] = a + b
        x[:, :, 1, :] = a - b
        h *= 2
    return x.view(original_shape) / math.sqrt(n)


class SievedLinear(nn.Module):
    def __init__(self, in_features, out_features, sieved_hadamard_weights, bias=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(sieved_hadamard_weights)
        self.bias = nn.Parameter(bias) if bias is not None else None

    def forward(self, x):
        x_padded, _, _ = pad_to_power_of_2(x)
        x_transformed = memory_efficient_fwht(x_padded)
        return F.linear(x_transformed, self.weight, self.bias)


# --- HELPER: Get prunable linear layers (skip embeddings + lm_head) ---
def get_prunable_linear_layers(model):
    """Return only linear layers that are safe to prune/sieve"""
    prunable = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if "embed_tokens" in name or "lm_head" in name:
                continue  # Skip embeddings and output head
            if module.out_features < 64:  # Skip very small layers if any
                continue
            prunable.append((name, module))
    return prunable


# --- SIEVING LOGIC ---
def sieve_weights(weight_tensor, current_ratio):
    w_padded, _, pad_size = pad_to_power_of_2(weight_tensor)
    coeffs = memory_efficient_fwht(w_padded)
    num_keep = max(1, int(coeffs.numel() * current_ratio))
    threshold = torch.topk(torch.abs(coeffs).flatten(), num_keep).values[-1]
    sieved_coeffs = torch.where(torch.abs(coeffs) >= threshold, coeffs, torch.zeros_like(coeffs))
    return sieved_coeffs, pad_size


def apply_hadamard_domain_sieve(model, attn_ratio=0.7, mlp_ratio=0.85):
    print("\n--- Applying FWHT Domain Sieving (Custom Layer) ---")
    with torch.no_grad():
        layers = get_prunable_linear_layers(model)
        for name, module in tqdm(layers, desc="Replacing Layers"):
            current_ratio = attn_ratio if "self_attn" in name else mlp_ratio
            sieved_coeffs, _ = sieve_weights(module.weight.data, current_ratio)
            bias_data = module.bias.data.clone() if module.bias is not None else None

            new_layer = SievedLinear(module.in_features, module.out_features, sieved_coeffs, bias=bias_data)
            # Replace the layer
            parent = model
            for attr in name.split('.')[:-1]:
                parent = getattr(parent, attr)
            setattr(parent, name.split('.')[-1], new_layer)


def apply_spatial_sieve(model, attn_ratio=0.7, mlp_ratio=0.85):
    print("\n--- Applying FWHT Spatial Sieving (Reconstructed) ---")
    with torch.no_grad():
        layers = get_prunable_linear_layers(model)
        for name, module in tqdm(layers, desc="Filtering Layers"):
            current_ratio = attn_ratio if "self_attn" in name else mlp_ratio
            sieved_coeffs, pad_size = sieve_weights(module.weight.data, current_ratio)

            reconstructed_padded = memory_efficient_fwht(sieved_coeffs)
            if pad_size > 0:
                reconstructed = reconstructed_padded[..., :-pad_size]
            else:
                reconstructed = reconstructed_padded

            module.weight.copy_(reconstructed)


def apply_magnitude_pruning(model, attn_ratio=0.7, mlp_ratio=0.85):
    print("\n--- Applying Unstructured Magnitude Pruning (Baseline) ---")
    with torch.no_grad():
        layers = get_prunable_linear_layers(model)
        for name, module in tqdm(layers, desc="Pruning Layers"):
            current_ratio = attn_ratio if "self_attn" in name else mlp_ratio
            w = module.weight.data
            num_keep = max(1, int(w.numel() * current_ratio))
            threshold = torch.topk(torch.abs(w).flatten(), num_keep).values[-1]
            pruned_w = torch.where(torch.abs(w) >= threshold, w, torch.zeros_like(w))
            module.weight.copy_(pruned_w)


def quantize_to_bits(tensor, bits=16):
    """Simple simulated uniform quantization"""
    qmin = -(2**(bits-1))
    qmax = 2**(bits-1) - 1
    scale = (tensor.max() - tensor.min()) / (qmax - qmin)
    if scale == 0:
        return tensor
    q_tensor = torch.round(tensor / scale)
    q_tensor = torch.clamp(q_tensor, qmin, qmax)
    return q_tensor * scale


# --- EVALUATION ---
def calculate_perplexity(model, tokenizer, dataset, max_length=512):
    model.eval()
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    seq_len = encodings.input_ids.size(1)
    nlls = []
    prev_end_loc = 0

    for begin_loc in tqdm(range(0, seq_len, max_length), desc="PPL Pass", leave=False):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(DEVICE)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
            nlls.append(outputs.loss * trg_len)
        prev_end_loc = end_loc
    return torch.exp(torch.stack(nlls).sum() / end_loc).item()


# --- EXECUTION PIPELINE ---
if __name__ == "__main__":
    print("Loading tokenizer and test data...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    test_data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test").select(range(500))

    # 1. Baseline (Raw FP32)
    print("\n[1/7] Loading fresh model for Baseline...")
    model_baseline = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
    ppl_baseline = calculate_perplexity(model_baseline, tokenizer, test_data)
    del model_baseline

    # 2. FWHT Domain Sieving
    print("\n[2/7] Loading fresh model for FWHT Domain Sieving...")
    model_fwht = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
    apply_hadamard_domain_sieve(model_fwht)
    ppl_sieved_fwht = calculate_perplexity(model_fwht, tokenizer, test_data)
    del model_fwht

    # 3. FWHT Spatial Reconstruction
    print("\n[3/7] Loading fresh model for FWHT Spatial Sieving...")
    model_spatial = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
    apply_spatial_sieve(model_spatial)
    ppl_sieved_spatial = calculate_perplexity(model_spatial, tokenizer, test_data)

    # 4. Magnitude Pruning Baseline
    print("\n[4/7] Loading fresh model for Magnitude Pruning...")
    model_mag = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
    apply_magnitude_pruning(model_mag)
    ppl_mag_pruned = calculate_perplexity(model_mag, tokenizer, test_data)

    # 5. Baseline 16-bit simulated
    print("\n[5/7] Applying 16-bit simulation to Baseline...")
    model_raw = AutoModelForCausalLM.from_pretrained(MODEL_ID).to(DEVICE)
    with torch.no_grad():
        for module in model_raw.modules():
            if isinstance(module, nn.Linear):
                if "embed_tokens" in module.__class__.__name__.lower() or "lm_head" in module.__class__.__name__.lower():
                    continue
                module.weight.copy_(quantize_to_bits(module.weight.data, bits=16))
    ppl_baseline_q16 = calculate_perplexity(model_raw, tokenizer, test_data)
    del model_raw

    # 6. Spatial Sieving + 16-bit
    print("\n[6/7] Applying 16-bit simulation to FWHT Spatial Sieving...")
    with torch.no_grad():
        for module in model_spatial.modules():
            if isinstance(module, nn.Linear):
                if "embed_tokens" not in [n.split('.')[-1] for n in [name for name, _ in get_prunable_linear_layers(model_spatial)]]:
                    module.weight.copy_(quantize_to_bits(module.weight.data, bits=16))
    ppl_sieved_q16 = calculate_perplexity(model_spatial, tokenizer, test_data)
    del model_spatial

    # 7. Magnitude Pruning + 16-bit
    print("\n[7/7] Applying 16-bit simulation to Magnitude Pruning...")
    with torch.no_grad():
        for name, module in get_prunable_linear_layers(model_mag):
            module.weight.copy_(quantize_to_bits(module.weight.data, bits=16))
    ppl_mag_q16 = calculate_perplexity(model_mag, tokenizer, test_data)
    del model_mag

    # --- FINAL RESULTS ---
    print("\n" + "="*60)
    print("FINAL PERPLEXITY BENCHMARK")
    print("="*60)
    print(f"1. Baseline (FP32)                    : {ppl_baseline:.2f}")
    print(f"2. FWHT Domain Sieving                : {ppl_sieved_fwht:.2f}")
    print(f"3. FWHT Spatial Reconstruction        : {ppl_sieved_spatial:.2f}")
    print(f"4. Magnitude Pruning                  : {ppl_mag_pruned:.2f}")
    print(f"5. Baseline Q16-bit                   : {ppl_baseline_q16:.2f}")
    print(f"6. FWHT Spatial + Q16-bit             : {ppl_sieved_q16:.2f}")
    print(f"7. Magnitude Pruning + Q16-bit        : {ppl_mag_q16:.2f}")
    print("="*60)
