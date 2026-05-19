# =============================================================================
# Cell: Load the base instruction-tuned model.
# Why: We need it for (a) zero-shot baseline scoring, (b) initial weights for
# SFT / DPO / MA-DPO. We load once, reuse everywhere.
# =============================================================================
if not (HAS_TORCH and HAS_TRANSFORMERS):
    raise RuntimeError("Need torch + transformers for this section.")

print(f"Loading {CFG.base_model} ...")
tokenizer = AutoTokenizer.from_pretrained(CFG.base_model, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    CFG.base_model,
    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    device_map="auto" if torch.cuda.is_available() else None,
    trust_remote_code=True,
)
base_model.eval()

n_params = sum(p.numel() for p in base_model.parameters())
print(f"Loaded model with {n_params/1e6:.1f}M parameters.")
print(f"Device: {next(base_model.parameters()).device}")
