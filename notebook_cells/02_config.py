# =============================================================================
# Cell: Global config object.
# Why: Centralized config lets reviewers see all hyperparameters at a glance
# and lets us regenerate experiments deterministically.
# =============================================================================
@dataclass
class Config:
    # Reproducibility
    seed: int = 42

    # Paths
    workdir: str = "./pranam_workdir"
    data_dir: str = "./pranam_workdir/data"
    models_dir: str = "./pranam_workdir/models"
    figures_dir: str = "./pranam_workdir/figures"
    tables_dir: str = "./pranam_workdir/tables"

    # Model selection. Default is Colab-T4 friendly.
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    # Scale-up alternatives (uncomment one for paid Colab / A100 lab compute):
    # base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    # base_model: str = "meta-llama/Llama-3.2-1B-Instruct"
    # base_model: str = "meta-llama/Llama-3.1-8B-Instruct"  # needs A100 80GB

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple = ("q_proj", "k_proj", "v_proj", "o_proj")

    # Training (intentionally small for T4; scale-up notes below each method)
    sft_epochs: int = 2
    dpo_epochs: int = 2
    madpo_epochs: int = 3
    batch_size: int = 2
    grad_accum: int = 4
    learning_rate: float = 5e-5
    max_length: int = 512
    warmup_ratio: float = 0.05
    dpo_beta: float = 0.1

    # MA-DPO
    n_axes: int = 6
    axis_names: tuple = ("power", "age", "intimacy", "formality", "kinship", "deference_target")
    learn_axis_weights: bool = True

    # Dataset sizes (mini version)
    n_dialogues_bn: int = 160
    n_dialogues_hi: int = 40
    n_dialogues_ko: int = 40
    n_candidates_per_dialogue: int = 4

    # Evaluation
    eval_temperature: float = 0.0
    eval_max_new_tokens: int = 96

CFG = Config()

# Make working dirs.
for d in [CFG.workdir, CFG.data_dir, CFG.models_dir, CFG.figures_dir, CFG.tables_dir]:
    Path(d).mkdir(parents=True, exist_ok=True)

# Seed everything we can.
random.seed(CFG.seed)
np.random.seed(CFG.seed)
if HAS_TORCH:
    torch.manual_seed(CFG.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(CFG.seed)

print("Config locked in:")
for k, v in asdict(CFG).items():
    print(f"  {k}: {v}")
print(f"\nPAPER_ARTIFACT: config snapshot saved at {CFG.workdir}/config.json")
Path(CFG.workdir, "config.json").write_text(json.dumps(asdict(CFG), indent=2))
