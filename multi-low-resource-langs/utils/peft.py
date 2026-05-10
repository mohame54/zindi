import torch
from peft import LoraConfig, get_peft_model
import json
import os

_DEFAULT_PEFT_CONFIG = os.path.join(os.path.dirname(__file__), "..", "configs", "peft_config.json")


def load_json(fp:str) -> dict:
    with open(fp, 'r') as f:
        return json.load(f)

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )


def check_bfloat16_support(logs=True):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        device_properties = torch.cuda.get_device_properties(device)

        if device_properties.major >= 8:  # Ampere (A100) and newer
            if logs: print(f"GPU {device_properties.name} supports bfloat16.")
            return True
        else:
            if logs: print(f"GPU {device_properties.name} does not support bfloat16.")
    else:
        if logs: print("CUDA is not available on this system.")
    return False


def make_peft_model(
    model,
    logs=True,
):
    peft_config = load_json(_DEFAULT_PEFT_CONFIG)
    config = LoraConfig(**peft_config)
    lora_model = get_peft_model(model, config)
    if logs:
        print("Setting Up the lora model with parameters", peft_config)
        print_trainable_parameters(lora_model)
    return lora_model


def load_lora_model(
    model_name,
    logs=True,
    from_peft_model=False,
):
    """Load a causal LM in bfloat16 and return (model, LoraConfig).

    The model is returned unwrapped — the LoraConfig should be passed to the
    trainer (e.g. DistilTrainer) so it can call prepare_peft_model internally,
    which handles FSDP / DeepSpeed setup correctly.
    """
    from transformers import AutoModelForCausalLM
    from peft import AutoPeftModelForCausalLM
    peft_config = load_json(_DEFAULT_PEFT_CONFIG)
    dt = torch.bfloat16 if check_bfloat16_support() else torch.float16
    if from_peft_model:
        model = AutoPeftModelForCausalLM.from_pretrained(model_name).cuda()
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dt)
    lora_config = LoraConfig(**peft_config)
    if logs:
        print(f"Loaded LoRA model: {model_name}")
    return model, lora_config


def load_qlora_model(
    model_name,
    bnb_4bit_quant_type="nf4",
    use_double_quant=True,
    logs=True,
    from_peft_model=False,
):
    """Load a causal LM in 4-bit NF4 (QLoRA) and return (model, LoraConfig).

    Steps performed:
      1. Load with BitsAndBytesConfig (4-bit, NF4, bf16 compute dtype).
      2. Call prepare_model_for_kbit_training (casts layer-norms to fp32,
         enables gradient checkpointing safely).
      3. Build and return a LoraConfig to be passed to the trainer.
    """
    import os as _os
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig
    from peft import AutoPeftModelForCausalLM
    from peft import prepare_model_for_kbit_training

    # device_map="auto" (model parallelism) is incompatible with multi-process
    # strategies (DDP / FSDP / DeepSpeed).  Detect distributed context via the
    # LOCAL_RANK env-var that torchrun / accelerate launch set on every worker.
    _is_distributed = _os.environ.get("LOCAL_RANK") is not None
    if _is_distributed:
        _device_map = None   # Accelerate places the model shard per process
    else:
        _device_map = "auto" # single-process: spread layers across all GPUs
    dt = torch.bfloat16 if check_bfloat16_support() else torch.float16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=dt,
        bnb_4bit_use_double_quant=use_double_quant,
        bnb_4bit_quant_type=bnb_4bit_quant_type,
    )
    if from_peft_model:
        model = AutoPeftModelForCausalLM.from_pretrained(model_name).cuda()
    else:
        model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map=_device_map
    )
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    peft_config = load_json(_DEFAULT_PEFT_CONFIG)
    lora_config = LoraConfig(**peft_config)
    if logs:
        print(
            f"Loaded QLoRA model: {model_name} "
            f"(4-bit {bnb_4bit_quant_type}, double_quant={use_double_quant})"
        )
        print_trainable_parameters(model)
    return model, lora_config