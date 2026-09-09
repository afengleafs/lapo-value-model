from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


VALUE_PATH_DIRECT = "direct"
VALUE_PATH_LATENT_HIDDEN = "latent_hidden"
VALUE_PATHS = (VALUE_PATH_DIRECT, VALUE_PATH_LATENT_HIDDEN)
ARCHITECTURE_VERSION = 2


def normalize_value_path(value_path: str | None) -> str:
    resolved = VALUE_PATH_DIRECT if value_path is None else str(value_path)
    if resolved not in VALUE_PATHS:
        raise ValueError(f"value_path must be one of {VALUE_PATHS}, got {resolved!r}")
    return resolved


class MLPHead(nn.Module):
    def __init__(self, dimensions: list[int], dropout: float = 0.1) -> None:
        super().__init__()
        layers: list[nn.Module] = [nn.LayerNorm(dimensions[0])]
        for input_dim, output_dim in zip(dimensions[:-2], dimensions[1:-1], strict=True):
            layers.extend((nn.Linear(input_dim, output_dim), nn.GELU(), nn.Dropout(dropout)))
        layers.append(nn.Linear(dimensions[-2], dimensions[-1]))
        self.network = nn.Sequential(*layers)

    def penultimate(self, value: torch.Tensor) -> torch.Tensor:
        """Return the representation immediately before the final projection."""
        return self.network[:-1](value)

    def project(self, feature: torch.Tensor) -> torch.Tensor:
        return self.network[-1](feature)

    def forward_with_penultimate(
        self, value: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.penultimate(value)
        return self.project(feature), feature

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output, _ = self.forward_with_penultimate(value)
        return output


class QwenValueModel(nn.Module):
    def __init__(
        self,
        config: dict[str, Any],
        *,
        trainable: bool = True,
        include_latent_head: bool = True,
        value_path: str | None = None,
    ) -> None:
        super().__init__()
        student_cfg = config["student"]
        model_path = str(student_cfg["model_source"])
        attention = "flash_attention_2" if student_cfg.get("use_flash_attention", True) else "sdpa"
        try:
            backbone = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                attn_implementation=attention,
                local_files_only=True,
            )
        except Exception:
            if attention == "sdpa":
                raise
            backbone = Qwen3VLForConditionalGeneration.from_pretrained(
                model_path,
                dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
            )
        backbone.config.use_cache = False
        # The fused Qwen3-VL vision-attention backward produces non-finite LoRA
        # gradients on the DTK/ROCm stack used by this project.  Keep the text
        # backend selected above, but use the numerically stable eager kernel for
        # the much smaller vision tower.
        backbone.model.visual.config._attn_implementation = student_cfg.get(
            "vision_attention_implementation", "eager"
        )
        if trainable:
            # Qwen3-VL checkpoints both its text and vision blocks.  Reentrant
            # checkpointing drops the vision-LoRA graph because the frozen patch
            # embedding produces an input without ``requires_grad``.  The
            # non-reentrant implementation does not have that restriction and
            # therefore keeps gradients for the visual qkv/proj adapters.
            backbone.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
            lora = LoraConfig(
                r=int(student_cfg["lora_rank"]),
                lora_alpha=int(student_cfg["lora_alpha"]),
                lora_dropout=float(student_cfg["lora_dropout"]),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
                target_modules=(
                    r".*(?:language_model.*\.(?:q_proj|k_proj|v_proj|o_proj)"
                    r"|visual\.blocks\.\d+\.attn\.(?:qkv|proj))$"
                ),
            )
            backbone = get_peft_model(backbone, lora)
            backbone.enable_input_require_grads()
        self.backbone = backbone
        text_config = getattr(backbone.config, "text_config", backbone.config)
        hidden_size = int(text_config.hidden_size)
        self.value_path = normalize_value_path(
            student_cfg.get("value_path") if value_path is None else value_path
        )
        if self.value_path == VALUE_PATH_LATENT_HIDDEN and not include_latent_head:
            raise ValueError("latent_hidden value path requires include_latent_head=True")
        value_input_size = 128 if self.value_path == VALUE_PATH_LATENT_HIDDEN else hidden_size
        self.value_feature_dim = value_input_size
        self.value_head = MLPHead(
            [value_input_size, 512, 256, int(student_cfg["value_bins"])]
        )
        self.latent_head = (
            MLPHead([hidden_size, 512, 128, int(student_cfg["latent_dim"])])
            if include_latent_head or self.value_path == VALUE_PATH_LATENT_HIDDEN
            else None
        )
        bins = torch.linspace(
            float(student_cfg["value_min"]),
            float(student_cfg["value_max"]),
            int(student_cfg["value_bins"]),
        )
        self.register_buffer("value_bin_centers", bins, persistent=True)

    def forward(self, **inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        output = self.backbone(
            **inputs,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )
        hidden = output.hidden_states[-1]
        attention_mask = inputs["attention_mask"]
        positions = torch.arange(attention_mask.shape[1], device=attention_mask.device)[None, :]
        last = (positions * attention_mask).argmax(dim=1)
        pooled = hidden[torch.arange(hidden.shape[0], device=hidden.device), last].float()
        latent = None
        latent_hidden = None
        if self.latent_head is not None:
            latent, latent_hidden = self.latent_head.forward_with_penultimate(pooled)
        value_feature = (
            latent_hidden if self.value_path == VALUE_PATH_LATENT_HIDDEN else pooled
        )
        if value_feature is None:
            raise RuntimeError("latent_hidden value feature was not computed")
        logits = self.value_head(value_feature)
        probabilities = logits.softmax(dim=-1)
        expected = (probabilities * self.value_bin_centers.float()).sum(dim=-1)
        result = {"value_logits": logits, "value": expected}
        if latent is not None:
            result["latent"] = latent
        return result


def architecture_metadata(model: QwenValueModel) -> dict[str, Any]:
    latent_dim = (
        int(model.latent_head.network[-1].out_features)
        if model.latent_head is not None
        else None
    )
    return {
        "version": ARCHITECTURE_VERSION,
        "value_path": model.value_path,
        "value_feature_dim": int(model.value_feature_dim),
        "latent_dim": latent_dim,
    }


def checkpoint_value_path(checkpoint: dict[str, Any]) -> str:
    """Read architecture metadata while treating all legacy checkpoints as direct."""
    architecture = checkpoint.get("architecture")
    if isinstance(architecture, dict) and architecture.get("value_path") is not None:
        return normalize_value_path(str(architecture["value_path"]))
    training_plan = checkpoint.get("training_plan")
    if isinstance(training_plan, dict) and training_plan.get("value_path") is not None:
        return normalize_value_path(str(training_plan["value_path"]))
    return VALUE_PATH_DIRECT


def load_processor(config: dict[str, Any]) -> Any:
    return AutoProcessor.from_pretrained(config["student"]["model_source"], local_files_only=True)


def process_student_batch(
    processor: Any,
    rows: list[dict[str, Any]],
    latent_lookup: dict[str, np.ndarray] | None,
    device: torch.device,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    texts = []
    images = []
    for row in rows:
        content = [{"type": "image"} for _ in range(3)]
        content.append({"type": "text", "text": row["metadata"]["task_prompt"]})
        messages = [{"role": "user", "content": content}]
        texts.append(processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True))
        images.extend(row["images"][:3])
    encoded = processor(text=texts, images=images, padding=True, return_tensors="pt")
    inputs = {key: value.to(device, non_blocking=True) for key, value in encoded.items() if torch.is_tensor(value)}
    targets: dict[str, torch.Tensor] = {
        "value_bin": torch.tensor(
            [int(row["metadata"]["value_bin"]) for row in rows], dtype=torch.long, device=device
        ),
        "value": torch.tensor(
            [float(row["metadata"]["value"]) for row in rows], dtype=torch.float32, device=device
        ),
    }
    if latent_lookup is not None:
        targets["latent"] = torch.from_numpy(
            np.stack([latent_lookup[row["key"]] for row in rows])
        ).to(device=device, dtype=torch.float32)
    return inputs, targets


def load_latent_lookup(artifact_root: str | Path, splits: tuple[str, ...]) -> dict[str, np.ndarray]:
    lookup: dict[str, np.ndarray] = {}
    root = Path(artifact_root) / "latents"
    for split in splits:
        for path in sorted((root / split).glob("rank-*.parquet")):
            table = pq.read_table(path, columns=["sample_key", "latent"])
            for key, latent in zip(table["sample_key"].to_pylist(), table["latent"].to_pylist(), strict=True):
                lookup[str(key)] = np.asarray(latent, dtype=np.float32)
    return lookup


def student_loss(
    output: dict[str, torch.Tensor], targets: dict[str, torch.Tensor], lambda_latent: float
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    value_loss = torch.nn.functional.cross_entropy(output["value_logits"], targets["value_bin"])
    latent_loss = torch.zeros((), device=value_loss.device)
    if lambda_latent > 0 and "latent" in targets:
        latent_loss = torch.nn.functional.huber_loss(output["latent"], targets["latent"])
    loss = value_loss + float(lambda_latent) * latent_loss
    return loss, {
        "loss": loss.detach(),
        "value_loss": value_loss.detach(),
        "latent_loss": latent_loss.detach(),
    }


def trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
