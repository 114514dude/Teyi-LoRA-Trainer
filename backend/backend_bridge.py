"""Bridge the Electron UI to Diffusion Pipe's real train.py entry point.

The bridge deliberately uses only the Python standard library so validation and
configuration generation still work before the trainer's ML dependencies load.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import locale
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
from typing import Any


TAG_START = "__JSON_START__"
TAG_END = "__JSON_END__"
CORE_EVENT_PREFIX = "__LORA_UI_CORE_EVENT__"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

# Derived from the concrete enable_block_swap implementations in the supplied
# core.  This table is authoritative on the backend; renderer flags are never
# trusted for config generation.
MODEL_CAPABILITIES = {
    "anima": {"block_swap": True, "timestep_sampling": True},
    "flux": {"block_swap": True, "timestep_sampling": True},
    "flux2": {"block_swap": True, "timestep_sampling": True},
    "sdxl": {"block_swap": False, "timestep_sampling": False},
    "sd3": {"block_swap": False, "timestep_sampling": True},
    "qwen_image": {"block_swap": True, "timestep_sampling": True},
    "qwen2511": {"block_swap": True, "timestep_sampling": True},
    "wan": {"block_swap": True, "timestep_sampling": False},
    "hunyuan-video": {"block_swap": True, "timestep_sampling": True},
    "hunyuan_video_15": {"block_swap": True, "timestep_sampling": True},
    "hunyuan_image": {"block_swap": True, "timestep_sampling": True},
    "ltx-video": {"block_swap": False, "timestep_sampling": True},
    "ltx2": {"block_swap": True, "timestep_sampling": True},
    "lumina_2": {"block_swap": False, "timestep_sampling": True},
    "chroma": {"block_swap": True, "timestep_sampling": True},
    "cosmos_predict2": {"block_swap": True, "timestep_sampling": True},
    "z_image": {"block_swap": True, "timestep_sampling": True},
    "auraflow": {"block_swap": True, "timestep_sampling": True},
    "hidream": {"block_swap": True, "timestep_sampling": True},
    "omnigen2": {"block_swap": False, "timestep_sampling": True},
    "cosmos": {"block_swap": False, "timestep_sampling": False},
    "ernie_image": {"block_swap": True, "timestep_sampling": True},
    "ideogram4": {"block_swap": True, "timestep_sampling": True},
    "krea2": {"block_swap": True, "timestep_sampling": True},
}

MODEL_REQUIRED_PATHS = {
    "anima": (("transformer_path", "file"), ("vae_path", "file"), ("llm_path", "file")),
    "flux": (("diffusers_path", "folder"),),
    "flux2": (("diffusion_model", "file"), ("vae", "file"), ("text_encoder", "file")),
    "sdxl": (("checkpoint_path", "file"),),
    "sd3": (("diffusers_path", "folder"),),
    "qwen_image": (("diffusers_path", "folder"),),
    "qwen2511": (("diffusers_path", "folder"),),
    "wan": (("ckpt_path", "folder"),),
    "hunyuan-video": (("ckpt_path", "folder"),),
    "hunyuan_video_15": (("diffusion_model", "file"), ("vae", "file"), ("text_encoder", "file")),
    "hunyuan_image": (("transformer_path", "file"), ("vae_path", "file"), ("text_encoder_path", "file"), ("byt5_path", "file")),
    "ltx-video": (("diffusers_path", "folder"), ("single_file_path", "file")),
    "ltx2": (("diffusion_model", "file"), ("vae", "file"), ("text_encoder", "file")),
    "lumina_2": (("transformer_path", "file"), ("vae_path", "file"), ("llm_path", "file")),
    "chroma": (("diffusers_path", "folder"), ("transformer_path", "file")),
    "cosmos_predict2": (("transformer_path", "file"), ("vae_path", "file"), ("t5_path", "file")),
    "z_image": (("checkpoint_path", "folder"),),
    "auraflow": (("transformer_path", "file"), ("vae_path", "file"), ("text_encoder_path", "file")),
    "hidream": (("diffusers_path", "folder"), ("llama3_path", "folder")),
    "omnigen2": (("diffusers_path", "folder"),),
    "cosmos": (("transformer_path", "file"), ("vae_path", "file"), ("text_encoder_path", "file")),
    "ernie_image": (("diffusion_model", "file"), ("vae", "file"), ("text_encoder", "file")),
    "ideogram4": (("diffusion_model", "file"), ("vae", "file"), ("text_encoder", "file")),
    "krea2": (("diffusion_model", "file"), ("vae", "file"), ("text_encoder", "file")),
}


def emit(payload: dict[str, Any]) -> None:
    print(f"{TAG_START}{json.dumps(payload, ensure_ascii=False)}{TAG_END}", flush=True)


def normalized_path(value: Any) -> str:
    return str(value or "").strip().replace("\\", "/")


def toml_string(value: Any) -> str:
    return json.dumps(normalized_path(value), ensure_ascii=False)


def toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, list):
        return "[" + ", ".join(toml_value(item) for item in value) + "]"
    if isinstance(value, dict):
        return "{ " + ", ".join(f"{key} = {toml_value(item)}" for key, item in value.items()) + " }"
    return toml_string(value)


def write_table(lines: list[str], name: str, values: dict[str, Any]) -> None:
    lines.extend(["", f"[{name}]"])
    for key, value in values.items():
        if value is not None and value != "":
            lines.append(f"{key} = {toml_value(value)}")


def locate_trainer_root(explicit: str) -> Path:
    candidates = [explicit, os.environ.get("LORA_TRAINER_ROOT", ""), str(Path.cwd())]
    for candidate in candidates:
        if not candidate:
            continue
        root = Path(candidate).expanduser().resolve()
        if locate_train_script(root, required=False):
            return root
    raise FileNotFoundError("未找到整合包训练后端。请在“环境诊断”中选择包含 app/backend/core/train.py 的整合包根目录。")


def locate_train_script(root: Path, required: bool = True) -> Path | None:
    platform_core = "core" if os.name == "nt" else "core_linux"
    candidates = [
        root / "app" / "backend" / platform_core / "train.py",
        root / "backend" / platform_core / "train.py",
        root / "app" / "backend" / "core" / "train.py",
        root / "backend" / "core" / "train.py",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    if required:
        raise FileNotFoundError("整合包内缺少 app/backend/core/train.py。")
    return None


def parse_resolution(value: Any) -> list[list[int]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("训练分辨率格式无效。") from exc
    if not isinstance(value, list) or not value:
        raise ValueError("至少需要一个训练分辨率。")
    if len(value) == 2 and all(isinstance(item, (int, float)) for item in value):
        value = [value]
    result = []
    for pair in value:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError("训练分辨率必须是宽 × 高。")
        width, height = (int(pair[0]), int(pair[1]))
        if width < 64 or height < 64:
            raise ValueError("训练分辨率不能小于 64 × 64。")
        result.append([width, height])
    return result


def media_directories(dataset_path: Path, excluded: Path | None = None) -> list[Path]:
    excluded = excluded.resolve() if excluded else None
    result = []
    for directory, child_names, file_names in os.walk(dataset_path):
        current = Path(directory).resolve()
        child_names[:] = [name for name in child_names if name.casefold() != "cache"]
        if excluded and (current == excluded or excluded in current.parents):
            child_names[:] = []
            continue
        if any(Path(name).suffix.lower() in IMAGE_EXTENSIONS for name in file_names):
            result.append(current)
    return sorted(result, key=lambda item: str(item).casefold())


def dataset_counts(dataset_path: Path, excluded: Path | None = None) -> tuple[int, int]:
    images = pairs = 0
    for directory in media_directories(dataset_path, excluded):
        files = [item for item in directory.iterdir() if item.is_file()]
        directory_images = [item for item in files if item.suffix.lower() in IMAGE_EXTENSIONS]
        captions = {item.stem.casefold() for item in files if item.suffix.lower() == ".txt"}
        images += len(directory_images)
        pairs += sum(1 for item in directory_images if item.stem.casefold() in captions)
    return images, pairs


def validate(config: dict[str, Any], trainer_root: str) -> tuple[Path, Path]:
    root = locate_trainer_root(trainer_root)
    train_script = locate_train_script(root)
    dataset = config.get("dataset", {})
    dataset_path = Path(str(dataset.get("input_path", ""))).expanduser()
    if not dataset_path.is_dir():
        raise FileNotFoundError("训练数据集目录不存在。")
    parse_resolution(dataset.get("resolutions"))
    eval_subdir = str(dataset.get("eval_subdir", "")).strip()
    eval_path = (dataset_path / eval_subdir).resolve() if eval_subdir else None
    image_count, pair_count = dataset_counts(dataset_path, eval_path)
    if image_count == 0:
        raise ValueError("训练数据集根目录内没有找到支持的图片。")
    if dataset.get("skip_empty_caption", True) and pair_count == 0:
        raise ValueError("训练数据集根目录内没有找到“图片 + 同名 .txt 标注”组合。")
    if eval_subdir:
        eval_path = dataset_path / eval_subdir
        if not eval_path.is_dir():
            raise FileNotFoundError(f"验证集子目录不存在：{eval_path}")
        eval_images, eval_pairs = dataset_counts(eval_path)
        if eval_images == 0 or (dataset.get("skip_empty_caption", True) and eval_pairs == 0):
            raise ValueError("验证集子目录内没有可用的图片与 .txt 标注组合。")

    model = config.get("model", {})
    model_type = str(model.get("backend_type", ""))
    if model_type not in MODEL_CAPABILITIES:
        raise ValueError(f"训练核心不支持模型类型：{model_type or '（空）'}")
    paths = model.get("paths", {})
    labels = {field.get("key"): field.get("label") for field in model.get("required_paths", [])}
    for key, expected in MODEL_REQUIRED_PATHS[model_type]:
        label = labels.get(key) or key
        value = Path(str(paths.get(key, ""))).expanduser()
        exists = value.is_dir() if expected == "folder" else value.is_file()
        if not exists:
            raise FileNotFoundError(f"模型路径无效：{label}")

    training = config.get("training", {})
    for key, minimum in (("epochs", 1), ("micro_batch_size_per_gpu", 1), ("gradient_accumulation_steps", 1), ("save_every_n_steps", 1), ("steps_per_print", 1)):
        try:
            if int(training.get(key, minimum)) < minimum:
                raise ValueError
        except (TypeError, ValueError) as exc:
            raise ValueError(f"训练参数 {key} 必须是大于等于 {minimum} 的整数。") from exc
    if int(training.get("pipeline_stages", 1)) != 1 and int(training.get("blocks_to_swap", 0)) > 0:
        raise ValueError("Block Swap 仅支持 pipeline_stages = 1。")

    output_name = str(config.get("training", {}).get("output_folder_name", "")).strip()
    if not output_name or output_name in {".", ".."} or Path(output_name).name != output_name:
        raise ValueError("输出文件夹名称不能为空，也不能包含路径分隔符。")
    return root, train_script


def scan_clip_captions(config: dict[str, Any], train_script: Path) -> dict[str, Any]:
    """Precisely scan SDXL captions with the CLIP tokenizer shipped in the core."""
    if config.get("model", {}).get("backend_type") != "sdxl":
        return {"applicable": False, "exact": True, "scanned": 0, "overLimit": []}
    dataset_path = Path(str(config["dataset"]["input_path"])).expanduser().resolve()
    tokenizer_dir = train_script.parent / "submodules" / "ComfyUI" / "comfy" / "sd1_tokenizer"
    try:
        from transformers import CLIPTokenizer
        tokenizer = CLIPTokenizer.from_pretrained(str(tokenizer_dir), local_files_only=True)
    except Exception as exc:
        return {
            "applicable": True,
            "exact": False,
            "scanned": 0,
            "overLimit": [],
            "message": f"无法加载训练核心自带的 CLIP tokenizer，未完成精确 77 Token 扫描：{exc}",
        }
    results = []
    scanned = 0
    eval_subdir = str(config.get("dataset", {}).get("eval_subdir", "")).strip()
    excluded = (dataset_path / eval_subdir).resolve() if eval_subdir else None
    caption_paths = []
    for directory in media_directories(dataset_path, excluded):
        caption_paths.extend(image.with_suffix(".txt") for image in directory.iterdir() if image.is_file() and image.suffix.lower() in IMAGE_EXTENSIONS and image.with_suffix(".txt").is_file())
    for caption_path in sorted(set(caption_paths), key=lambda item: str(item).casefold()):
        try:
            caption = caption_path.read_text(encoding="utf-8").strip()
        except UnicodeDecodeError as exc:
            results.append({"file": str(caption_path), "tokenCount": None, "error": f"标注不是有效 UTF-8：{exc}"})
            continue
        token_count = len(tokenizer(caption, add_special_tokens=False, truncation=False)["input_ids"]) + 2
        scanned += 1
        if token_count > 77:
            results.append({"file": str(caption_path), "tokenCount": token_count, "limit": 77})
    return {"applicable": True, "exact": True, "scanned": scanned, "overLimit": results, "limit": 77}


def model_table(config: dict[str, Any]) -> dict[str, Any]:
    source = config.get("model", {})
    adapter = config.get("adapter", {})
    values: dict[str, Any] = {
        "type": source.get("backend_type", "anima"),
        "dtype": source.get("dtype", "bfloat16"),
        "transformer_dtype": source.get("transformer_dtype", "bfloat16"),
    }
    values.update(source.get("defaults") or {})
    values.update({key: normalized_path(value) for key, value in (source.get("paths") or {}).items() if value})
    if source.get("text_encoder_array") and values.get("text_encoder"):
        values["text_encoders"] = [{
            "path": values.pop("text_encoder"),
            "type": source.get("text_encoder_type") or "stable_diffusion",
        }]

    model_type = values["type"]
    capability = MODEL_CAPABILITIES[model_type]
    if capability["timestep_sampling"]:
        method = str(source.get("timestep_sample_method", "logit_normal"))
        if method not in {"logit_normal", "uniform"}:
            raise ValueError(f"不支持的时间步采样方法：{method}")
        values["timestep_sample_method"] = method
    base_lr = float(adapter.get("lr", 1e-4))
    if model_type == "sdxl":
        values.update({
            "unet_lr": float(adapter.get("unet_lr", base_lr)),
            "text_encoder_1_lr": float(adapter.get("text_encoder_1_lr", base_lr)),
            "text_encoder_2_lr": float(adapter.get("text_encoder_2_lr", base_lr)),
        })
    elif model_type in {"anima", "cosmos_predict2"}:
        values["llm_adapter_lr"] = float(adapter.get("text_encoder_lr", base_lr))
    return values


def build_dataset(config: dict[str, Any], dataset_path: Path) -> str:
    source = config.get("dataset", {})
    lines = [
        f"resolutions = {toml_value(parse_resolution(source.get('resolutions')))}",
        f"enable_ar_bucket = {toml_value(bool(source.get('enable_ar_bucket', True)))}",
        f"min_ar = {toml_value(float(source.get('min_ar', 0.5)))}",
        f"max_ar = {toml_value(float(source.get('max_ar', 2.0)))}",
        f"num_ar_buckets = {toml_value(int(source.get('num_ar_buckets', 7)))}",
        f"skip_empty_caption = {toml_value(bool(source.get('skip_empty_caption', True)))}",
    ]
    excluded = None
    eval_subdir = str(source.get("eval_subdir", "")).strip()
    if eval_subdir and dataset_path == Path(str(source.get("input_path", ""))).expanduser().resolve():
        excluded = (dataset_path / eval_subdir).resolve()
    directories = media_directories(dataset_path, excluded)
    if not directories:
        directories = [dataset_path]
    for directory in directories:
        lines.extend(["", "[[directory]]", f"path = {toml_string(directory)}", f"num_repeats = {toml_value(int(source.get('num_repeats', 1)))}"])
        if source.get("mask_path"):
            lines.append(f"mask_path = {toml_string(source['mask_path'])}")
    return "\n".join(lines) + "\n"


def build_train(config: dict[str, Any], dataset_config: Path, eval_config: Path | None, output_dir: Path) -> str:
    training = config.get("training", {})
    adapter = config.get("adapter", {})
    optimizer_names = {
        "AdamW": "adamw_optimi",
        "AdamW8bit": "AdamW8bit",
        "AdamW8bitKahan": "AdamW8bitKahan",
        "Prodigy": "Prodigy",
        "Lion": "Lion",
    }
    try:
        optimizer_ui = json.loads(training.get("optimizer_config") or "{}")
    except json.JSONDecodeError:
        optimizer_ui = {}
    optimizer_name = optimizer_names.get(optimizer_ui.get("type"), optimizer_ui.get("type") or "adamw_optimi")
    scheduler = str(optimizer_ui.get("scheduler") or "constant").lower()
    if scheduler not in {"constant", "linear", "cosine"}:
        scheduler = "cosine"

    top: dict[str, Any] = {
        "epochs": int(training.get("epochs", 20)),
        "micro_batch_size_per_gpu": int(training.get("micro_batch_size_per_gpu", 1)),
        "gradient_accumulation_steps": int(training.get("gradient_accumulation_steps", 1)),
        "warmup_steps": int(training.get("warmup_steps", 0)),
        "output_dir": normalized_path(output_dir),
        "dataset": normalized_path(dataset_config),
        "save_dtype": training.get("save_dtype", "bfloat16"),
        "partition_method": training.get("partition_method", "parameters"),
        "activation_checkpointing": bool(training.get("activation_checkpointing", True)),
        "pipeline_stages": int(training.get("pipeline_stages", 1)),
        "caching_batch_size": int(training.get("caching_batch_size", 1)),
        "save_every_n_steps": max(1, int(training.get("save_every_n_steps", 1000))),
        "gradient_clipping": float(training.get("gradient_clipping", 1.0)),
        "lr_scheduler": scheduler,
        "steps_per_print": max(1, int(training.get("steps_per_print", 1))),
        "logging_steps": 1,
    }
    blocks_to_swap = int(training.get("blocks_to_swap", 0))
    model_type = str(config.get("model", {}).get("backend_type", ""))
    if MODEL_CAPABILITIES[model_type]["block_swap"] and blocks_to_swap > 0:
        top["blocks_to_swap"] = blocks_to_swap
    if eval_config:
        top["eval_datasets"] = [{"name": "validation_set", "config": normalized_path(eval_config)}]
        top.update({
            "eval_micro_batch_size_per_gpu": 1,
            "eval_gradient_accumulation_steps": 1,
            "eval_every_n_epochs": 1,
            "eval_before_first_step": False,
        })

    lines = [f"{key} = {toml_value(value)}" for key, value in top.items()]
    write_table(lines, "model", model_table(config))
    write_table(lines, "optimizer", {
        "type": optimizer_name,
        "lr": float(adapter.get("lr", 1e-4)),
        "weight_decay": 0.01,
        "eps": 1e-8,
        "betas": [0.9, 0.99],
    })
    adapter_type = str(adapter.get("type", "LoRA"))
    if adapter_type != "None (FFT)":
        adapter_values: dict[str, Any] = {
            "type": "lokr" if adapter_type == "LoKr" else "lora",
            "rank": max(1, int(adapter.get("rank", 32))),
            "dtype": config.get("model", {}).get("dtype", "bfloat16"),
        }
        if adapter_type == "LoKr":
            adapter_values.update({"decompose_factor": int(adapter.get("decompose_factor", -1)), "rank_dropout": float(adapter.get("dropout", 0))})
        else:
            adapter_values["dropout"] = float(adapter.get("dropout", 0))
        if adapter.get("init_from"):
            adapter_values["init_from_existing"] = normalized_path(adapter["init_from"])
        write_table(lines, "adapter", adapter_values)
    write_table(lines, "monitoring", {"enable_wandb": False})
    return "\n".join(lines) + "\n"


def prepare(config: dict[str, Any], trainer_root: str) -> dict[str, Any]:
    root, train_script = validate(config, trainer_root)
    training = config.get("training", {})
    output_root = Path(str(training.get("output_path") or (root / "output"))).expanduser().resolve()
    session = output_root / dt.datetime.now().strftime("%Y%m%d_%H-%M-%S")
    suffix = 1
    while session.exists():
        session = session.with_name(f"{session.name}-{suffix}")
        suffix += 1
    session.mkdir(parents=True)

    dataset_path = Path(str(config["dataset"]["input_path"])).expanduser().resolve()
    dataset_config = session / "dataset.toml"
    dataset_config.write_text(build_dataset(config, dataset_path), encoding="utf-8")
    eval_config = None
    eval_subdir = str(config.get("dataset", {}).get("eval_subdir", "")).strip()
    if eval_subdir:
        eval_path = dataset_path / eval_subdir
        if not eval_path.is_dir():
            raise FileNotFoundError(f"验证集子目录不存在：{eval_path}")
        eval_config = session / "evaldataset.toml"
        eval_config.write_text(build_dataset(config, eval_path), encoding="utf-8")
    output_dir = session / str(training["output_folder_name"])
    train_config = session / "trainconfig.toml"
    train_config.write_text(build_train(config, dataset_config, eval_config, output_dir), encoding="utf-8")
    excluded = (dataset_path / eval_subdir).resolve() if eval_subdir else None
    images, pairs = dataset_counts(dataset_path, excluded)
    usable = pairs if config.get("dataset", {}).get("skip_empty_caption", True) else images
    repeats = max(1, int(config.get("dataset", {}).get("num_repeats", 1)))
    batch = max(1, int(training.get("micro_batch_size_per_gpu", 1)))
    accumulation = max(1, int(training.get("gradient_accumulation_steps", 1)))
    data_parallel = max(1, int(training.get("num_gpus", 1))) if os.name != "nt" else 1
    steps_per_epoch = max(1, (usable * repeats) // (batch * accumulation * data_parallel))
    total_steps = steps_per_epoch * max(1, int(training.get("epochs", 1)))
    return {
        "trainer_root": str(root),
        "train_script": str(train_script),
        "session_dir": str(session),
        "config_path": str(train_config),
        "dataset_config": str(dataset_config),
        "eval_config": str(eval_config) if eval_config else "",
        "output_dir": str(output_dir),
        "total_steps": total_steps,
        "total_steps_estimated": True,
    }


def training_command(prepared: dict[str, Any], config: dict[str, Any]) -> list[str]:
    runner = Path(__file__).with_name("training_runner.py")
    train_args = [str(runner), "--ui-target", prepared["train_script"], "--config", prepared["config_path"]]
    start = config.get("start_args", {})
    resume = str(start.get("resume_from_checkpoint", "")).strip()
    if resume:
        train_args.extend(["--resume_from_checkpoint", resume])
    for key in ("reset_dataloader", "reset_optimizer_params", "cache_only", "regenerate_cache", "trust_cache", "i_know_what_i_am_doing"):
        if start.get(key):
            train_args.append("--" + key)
    if start.get("dump_dataset"):
        train_args.extend(["--dump_dataset", str(start["dump_dataset"])])
    train_args.append("--deepspeed")
    if os.name == "nt" or os.environ.get("LORA_TRAINER_DIRECT") == "1":
        return [sys.executable, *train_args]
    deepspeed = Path(sys.executable).parent / "deepspeed"
    launcher = str(deepspeed) if deepspeed.is_file() else "deepspeed"
    num_gpus = max(1, int(config.get("training", {}).get("num_gpus", 1)))
    # The wrapper itself launches in the selected interpreter; the original
    # trainer still initializes DeepSpeed exactly as before.
    return [sys.executable, *train_args]


def decode_console_line(raw: bytes) -> tuple[str, str]:
    """Decode a complete console record without corrupting split UTF-8."""
    encodings = ["utf-8", locale.getpreferredencoding(False), "gb18030"]
    for encoding in dict.fromkeys(item for item in encodings if item):
        try:
            return raw.decode(encoding, errors="strict"), encoding
        except (UnicodeDecodeError, LookupError):
            continue
    return raw.decode("utf-8", errors="replace"), "utf-8-replace"


def record_kind(line: str) -> str:
    value = re.sub(r"\s+", " ", line).strip().lower()
    if re.search(r"lnk1181.*(?:aio|cufile)\.lib|redirects are currently not supported|need pytorch with cu130|optional.*(?:aio|cufile)", value):
        return "noise"
    if re.search(r"traceback|runtimeerror|\berror\b|exception|fatal|failed", value):
        return "error"
    if re.search(r"\bwarning\b|\bwarn\b|fallback", value):
        return "warning"
    if re.search(r"saving model|checkpoint|started new epoch|cache.*(?:complete|finished)|training.*(?:complete|finished)", value):
        return "event"
    if re.search(r"\bstep\s*[=:]|\bloss\s*[=:]|\blr\s*[=:]|epoch", value):
        return "progress"
    if re.search(r"load|cache|dataset|model|optimizer", value):
        return "loading"
    return "normal"


def iter_console_records(stream, raw_log):
    pending = bytearray()
    while True:
        chunk = stream.read(65536)
        if not chunk:
            break
        raw_log.write(chunk)
        raw_log.flush()
        pending.extend(chunk)
        start = 0
        for index, byte in enumerate(pending):
            if byte not in (10, 13):
                continue
            if index > start:
                yield bytes(pending[start:index])
            start = index + 1
        if start:
            del pending[:start]
    if pending:
        yield bytes(pending)


def run_training(prepared: dict[str, Any], config: dict[str, Any]) -> int:
    command = training_command(prepared, config)
    emit({"status": "TRAINING_STARTED", **prepared, "command": command})
    num_gpus = max(1, int(config.get("training", {}).get("num_gpus", 1)))
    child_env = dict(os.environ)
    # Do not force Python UTF-8 mode on the training core. On Chinese Windows,
    # optional compiler probes emit GBK and third-party subprocess readers use
    # the locale codec. Forcing UTF-8 makes those reader threads crash.
    child_env.pop("PYTHONUTF8", None)
    child_env.update({
        "LOCAL_SIZE": str(num_gpus if os.name != "nt" else 1),
        "LOCAL_WORLD_SIZE": str(num_gpus if os.name != "nt" else 1),
        "PYTHONIOENCODING": "utf-8",
        "PYTHONUNBUFFERED": "1",
    })
    raw_log_path = Path(prepared["session_dir"]) / "training-raw.log"
    emit({"status": "RAW_LOG_READY", "rawLogPath": str(raw_log_path), "outputDir": prepared["output_dir"]})
    child = subprocess.Popen(
        command,
        cwd=str(Path(prepared["train_script"]).parent),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=False,
        bufsize=0,
        env=child_env,
    )

    def stop_child(_signum: int, _frame: Any) -> None:
        if child.poll() is None:
            child.terminate()

    signal.signal(signal.SIGTERM, stop_child)
    if hasattr(signal, "SIGINT"):
        signal.signal(signal.SIGINT, stop_child)
    assert child.stdout is not None
    with open(raw_log_path, "ab", buffering=0) as raw_log:
        for raw in iter_console_records(child.stdout, raw_log):
            line, encoding = decode_console_line(raw)
            if line.startswith(CORE_EVENT_PREFIX):
                try:
                    payload = json.loads(line[len(CORE_EVENT_PREFIX):])
                    payload.setdefault("outputDir", prepared["output_dir"])
                    payload.setdefault("rawLogPath", str(raw_log_path))
                    emit(payload)
                    continue
                except json.JSONDecodeError:
                    pass
            kind = record_kind(line)
            emit({"status": "LOG", "line": line, "kind": kind, "encoding": encoding, "rawLogPath": str(raw_log_path)})
            if kind == "event" and "saving model" in line.lower():
                emit({"status": "CHECKPOINT_SAVED", "message": line.strip(), "outputDir": prepared["output_dir"]})
    code = child.wait()
    emit({"status": "COMPLETED" if code == 0 else "FAILED", "code": code, "outputDir": prepared["output_dir"], "rawLogPath": str(raw_log_path)})
    return code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", required=True, choices=["validate_config", "preview_plan", "generate_config", "start_training", "check_requirements"])
    parser.add_argument("--full_config", default="{}")
    parser.add_argument("--trainer_root", default="")
    args = parser.parse_args()
    try:
        if args.action == "check_requirements":
            root = locate_trainer_root(args.trainer_root)
            emit({"success": True, "trainer_root": str(root), "train_script": str(locate_train_script(root))})
            return 0
        config = json.loads(args.full_config)
        root, train_script = validate(config, args.trainer_root)
        dataset_path = Path(str(config["dataset"]["input_path"])).expanduser().resolve()
        eval_subdir = str(config.get("dataset", {}).get("eval_subdir", "")).strip()
        excluded = (dataset_path / eval_subdir).resolve() if eval_subdir else None
        images, pairs = dataset_counts(dataset_path, excluded)
        caption_scan = scan_clip_captions(config, train_script)
        if args.action == "validate_config":
            emit({"success": True, "trainer_root": str(root), "train_script": str(train_script), "images": images, "pairs": pairs, "captionScan": caption_scan})
            return 0
        if args.action == "preview_plan":
            training = config.get("training", {})
            usable = pairs if config.get("dataset", {}).get("skip_empty_caption", True) else images
            repeats = max(1, int(config.get("dataset", {}).get("num_repeats", 1)))
            batch = max(1, int(training.get("micro_batch_size_per_gpu", 1)))
            accumulation = max(1, int(training.get("gradient_accumulation_steps", 1)))
            data_parallel = max(1, int(training.get("num_gpus", 1))) if os.name != "nt" else 1
            samples_per_epoch = usable * repeats
            steps_per_epoch = max(1, samples_per_epoch // (batch * accumulation * data_parallel))
            planned_steps = steps_per_epoch * max(1, int(training.get("epochs", 1)))
            emit({"success": True, "images": images, "pairs": pairs, "usable": usable, "samplesPerEpoch": samples_per_epoch, "stepsPerEpoch": steps_per_epoch, "totalSteps": planned_steps, "estimated": True, "captionScan": caption_scan})
            return 0
        prepared = prepare(config, args.trainer_root)
        if args.action == "generate_config":
            emit({"success": True, "status": "CONFIG_GENERATED", **prepared})
            return 0
        return run_training(prepared, config)
    except Exception as exc:
        emit({"success": False, "status": "ERROR", "message": str(exc), "error_type": type(exc).__name__})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
