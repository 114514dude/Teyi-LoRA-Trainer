"""Recursive WD14 tagging with the official SmilingWolf EVA02-large v3 model."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time
import urllib.request


MODEL_REPO = "SmilingWolf/wd-eva02-large-tagger-v3"
MODEL_BASE = f"https://huggingface.co/{MODEL_REPO}/resolve/main"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
TAG_START = "__JSON_START__"
TAG_END = "__JSON_END__"


def emit(payload):
    print(f"{TAG_START}{json.dumps(payload, ensure_ascii=False)}{TAG_END}", flush=True)


def download(url: str, destination: Path) -> None:
    if destination.is_file() and destination.stat().st_size > 0:
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "Exceptional-LoRA-Trainer/0.3"})
    with urllib.request.urlopen(request, timeout=60) as response, open(temporary, "wb") as output:
        total = int(response.headers.get("Content-Length", 0))
        completed = 0
        last_percent = -1
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            output.write(chunk)
            completed += len(chunk)
            percent = int(completed * 100 / total) if total else 0
            if percent != last_percent:
                emit({"status": "DOWNLOADING", "file": destination.name, "completed": completed, "total": total, "percent": percent})
                last_percent = percent
    temporary.replace(destination)


def load_tags(csv_path: Path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows.append((row["name"], int(row["category"])))
    return rows


def prepare_image(path: Path, size: int, np, Image):
    image = Image.open(path).convert("RGB")
    scale = size / max(image.size)
    resized = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (size, size), "white")
    canvas.paste(resized, ((size - resized.width) // 2, (size - resized.height) // 2))
    array = np.asarray(canvas, dtype=np.float32)
    array = array[:, :, ::-1]
    return np.expand_dims(array, axis=0)


def split_words(value: str):
    return [word.strip() for word in value.replace("\n", ",").split(",") if word.strip()]


def canonical(value: str):
    return value.strip().lower().replace("_", " ")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root")
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--model-dir", default="")
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--character-threshold", type=float, default=0.85)
    parser.add_argument("--trigger-words", default="")
    parser.add_argument("--blocked-words", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--replace-underscores", action="store_true")
    args = parser.parse_args()

    try:
        import numpy as np
        import onnxruntime as ort
        from PIL import Image
    except ImportError as exc:
        emit({"status": "ERROR", "message": f"WD14 缺少依赖：{exc.name}。请在整合包 Python 中安装 onnxruntime、numpy 和 Pillow。"})
        return 2

    if args.diagnose:
        available = ort.get_available_providers()
        cuda_advertised = "CUDAExecutionProvider" in available
        cuda_error = ""
        try:
            # Creating a tiny session forces ORT to load its CUDA/cuDNN DLL chain.
            # The model is intentionally not needed for this environment diagnostic.
            import tempfile
            from pathlib import Path as _Path
            probe = _Path(tempfile.gettempdir()) / "wd14-ort-provider-probe.onnx"
            # A missing probe is expected; provider advertisement is still useful.
            if probe.is_file(): ort.InferenceSession(str(probe), providers=["CUDAExecutionProvider"])
        except Exception as exc:
            cuda_error = str(exc)
        emit({"status": "DIAGNOSTICS", "availableProviders": available, "cudaAdvertised": cuda_advertised, "cudaLoadError": cuda_error, "message": "CUDA provider 可用性需在加载 WD14 模型时最终确认。" if cuda_advertised else "当前 Python 的 ONNX Runtime 未提供 CUDAExecutionProvider。"})
        return 0

    if not args.root:
        emit({"status": "ERROR", "message": "图片根目录不存在。"})
        return 2
    if not args.model_dir:
        emit({"status": "ERROR", "message": "WD14 模型目录不能为空。"})
        return 2
    root = Path(args.root).expanduser().resolve()
    if not root.is_dir():
        emit({"status": "ERROR", "message": "图片根目录不存在。"})
        return 2
    model_dir = Path(args.model_dir).expanduser().resolve() / "wd-eva02-large-tagger-v3"
    model_path = model_dir / "model.onnx"
    tags_path = model_dir / "selected_tags.csv"
    try:
        emit({"status": "MODEL_PREPARING", "model": MODEL_REPO})
        download(f"{MODEL_BASE}/model.onnx?download=true", model_path)
        download(f"{MODEL_BASE}/selected_tags.csv?download=true", tags_path)
        available = ort.get_available_providers()
        cuda_error = ""
        session = None
        if "CUDAExecutionProvider" in available:
            try:
                session = ort.InferenceSession(str(model_path), providers=["CUDAExecutionProvider"])
                if session.get_providers()[0] != "CUDAExecutionProvider":
                    cuda_error = "ONNX Runtime 未能激活 CUDAExecutionProvider。"
                    session = None
            except Exception as exc:
                cuda_error = str(exc)
        if session is None:
            session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        input_meta = session.get_inputs()[0]
        size = int(input_meta.shape[1])
        tag_rows = load_tags(tags_path)
        images = sorted(item for item in root.rglob("*") if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)
        active_provider = session.get_providers()[0]
        emit({"status": "STARTED", "total": len(images), "model": MODEL_REPO, "provider": active_provider, "cpuFallback": active_provider == "CPUExecutionProvider"})
        if active_provider == "CPUExecutionProvider":
            detail = f" 详细原因：{cuda_error}" if cuda_error else ""
            emit({"status": "WARNING", "message": f"WD14 CUDA Provider 不可用，已回退到 CPU；结果不变，但速度会更慢。{detail}", "availableProviders": available, "cudaLoadError": cuda_error})
        blocked = {canonical(word) for word in split_words(args.blocked_words)}
        triggers = split_words(args.trigger_words)
        written = skipped = failed = total_tags = 0
        started_at = time.perf_counter()
        for index, image_path in enumerate(images, 1):
            image_started_at = time.perf_counter()
            caption_path = image_path.with_suffix(".txt")
            if caption_path.exists() and not args.overwrite:
                skipped += 1
                emit({"status": "PROGRESS", "current": index, "total": len(images), "written": written, "skipped": skipped, "failed": failed, "file": str(image_path), "result": "skipped", "seconds": time.perf_counter() - image_started_at})
                continue
            image_stats = {"tagCount": 0, "maxConfidence": None, "minConfidence": None, "result": "failed"}
            try:
                tensor = prepare_image(image_path, size, np, Image)
                probabilities = session.run(None, {input_meta.name: tensor})[0][0]
                selected = []
                for (name, category), probability in zip(tag_rows, probabilities):
                    threshold = args.character_threshold if category == 4 else args.threshold
                    if category not in (0, 4) or float(probability) < threshold or canonical(name) in blocked:
                        continue
                    selected.append((name, float(probability)))
                selected.sort(key=lambda item: item[1], reverse=True)
                generated = [name.replace("_", " ") if args.replace_underscores else name for name, _ in selected]
                final_tags = list(dict.fromkeys([*triggers, *generated]))
                caption_path.write_text(", ".join(final_tags), encoding="utf-8")
                written += 1
                total_tags += len(final_tags)
                image_stats = {
                    "tagCount": len(final_tags),
                    "generatedTagCount": len(generated),
                    "maxConfidence": selected[0][1] if selected else None,
                    "minConfidence": selected[-1][1] if selected else None,
                    "result": "written",
                    "caption": str(caption_path),
                }
            except Exception as exc:
                failed += 1
                print(f"[WD14] {image_path}: {exc}", file=sys.stderr, flush=True)
            emit({"status": "PROGRESS", "current": index, "total": len(images), "written": written, "skipped": skipped, "failed": failed, "file": str(image_path), "seconds": time.perf_counter() - image_started_at, **image_stats})
        elapsed = time.perf_counter() - started_at
        emit({"status": "COMPLETED", "total": len(images), "written": written, "skipped": skipped, "failed": failed, "totalTags": total_tags, "elapsedSeconds": elapsed, "averageSeconds": elapsed / len(images) if images else 0, "model": MODEL_REPO, "provider": active_provider})
        return 0 if failed == 0 else 1
    except Exception as exc:
        emit({"status": "ERROR", "message": str(exc), "error_type": type(exc).__name__})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
