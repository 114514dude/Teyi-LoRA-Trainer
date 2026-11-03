"""Run Diffusion Pipe and expose read-only UI telemetry.

The target trainer remains the authoritative implementation.  This wrapper uses
Python's line tracing only in the target ``train.py`` frame so it can publish
values the trainer already computed (step, epoch, loss, exact epoch length and
learning rate) without changing its optimizer, data loader, or training math.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
from pathlib import Path
import runpy
import sys
import time


EVENT_PREFIX = "__LORA_UI_CORE_EVENT__"
IS_MAIN_RANK = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) == 0


def emit(payload: dict) -> None:
    if IS_MAIN_RANK:
        print(EVENT_PREFIX + json.dumps(payload, ensure_ascii=False, separators=(",", ":")), flush=True)


def assignment_following_lines(source: str) -> tuple[int, int]:
    """Return lines immediately after the exact-plan and train-loss assignments."""
    tree = ast.parse(source)
    plan_line = progress_line = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == "steps_per_epoch":
            plan_line = max(plan_line, int(getattr(node, "end_lineno", node.lineno)) + 1)
        if isinstance(target, ast.Name) and target.id == "loss" and isinstance(node.value, ast.Call):
            function = node.value.func
            if isinstance(function, ast.Attribute) and function.attr == "item":
                inner = function.value
                if isinstance(inner, ast.Call) and isinstance(inner.func, ast.Attribute) and inner.func.attr == "train_batch":
                    progress_line = int(getattr(node, "end_lineno", node.lineno)) + 1
    return plan_line, progress_line


def learning_rates(optimizer) -> list[float]:
    result = []
    for group in getattr(optimizer, "param_groups", ()):
        try:
            result.append(float(group["lr"]))
        except (KeyError, TypeError, ValueError):
            continue
    return result


def main() -> int:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--ui-target", required=True)
    known, target_args = parser.parse_known_args()
    target = Path(known.ui_target).resolve()
    source = target.read_text(encoding="utf-8")
    plan_line, progress_line = assignment_following_lines(source)
    if not plan_line or not progress_line:
        emit({"status": "WARNING", "message": "训练核心未找到兼容的结构化指标采集点；训练仍会启动，但该核心版本无法提供逐步指标。"})
    started_at = time.perf_counter()
    last_step_at = started_at
    step_durations: list[float] = []
    total_steps = 0
    # runpy compiles the target using this exact filename. Comparing that
    # string is deliberately cheap: a trace callback runs on every Python
    # call event, including imports. Resolving a Path here made large Torch /
    # DeepSpeed imports take tens of minutes on Windows before train.py could
    # print its first line.
    target_name = str(target)

    def trace(frame, event, _arg):
        nonlocal last_step_at, total_steps
        if frame.f_code.co_filename != target_name:
            return None
        if event != "line":
            return trace
        values = frame.f_locals
        # The line immediately following the assignment can be blank, and
        # Python does not emit trace events for blank lines. Capture at the
        # first subsequent executable line instead.
        if total_steps == 0 and frame.f_lineno >= plan_line and "steps_per_epoch" in values:
            steps_per_epoch = int(values["steps_per_epoch"])
            config = values.get("config", {})
            epochs = int(config.get("epochs", 1))
            total_steps = steps_per_epoch * epochs
            max_steps = int(config.get("max_steps", 0) or 0)
            if max_steps > 0:
                total_steps = min(total_steps, max_steps)
            last_step_at = time.perf_counter()
            emit({"status": "TRAINING_PLAN", "stepsPerEpoch": steps_per_epoch, "totalSteps": total_steps, "estimated": False})
        elif frame.f_lineno == progress_line and "loss" in values and "step" in values:
            now = time.perf_counter()
            step = int(values["step"])
            rates = learning_rates(values.get("optimizer"))
            step_seconds = max(0.0, now - last_step_at)
            step_durations.append(step_seconds)
            del step_durations[:-20]
            average_step_seconds = sum(step_durations) / len(step_durations)
            elapsed = max(0.0, now - started_at)
            last_step_at = now
            remaining = max(0, total_steps - step) * average_step_seconds if total_steps else None
            emit({
                "status": "PROGRESS",
                "currentStep": step,
                "totalSteps": total_steps,
                "epoch": int(values.get("epoch", 0)),
                "loss": float(values["loss"]),
                "learningRate": rates[0] if rates else None,
                "learningRates": rates,
                "stepSeconds": average_step_seconds,
                "lastStepSeconds": step_seconds,
                "elapsedSeconds": elapsed,
                "remainingSeconds": remaining,
                "estimated": False,
                "source": "core-trace",
            })
        return trace

    sys.argv = [str(target), *target_args]
    sys.path.insert(0, str(target.parent))
    if plan_line and progress_line:
        sys.settrace(trace)
    try:
        runpy.run_path(str(target), run_name="__main__")
    finally:
        sys.settrace(None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
