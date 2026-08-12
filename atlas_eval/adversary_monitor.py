"""Early warning for the adversary/encoder chase.

The v6 conditional adversary was disabled because the classifier and encoder
oscillated: the batch classifier climbed, the encoder destroyed its signal, and
the cycle repeated without the representation ever losing batch information.
That failure is visible within a few hundred steps if the classifier accuracy
is watched against chance, so it should never cost a full training run again.

Live use inside the training loop::

    monitor = AdversaryChaseMonitor(num_classes=len(dataset_ids))
    ...
    status = monitor.update(step, float(outputs["batch_adversary_accuracy"]))
    if status["should_abort"]:
        raise RuntimeError(status["reason"])

Post-hoc use on a finished run's per-epoch history::

    python -m atlas_eval.adversary_monitor --metrics <run>/metrics.json
"""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from statistics import mean
from typing import Any, Deque, Dict, List, Optional, Sequence


class AdversaryChaseMonitor:
    """Flags oscillating adversarial training from the classifier accuracy trace.

    ``window`` consecutive logged values are smoothed into ``segment`` sized
    chunks; a chase shows up as repeated direction changes across those chunks
    combined with a swing larger than ``amplitude``.
    """

    def __init__(
        self,
        num_classes: int,
        *,
        window: int = 40,
        segment: int = 4,
        amplitude: float = 0.1,
        min_direction_changes: int = 4,
        warmup_steps: int = 200,
    ) -> None:
        self.chance = 1.0 / max(int(num_classes), 2)
        self.window = int(window)
        self.segment = max(int(segment), 1)
        self.amplitude = float(amplitude)
        self.min_direction_changes = int(min_direction_changes)
        self.warmup_steps = int(warmup_steps)
        self.history: Deque[Dict[str, float]] = deque(maxlen=self.window)
        self.last_step = 0

    def update(
        self,
        step: int,
        adversary_accuracy: float,
        encoder_probe_accuracy: Optional[float] = None,
    ) -> Dict[str, Any]:
        self.last_step = int(step)
        row = {"step": float(step), "adversary_accuracy": float(adversary_accuracy)}
        if encoder_probe_accuracy is not None:
            row["encoder_probe_accuracy"] = float(encoder_probe_accuracy)
        self.history.append(row)
        return self.status()

    def status(self) -> Dict[str, Any]:
        values = [row["adversary_accuracy"] for row in self.history]
        report: Dict[str, Any] = {
            "num_observations": len(values),
            "chance_accuracy": self.chance,
            "should_abort": False,
            "reason": None,
        }
        if len(values) < self.segment * 3:
            return report
        segments = [
            mean(values[start : start + self.segment])
            for start in range(0, len(values) - self.segment + 1, self.segment)
        ]
        deltas = [later - earlier for earlier, later in zip(segments, segments[1:])]
        direction_changes = sum(
            1 for first, second in zip(deltas, deltas[1:]) if first * second < 0
        )
        swing = max(segments) - min(segments)
        latest = mean(values[-self.segment :])
        report.update(
            {
                "smoothed_segments": segments,
                "direction_changes": int(direction_changes),
                "swing": float(swing),
                "latest_accuracy": float(latest),
                "accuracy_above_chance": float(latest - self.chance),
            }
        )
        if self.last_step >= self.warmup_steps:
            if direction_changes >= self.min_direction_changes and swing >= self.amplitude:
                report["should_abort"] = True
                report["reason"] = (
                    f"adversary chase: {direction_changes} direction changes with a "
                    f"{swing:.3f} accuracy swing over the last {len(values)} logs"
                )
            elif latest - self.chance >= self.amplitude and direction_changes >= self.min_direction_changes // 2:
                report["reason"] = (
                    "adversary is still well above chance while oscillating; the encoder is "
                    "not removing batch information, only disrupting this classifier"
                )
        return report


def scan_history(
    rows: Sequence[Dict[str, Any]],
    num_classes: int,
    key: str = "train/batch_adversary_accuracy",
    **monitor_kwargs: Any,
) -> Dict[str, Any]:
    """Replay a recorded accuracy trace (per epoch or per step) through the monitor."""
    monitor = AdversaryChaseMonitor(num_classes, warmup_steps=0, **monitor_kwargs)
    observations: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        if key not in row:
            continue
        observations.append(monitor.update(int(row.get("epoch", index)), float(row[key])))
    if not observations:
        return {"num_observations": 0, "reason": f"no {key} in history", "should_abort": False}
    final = observations[-1]
    final["flagged_at_any_point"] = any(item["should_abort"] for item in observations)
    return final


def main(argv: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--metrics", required=True, help="metrics.json written by src/train.py")
    parser.add_argument("--num-classes", type=int, required=True, help="Adversary target classes, e.g. dataset count")
    parser.add_argument("--key", default="train/batch_adversary_accuracy")
    args = parser.parse_args(argv)
    with Path(args.metrics).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    history = payload.get("history") or payload.get("training", {}).get("history") or []
    report = scan_history(history, args.num_classes, args.key)
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
