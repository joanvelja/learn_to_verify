# pvg/components/metrics_logger.py
import logging
import statistics
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Tuple

import torch
from accelerate.utils import gather_object

from pvg.components.accelerator_manager import AcceleratorManager

logger = logging.getLogger("pvg.metrics_logger")


@dataclass
class _Metric:
    value: float
    step: int


@dataclass
class _RingBuffer:
    """FIFO buffer keyed by (phase, mode) -> model -> metric_name -> List[_Metric]"""

    store: Dict[Tuple[str, str], Dict[str, Dict[str, List[_Metric]]]] = field(
        default_factory=lambda: defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
    )

    def add(
        self, phase: str, mode: str, model: str, name: str, value: float, step: int
    ) -> None:
        self.store[(phase, mode)][model][name].append(_Metric(value, step))

    def pop_phase_mode(self, phase: str, mode: str):
        return self.store.pop((phase, mode), {})  # return and delete


class MetricsLogger:
    """
    Robust, dead-lock-free metrics aggregator for multi–process training.
    Usage:
        logger.record(...);  # any rank, any time
        logger.flush(phase, mode)  # exactly once per optimization step
    """

    def __init__(
        self,
        accelerator_manager: AcceleratorManager,
        global_step: Callable[[], int],
    ) -> None:
        self.acc = accelerator_manager
        self._step_fn = global_step
        self._buf = _RingBuffer()

    # ---------- public helpers -------------------------------------------------
    def record(
        self, *, phase: str, mode: str, model: str, name: str, value: Any
    ) -> None:
        """
        Accept **any** tensor/number; convert to python float ASAP so that
        the object is pickle-able inside gather_object().
        """
        val: float
        if torch.is_tensor(value):
            val = float(value.mean().item())  # safety: scalarise
        elif isinstance(value, (int, float)):
            val = float(value)
        else:
            logger.debug(
                "Skipping non-numeric metric %s/%s – type=%s", model, name, type(value)
            )
            return

        if val != val:  # NaN check
            return
        self._buf.add(phase, mode, model, name, val, self._step_fn())

    # convenience wrappers so existing trainers compile unchanged
    store_metric = record

    def store_metrics(
        self, *, phase: str, mode: str, model: str, metrics: Dict[str, Any]
    ) -> None:
        for k, v in metrics.items():
            self.record(phase=phase, mode=mode, model=model, name=k, value=v)

    def store_entropy(
        self, *, phase: str, mode: str, model: str, per_token_entropy: torch.Tensor
    ) -> None:
        self.record(
            phase=phase,
            mode=mode,
            model=model,
            name="entropy_mean",
            value=per_token_entropy.mean(),
        )
        self.record(
            phase=phase,
            mode=mode,
            model=model,
            name="entropy_std",
            value=per_token_entropy.std(),
        )

    def get_latest_metric(
        self, mode: str, model: str, name: str, phase: str
    ) -> float | None:
        """
        Get the latest metric value for a given phase/mode/model/name combination.
        Returns None if no metric is found.
        """
        try:
            metrics_list = self._buf.store.get((phase, mode), {}).get(model, {}).get(name, [])
            if metrics_list:
                return metrics_list[-1].value  # Return the latest metric
            return None
        except Exception:
            return None

    # ---------- aggregation ----------------------------------------------------
    def flush(self, *, phase: str, mode: str) -> None:
        """
        - convert local buffer → [(key, values), ...]
        - gather once (every rank sees the full list)
        - rank-0 merges → mean/std and logs via Accelerator.log().
        """
        # shape local metrics as list of tuples to avoid gather_object recursion issues
        local_items: List[Tuple[str, List[float]]] = []
        grouped = self._buf.pop_phase_mode(phase, mode)
        for model, m_dict in grouped.items():
            for name, points in m_dict.items():
                key = f"{phase}/{mode}/{model}/{name}"
                values = [m.value for m in points]
                local_items.append((key, values))

        # guarantee *all* ranks call gather exactly once
        # gather_object with simple list of tuples - much more predictable
        all_items: List[Tuple[str, List[float]]] = gather_object(local_items)

        if not self.acc.get_state_property("is_main_process"):
            return  # non-main ranks are done

        # merge all gathered items
        merged: Dict[str, List[float]] = defaultdict(list)
        for key, values in all_items:
            merged[key].extend(values)

        if not merged:
            logger.info(
                "MetricsLogger: nothing to log for %s/%s at step %d",
                phase,
                mode,
                self._step_fn(),
            )
            return

        payload: Dict[str, float] = {}
        for k, vs in merged.items():
            payload[k] = statistics.mean(vs)
            if len(vs) > 1:
                payload[f"{k}_std"] = statistics.stdev(vs)

        self.acc.log(payload, step=self._step_fn())
