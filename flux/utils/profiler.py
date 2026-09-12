import time
import numpy as np
from flux.tensor.tensor import Tensor

class MemoryProfiler:
    def __init__(self):
        self.records = []

    def profile(self, model, x, label="forward"):
        """Run forward pass and record per-layer memory and time."""
        self.records = []
        current = x
        layers = getattr(model, "layers", [model])
        for i, layer in enumerate(layers):
            t0 = time.perf_counter()
            current = layer(current)
            t1 = time.perf_counter()
            mem_bytes = current.data.nbytes
            name = repr(layer) if hasattr(layer, "__repr__") else f"layer_{i}"
            self.records.append({
                "layer": name,
                "output_shape": current.shape,
                "output_mb": mem_bytes / 1024**2,
                "time_ms": (t1 - t0) * 1000,
            })
        return current

    def param_count(self, model):
        return sum(p.data.size for p in model.parameters())

    def report(self):
        total_mb = sum(r["output_mb"] for r in self.records)
        total_ms = sum(r["time_ms"] for r in self.records)
        print(f"{'Layer':<35} {'Shape':<20} {'MB':>6} {'ms':>8}")
        print("-" * 72)
        for r in self.records:
            shape_str = str(r["output_shape"])
            print(f"{r['layer']:<35} {shape_str:<20} {r['output_mb']:>6.3f} {r['time_ms']:>8.2f}")
        print("-" * 72)
        print(f"{'TOTAL':<35} {'':<20} {total_mb:>6.3f} {total_ms:>8.2f}")
