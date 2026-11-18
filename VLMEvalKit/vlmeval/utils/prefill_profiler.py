
import functools
import torch

class NEOPrefillProfiler:
    """
    CUDA-event profiler for NEO.

    Total prefill:
        Vision encoder + compression
        + input assembly
        + first LLM forward

    Excludes decoding, image preprocessing, and tokenization.
    """

    def __init__(self, model):
        self.model = model
        self.handles = []
        self.enabled = False
        self.original_extract = model.extract_feature_with_positions
        self.reset()

    def reset(self):
        self.start = None
        self.end = None
        self.end_recorded = False

    def _extract(self, *args, **kwargs):
        if self.enabled and self.start is None:
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()

        return self.original_extract(*args, **kwargs)

    def _llm_post(self, module, args, output):
        if (
            self.enabled
            and self.start is not None
            and not self.end_recorded
        ):
            self.end.record()
            self.end_recorded = True

    def install(self):
        self.model.extract_feature_with_positions = (
            functools.wraps(self.original_extract)(self._extract)
        )

        self.handles.append(
            self.model.language_model.register_forward_hook(
                self._llm_post
            )
        )

    def start_profiling(self):
        self.reset()
        self.enabled = True

    def stop_profiling(self):
        self.enabled = False

        if not self.end_recorded:
            return None

        self.end.synchronize()
        return self.start.elapsed_time(self.end)

    def close(self):
        self.enabled = False

        for handle in self.handles:
            handle.remove()

        self.handles.clear()

        self.model.extract_feature_with_positions = (
            self.original_extract
        )
