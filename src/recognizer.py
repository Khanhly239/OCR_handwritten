"""Text recognition wrapper around VietOCR.

Supports both built-in backbones:

* ``vgg_transformer`` -- higher accuracy, slower; recommended for handwriting.
* ``vgg_seq2seq``    -- lighter and faster; good for realtime / CPU demos.

The predictor is loaded lazily on the first ``recognize`` call so that
importing this module does not require VietOCR weights to be downloaded.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple, Union

from PIL import Image


_SUPPORTED_MODELS = ("vgg_transformer", "vgg_seq2seq")


def _resolve_device(device: str) -> str:
    """Map ``'auto'`` to ``'cuda:0'`` if available, else ``'cpu'``."""
    if device != "auto":
        return device
    try:
        import torch

        return "cuda:0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


class TextRecognizer:
    """Run a VietOCR model on cropped text-line images.

    Parameters
    ----------
    model : str
        VietOCR backbone name (``vgg_transformer`` or ``vgg_seq2seq``).
    device : str
        ``'auto'``, ``'cpu'`` or a CUDA device string like ``'cuda:0'``.
    beamsearch : bool
        Enable beam-search decoding (slower, marginally better).
    weights_path : str or Path, optional
        Path to a fine-tuned ``.pth`` checkpoint. If omitted, VietOCR's
        default pretrained weights for ``model`` are downloaded.
    extra_config : dict, optional
        Extra entries merged into the VietOCR config before predictor init.
        Useful for overriding ``vocab`` or ``dataset.image_max_width``.
    """

    def __init__(
        self,
        model: str = "vgg_transformer",
        *,
        device: str = "auto",
        beamsearch: bool = False,
        weights_path: "str | None" = None,
        extra_config: "dict | None" = None,
    ) -> None:
        if model not in _SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported VietOCR model {model!r}. "
                f"Choose one of {_SUPPORTED_MODELS}."
            )
        self.model_name = model
        self.device = _resolve_device(device)
        self.beamsearch = beamsearch
        self.weights_path = str(weights_path) if weights_path else None
        self.extra_config = dict(extra_config or {})
        self._predictor = None  # lazy

    def _load(self) -> None:
        if self._predictor is not None:
            return
        from vietocr.tool.config import Cfg
        from vietocr.tool.predictor import Predictor

        config = Cfg.load_config_from_name(self.model_name)
        config["device"] = self.device
        config.setdefault("predictor", {})
        config["predictor"]["beamsearch"] = self.beamsearch
        # Use pretrained CNN weights bundled with VietOCR.
        if "cnn" in config and isinstance(config["cnn"], dict):
            config["cnn"]["pretrained"] = True

        if self.weights_path:
            from pathlib import Path

            wp = Path(self.weights_path)
            if not wp.is_file():
                raise FileNotFoundError(
                    f"VietOCR weights file not found: {wp}. "
                    "Run `python -m src.train.train_vietocr ...` first."
                )
            # Tell VietOCR to load from local file instead of downloading.
            config["weights"] = str(wp)
            config["pretrain"] = str(wp)  # some VietOCR versions read this key.

        for k, v in self.extra_config.items():
            if isinstance(v, dict) and isinstance(config.get(k), dict):
                config[k].update(v)
            else:
                config[k] = v

        self._predictor = Predictor(config)

    def recognize(self, image: Image.Image) -> Tuple[str, float]:
        """Recognize text on a single PIL image.

        Returns
        -------
        (text, confidence) : tuple[str, float]
            Confidence is VietOCR's posterior probability for the predicted
            sequence (``0.0`` if the model does not return it).
        """
        self._load()
        out = self._predictor.predict(image, return_prob=True)
        if isinstance(out, tuple) and len(out) == 2:
            text, prob = out
            try:
                prob_f = float(prob)
            except (TypeError, ValueError):
                prob_f = 0.0
            return str(text), prob_f
        return str(out), 0.0

    def recognize_batch(
        self, images: Sequence[Image.Image]
    ) -> List[Tuple[str, float]]:
        """Recognize a batch of PIL images.

        Falls back to a per-image loop if VietOCR's batch API is unavailable
        in the installed version.
        """
        if not images:
            return []
        self._load()

        predict_batch = getattr(self._predictor, "predict_batch", None)
        if callable(predict_batch):
            try:
                out = predict_batch(list(images), return_prob=True)
            except TypeError:
                # Older VietOCR versions do not accept return_prob.
                texts = predict_batch(list(images))
                return [(str(t), 0.0) for t in texts]
            if isinstance(out, tuple) and len(out) == 2:
                texts, probs = out
                return [
                    (str(t), float(p) if p is not None else 0.0)
                    for t, p in zip(texts, probs)
                ]
            return [(str(t), 0.0) for t in out]

        return [self.recognize(img) for img in images]

    @property
    def is_loaded(self) -> bool:
        return self._predictor is not None
