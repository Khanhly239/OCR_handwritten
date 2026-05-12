"""Phase 1.5 -- fine-tune VietOCR on a Vietnamese handwriting dataset.

Modules:

- :mod:`src.train.prepare_data` -- split train annotations, sanity-check vocab.
- :mod:`src.train.train_vietocr` -- driver around :class:`vietocr.model.trainer.Trainer`.
- :mod:`src.train.evaluate` -- compute CER / WER / exact match on the test set.
"""
