"""Multi-factor disentanglement scorecard for the CELLxGENE atlas VQ-VAE.

The modules here are import-light on purpose: :mod:`atlas_eval.metrics` and
:mod:`atlas_eval.matched_biology` depend only on numpy/scipy/scikit-learn so
they can be unit tested away from the training server, while
:mod:`atlas_eval.representations` and :mod:`atlas_eval.scorecard` pull in torch
and the ``vq_2608`` pipeline.
"""

__all__ = ["metrics", "matched_biology", "representations", "scorecard", "adversary_monitor"]
