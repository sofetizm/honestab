"""honestab — A/B test analysis with methods verified by simulation."""
from . import bayes, checks, datasets, design, methods, multiple, sequential
from .checks import srm
from .design import duration_days, mde, sample_size, sample_size_proportion
from .experiment import Metric, Report, analyze
from .methods import bootstrap, cuped, proportions, ratio, welch
from .multiple import benjamini_hochberg, holm
from .result import EffectResult
from .sequential import msprt

__version__ = "0.1.0"

__all__ = [
    "EffectResult", "Metric", "Report", "analyze", "bayes", "benjamini_hochberg", "bootstrap",
    "checks", "cuped", "datasets", "design", "duration_days", "holm", "mde", "methods", "msprt",
    "multiple", "proportions", "ratio", "sample_size", "sample_size_proportion", "sequential",
    "srm", "welch",
]
