"""LAKER: Learning-based Attention Kernel Regression."""

from laker.backend import get_default_device, set_default_device
from laker.benchmark import benchmark_laker_vs_baselines
from laker.data import generate_grid, generate_radio_field
from laker.embeddings import PositionEmbedding
from laker.kernels import AttentionKernelOperator
from laker.low_rank_kernels import (
    NystromAttentionKernelOperator,
    RandomFeatureAttentionKernelOperator,
)
from laker.models import LAKERRegressor
from laker.preconditioner import CCCPPreconditioner
from laker.ski_kernels import SKIAttentionKernelOperator
from laker.solvers import PreconditionedConjugateGradient
from laker.sparse_kernels import SparseKNNAttentionKernelOperator
from laker.visualize import plot_convergence, plot_radio_map

__version__ = "0.2.0"

__all__ = [
    "get_default_device",
    "set_default_device",
    "benchmark_laker_vs_baselines",
    "generate_grid",
    "generate_radio_field",
    "PositionEmbedding",
    "AttentionKernelOperator",
    "NystromAttentionKernelOperator",
    "RandomFeatureAttentionKernelOperator",
    "SparseKNNAttentionKernelOperator",
    "SKIAttentionKernelOperator",
    "CCCPPreconditioner",
    "PreconditionedConjugateGradient",
    "LAKERRegressor",
    "plot_convergence",
    "plot_radio_map",
]
