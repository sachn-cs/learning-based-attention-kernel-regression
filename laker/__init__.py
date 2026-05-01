"""LAKER: Learning-based Attention Kernel Regression."""

from laker.backend import get_default_device, set_default_device
from laker.benchmark import (
    BaselineBenchmark,
    SolverBenchmark,
    benchmark_laker_vs_baselines,
    benchmark_solver,
)
from laker.data import generate_grid, generate_radio_field
from laker.embeddings import PositionEmbedding
from laker.kernels import (
    AttentionKernelOperator,
    NystromAttentionKernelOperator,
    RandomFeatureAttentionKernelOperator,
    SKIAttentionKernelOperator,
    SparseKNNAttentionKernelOperator,
)
from laker.models import LAKERRegressor
from laker.preconditioner import CCCPPreconditioner
from laker.solvers import PreconditionedConjugateGradient
from laker.visualize import plot_convergence, plot_radio_map

__version__ = "0.3.0"

__all__ = [
    "get_default_device",
    "set_default_device",
    "BaselineBenchmark",
    "SolverBenchmark",
    "benchmark_laker_vs_baselines",
    "benchmark_solver",
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
