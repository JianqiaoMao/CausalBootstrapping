from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union, TYPE_CHECKING
import numpy as np

# ---- Core containers ----
DataDict  = Dict[str, np.ndarray]                    # Observational data: var -> (N, d) array
IntvValue = Union[float, Sequence[float], np.ndarray]
IntvDict  = Dict[str, IntvValue]                     # Intervention var -> value(s)

# ---- Function types ----
# All return a numpy array shaped (N,) or (N,1)
WeightFunc = Callable[..., np.ndarray]               # w_func(**kwargs) -> weights
DistFunc   = Callable[..., np.ndarray]               # pdf(**kwargs)    -> densities
KernelFunc = Callable[..., np.ndarray]               # kernel(**kwargs) -> kernel values

DistMap       = Dict[str, DistFunc]                  # e.g. "Y,U" -> pdf(Y,U)
CauseIntvMap  = Dict[str, str]                       # e.g. {"Y": "intv_Y"}

# ---- Factory for building a weight function from an ID expression ----
# Returns (weight_function, expression_like_object).
if TYPE_CHECKING:
    from causalbootstrapping.expr_extend import weightExpr, DOExpr
    from grapl.eqn import Eqn
    from grapl.admg import ADMG

WeightBuilder = Callable[
    [DistMap, int, Optional[KernelFunc], CauseIntvMap],
    Tuple[WeightFunc, "Expr"]
]

# External ID-expression from GRAPL (kept as Any to avoid tight coupling)
IdExpr = "Eqn"  # Eqn