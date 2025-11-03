"""
pgg_reflective_demo
公共物品博弈 + 内生制度演化：最小可运行 Demo
"""

from .pgg_model import PGGModel
from .pgg_agent import PGGAgent, Institution
from .metrics import gini, compliance_rate, social_welfare

__version__ = "0.1.0"

__all__ = [
    'PGGModel',
    'PGGAgent',
    'Institution',
    'gini',
    'compliance_rate',
    'social_welfare',
]

