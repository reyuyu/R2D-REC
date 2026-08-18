"""DSR-GRPO ablation package.

The production GR_REC_v1 implementation lives outside this package and is
imported without modification.
"""

from .dsr_parser import parse_interest_section
from .dsr_trainer import DsrGRPOTrainer

__all__ = ["DsrGRPOTrainer", "parse_interest_section"]
