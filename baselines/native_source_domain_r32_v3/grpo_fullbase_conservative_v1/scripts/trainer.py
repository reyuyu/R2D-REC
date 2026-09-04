"""Evidence-only subclass composition around the frozen legacy GRPO trainer."""
from evidence import EvidenceRecGRPOTrainerMixin
from grpo_trl_trainer import RecGRPOTrainer


class FullBaseConservativeTrainer(EvidenceRecGRPOTrainerMixin, RecGRPOTrainer):
    pass
