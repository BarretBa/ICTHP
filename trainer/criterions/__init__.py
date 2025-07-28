from hydra.core.config_store import ConfigStore

from trainer.criterions.icthp_criterion import ICTHPCriterionConfig

cs = ConfigStore.instance()
cs.store(group="criterion", name="icthp", node=ICTHPCriterionConfig) 

