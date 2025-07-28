from hydra.core.config_store import ConfigStore

from trainer.models.icthp_model import ICTHPModelConfig

cs = ConfigStore.instance()
cs.store(group="model", name="icthp", node=ICTHPModelConfig)
