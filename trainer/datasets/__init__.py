from hydra.core.config_store import ConfigStore

from trainer.datasets.icthp_dataset import ICTHPDatasetConfig



cs = ConfigStore.instance()
cs.store(group="dataset", name="icthp", node=ICTHPDatasetConfig)

