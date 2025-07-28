
from hydra.core.config_store import ConfigStore

from trainer.tasks.icthp_task import ICTHPTaskConfig
from trainer.tasks.ict_task import ICTTaskConfig

cs = ConfigStore.instance()
cs.store(group="task", name="icthp", node=ICTHPTaskConfig)
cs.store(group="task", name="ict", node=ICTTaskConfig)
