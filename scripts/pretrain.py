import hydra
from omegaconf import DictConfig

from amplify import trainer, trainer_ally


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def pipeline(cfg: DictConfig):
    if cfg.strategy._name_ == 'ally':
        trainer_ally(cfg)
    else:
        trainer(cfg)


if __name__ == "__main__":
    pipeline()
