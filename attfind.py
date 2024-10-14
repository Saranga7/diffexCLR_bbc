import hydra
from omegaconf import DictConfig

import colat.runner as runner
import colat.utils.log_utils as utils


@hydra.main(config_path="conf", config_name="attfind")
def att_find(cfg: DictConfig):
    utils.display_config(cfg)
    runner.att_find(cfg)


if __name__ == "__main__":
    att_find()
