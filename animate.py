import hydra
from omegaconf import DictConfig

import colat.runner as runner
import colat.utils.log_utils as utils


@hydra.main(config_path="conf", config_name="animate")
def animate(cfg: DictConfig):
    utils.display_config(cfg)
    runner.animate(cfg)


if __name__ == "__main__":
    animate()
