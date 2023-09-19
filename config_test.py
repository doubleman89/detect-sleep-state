from configs import config
from utils import config as util_config
from utils.config import Paths
from pathlib import WindowsPath
if __name__ =="__main__":
    cfg =util_config.Config.from_json(config.CFG)
    assert type(cfg.paths) == Paths 
    assert type(cfg.paths.filepath) == WindowsPath , type(cfg.paths.filepath)