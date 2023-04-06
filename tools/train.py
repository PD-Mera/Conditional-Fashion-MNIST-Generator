import os, sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from cfg.cfg import Config
from src.trainer import Trainer


if __name__ == "__main__":
    config = Config()
    trainer = Trainer(config)
    trainer.train()


            


