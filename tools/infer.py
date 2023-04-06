import os, sys
from pathlib import Path
import torch
from torchvision import transforms as T
import random

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from cfg.cfg import Config
from src.model import Generator
from src.plot import draw_plot




def infer(config: Config):
    device = config.device

    net_G = Generator(num_classes=config.total_classes, in_channels=config.input_dims).to(device)
    net_G.load_state_dict(torch.load(config.best_checkpoint))

    inputs = torch.randn(config.num_col * config.num_row, config.input_dims).to(device)
    fixed_label = torch.zeros([config.num_col * config.num_row, 10]).to(device)
    for i in range(10):
        fixed_label[10 * i:config.num_col + 10 * i, i] = 1.0
    # fixed_label[:, random.randint(0, 9)] = 1.0
    outputs = net_G(inputs, fixed_label) 

    draw_plot(outputs, config)

if __name__ == "__main__":
    config = Config(phase = 'valid')
    infer(config)