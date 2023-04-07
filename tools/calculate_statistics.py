import os, sys
from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from torch.utils.data import DataLoader

from src.dataloader import LoadDataset
from cfg.cfg import Config
from src.trainer import Trainer
from src.metrics.fid import *

if __name__ == "__main__":
    config = Config()

    inception_dims = 2048
    inception_model = init_inception_model(dims = inception_dims, device = config.device)
    train_data = LoadDataset(config)
    fid_train_loader = DataLoader(
        train_data,
        batch_size=config.train_batch_size // 2, 
        shuffle=False)
    
    mu_train, sigma_train = calculate_activation_statistics(fid_train_loader, inception_model, inception_dims, config.device)
    print(f"Trainset: mu = {mu_train} | sigma = {sigma_train}")

    np.save(f"{config.metrics}_metrics/mu.npy", mu_train)
    np.save(f"{config.metrics}_metrics/sigma.npy", sigma_train)
