from .fid_score import *
from .inception import *
from .pred_loader import *


def init_inception_model(dims = 2048, device = "cpu"):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    model.eval()
    return model
