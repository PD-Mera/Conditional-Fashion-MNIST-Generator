import os
from matplotlib import pyplot as plt
import torch
from torchvision import transforms as T

from cfg.cfg import Config



def postprocess_tensor(inputs: torch.Tensor, config: Config):
    mean = [-x / y for x, y in zip(config.mean, config.std)]
    std = [1.0 / y for y in config.std]
    inv_normalize = T.Normalize(
        mean = mean,
        std = std,
    )
    inputs = inv_normalize(inputs)
    # inputs = inputs * torch.Tensor(config.std).to(config.device) + torch.Tensor(config.mean).to(config.device)
    return inputs

def draw_plot(inputs: torch.Tensor, config: Config, **kwarg):
    # inputs is torch tensor with BS x C x W x H
    draw_part = postprocess_tensor(inputs[:config.num_col * config.num_row], config)
    list_img_pil = [T.ToPILImage()(draw_part[i]) for i in range(draw_part.size(0))]
    plt.figure(figsize=(config.num_col, config.num_row))
    # plt.axis(False)
    for idx, img in enumerate(list_img_pil):
        plt.subplot(config.num_row,config.num_col, idx+1)
        plt.axis(False)
        plt.imshow(img, cmap="Greys")
        plt.subplots_adjust(wspace=0, hspace=0)
        plt.tight_layout()

    savefig_name = config.savefig_name
    for key in kwarg.keys():
        savefig_name += f"_{kwarg[key]}"

    savefig_name += ".jpg"

    if config.phase == "train":
        plt.savefig(os.path.join(config.exp_run_folder, savefig_name))
    else:
        os.makedirs(config.output_dir, exist_ok=True)
        plt.savefig(os.path.join(config.output_dir, savefig_name))

    plt.close()
    

