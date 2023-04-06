import os, random

import torch
from torch import nn
from torch.utils.data import DataLoader

from cfg.cfg import Config
from src.dataloader import LoadDataset
from src.model import Generator, Discriminator
from src.plot import draw_plot

class Trainer:
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.seed_everything()

        train_data = LoadDataset(config)
        self.train_loader = DataLoader(
            train_data, 
            batch_size=config.train_batch_size, 
            shuffle=True,
            num_workers=config.train_num_worker
        )

        self.device = self.config.device

        self.net_G = Generator(num_classes=config.total_classes, in_channels=config.input_dims).to(self.device)
        self.net_D = Discriminator(num_classes=config.total_classes).to(self.device)
        
        self.net_G = self.weights_init(self.net_G, self.config.pretrained_G_weight)
        self.net_D = self.weights_init(self.net_D, self.config.pretrained_D_weight)

        self.optim_G = torch.optim.Adam(params=self.net_G.parameters(), lr = config.learning_rate_G, betas = (0.5, 0.999))
        self.optim_D = torch.optim.Adam(params=self.net_D.parameters(), lr = config.learning_rate_D, betas = (0.5, 0.999))

        self.criterion = nn.BCELoss()

        self.fixed_noise = torch.randn([config.num_col * config.num_row, config.input_dims]).to(self.device)
        

    def seed_everything(self):
        torch.manual_seed(self.config.seed)
        random.seed(self.config.seed)

    def train_D(self, real_image: torch.Tensor, label: torch.Tensor, gauss_input: torch.Tensor):
        self.net_D.zero_grad()
        flip_label = random.random() < self.config.flip_label

        # Feed real images
        D_x = self.net_D(real_image, label)

        if not flip_label:
            real_label = torch.ones_like(D_x).to(self.device) * 0.9
        else:
            real_label = torch.zeros_like(D_x).to(self.device)

        loss_D_real = self.criterion(D_x, real_label)
        loss_D_real.backward()

        # Feed fake images
        G_z = self.net_G(gauss_input, label)
        D_G_z = self.net_D(G_z, label)
        
        if not flip_label:
            fake_label = torch.zeros_like(D_G_z).to(self.device)
        else:
            fake_label = torch.ones_like(D_G_z).to(self.device)

        loss_D_fake = self.criterion(D_G_z, fake_label)
        loss_D_fake.backward()

        loss_D = loss_D_real + loss_D_fake

        self.optim_D.step()

        return loss_D

    def train_G(self, real_image: torch.Tensor, label: torch.Tensor, gauss_input: torch.Tensor):
        self.net_G.zero_grad()

        # D_x = self.net_D(real_image, label)
        
        G_z = self.net_G(gauss_input, label)
        D_G_z = self.net_D(G_z, label)
        
        loss_G = self.criterion(D_G_z, torch.ones_like(D_G_z).to(self.device))

        loss_G.backward()
        self.optim_G.step()
        return loss_G

    def train_one_epoch(self):
        self.net_D.train()
        self.net_G.train()
        loss_D = 0.0
        loss_G = 0.0
        for idx, (image, label) in enumerate(self.train_loader):
            image, label = image.to(self.device), label.to(self.device)
            gauss = torch.randn(image.size(0), self.config.input_dims).to(self.device)
            
            # Train D
            loss_D += self.train_D(image, label, gauss)

            # train G
            loss_G += self.train_G(image, label, gauss)
             
        return loss_D / idx, loss_G / idx

    def save_model(self):
        torch.save(self.net_G.state_dict(), os.path.join(self.config.exp_run_folder, self.config.model_G_savepath))
        torch.save(self.net_D.state_dict(), os.path.join(self.config.exp_run_folder, self.config.model_D_savepath))

    def weights_init_(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def weights_init(self, net: nn.Module, pretrained: str):
        if pretrained != '':
            net.load_state_dict(torch.load(pretrained))
        else:
            net.apply(self.weights_init_)
        return net

    def train(self):
        for epoch in range(self.config.EPOCH):
            loss_D, loss_G = self.train_one_epoch()
            print(f"Epoch {epoch}: Loss D: {loss_D} | Loss G: {loss_G}")
            self.save_model()
            self.net_G.eval()

            self.fixed_label = torch.zeros([self.config.num_col * self.config.num_row, 10]).to(self.device)
            self.fixed_label[:, random.randint(0, 9)] = 1.0
            outputs = self.net_G(self.fixed_noise, self.fixed_label)
            if epoch % 10 == 0:
                draw_plot(outputs, self.config, epoch=epoch)
            
            draw_plot(outputs, self.config, name = "last")


        
        

        
            
        
