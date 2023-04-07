import os, shutil
import torch

class Config:
    def __init__(self, phase = "train"):
        super(Config, self).__init__()
        self.data_path = [
            "./data/train/",
            "./data/test/",
        ]
        self.total_classes = 10
        
        self.phase = phase
        self.input_dims = 128
        self.best_checkpoint = ''
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        ### Image ###
        self.image_size = (28, 28)
        self.channel = 1
        self.mean = [0.5] * self.channel
        self.std = [0.5] * self.channel

        ### Train config ###
        self.EPOCH = 1000
        self.train_batch_size = 512
        self.train_num_worker = 8
        self.learning_rate_G = 0.0002
        self.learning_rate_D = 0.0002
        self.pretrained_G_weight = ''
        self.pretrained_D_weight = ''
        self.run_folder = "training_runs"
        self.exp_name = "exp"
        self.exp_number = self.get_exp_number()
        self.exp_run_folder = os.path.join(self.run_folder, self.exp_name + str(self.exp_number))
        if self.phase == 'train':
            os.makedirs(self.exp_run_folder)
            shutil.copy(os.path.abspath(__file__), self.exp_run_folder)
        self.model_G_savepath = "model_G_best.pth"
        self.model_D_savepath = "model_D_best.pth"
        self.training_log_savepath = "log.csv"

        ### Hyperparameter ###
        self.seed = 42
        self.flip_label = 0.0
        
        ### Metrics ###
        self.metrics = "FID" # None or FID
        if self.metrics is not None:
            os.makedirs(f"{self.metrics}_metrics", exist_ok=True)
        self.reload_metrics = False

        ### Test config ###
        self.output_dir = "outputs"
        self.num_col = 10
        self.num_row = 10
        self.savefig_name = "sample"

    def get_exp_number(self):
        os.makedirs(self.run_folder, exist_ok=True)
        exp_number = 0
        for folder in os.listdir(self.run_folder):
            try:
                if int(folder.replace(self.exp_name, "")) >= exp_number:
                    exp_number = int(folder.replace(self.exp_name, "")) + 1
            except:
                continue

        return exp_number
    