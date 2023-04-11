# Conditional Fashion MNIST Image Generation with Conditional GAN (cGAN)

Implement **Conditional GAN (cGAN)** to generate Fashion MNIST images of specific class

![asset](./assets/sample.jpg "asset")

## Environments

- Python 3.8.0

Install environments with

``` bash
bash scripts/setup_environments
```

## Data

You can get dataset from [here](https://github.com/DeepLenin/fashion-mnist_png) or run these scripts

``` bash
bash scripts/download_data.sh
```

Data follow this structure

```
data
├── test
│   ├── 0
│   ├── 1
│   ├── ...
└── train
    ├── 0
    ├── 1
    ├── ...
```

## Checkpoint

You can download pretrained weight for [Net_G](https://drive.google.com/file/d/148jG1AiENfAr6tc3CCtjXTGfhmOfWXXp/view?usp=share_link), train with default config. But I recommend you to retrain from scratch with your own experiments

Here is pretrained for [Net_D](https://drive.google.com/file/d/1kkS-ZoaGy9feiGQPXD2W7qrochDIpGHX/view?usp=share_link)

## Train

Firstly, compute statistics for train dataset (mean and variance) with 

```bash 
python tools/calculate_statistics.py
```

You can modify config in `cfg/cfg.py` and run

``` bash
python tools/train.py
```

## Infer

After training, you get weight for model in `training_runs/exp...`, set weight in `best_checkpoint` of config and run

``` bash
python tools/infer.py
```

## Result

After run, FID minimum score is **56.36** at epoch 409

- Example output class 0

![asset0](./assets/sample_0.jpg "asset0")

- Example output class 1

![asset0](./assets/sample_1.jpg "asset0")

## Citation

``` bibtex
@article{mirza2014conditional,
  added-at = {2022-01-13T16:13:14.000+0100},
  author = {Mirza, Mehdi and Osindero, Simon},
  biburl = {https://www.bibsonomy.org/bibtex/2cfa802085dbe67e0697a9acfffa30a3f/mo0000},
  description = {Erklärt cGAN Teil von Seminararbeitsthema cWGAN.},
  interhash = {efbbaeaebb1ea8d88264d258624d364c},
  intrahash = {cfa802085dbe67e0697a9acfffa30a3f},
  journal = {arXiv preprint arXiv:1411.1784},
  keywords = {cGAN final gan thema:conditional_wasserstein_gan-based_oversampling_of_tabular_data_for_imbalanced_learning},
  timestamp = {2022-01-13T16:13:14.000+0100},
  title = {Conditional generative adversarial nets},
  year = 2014
}

@misc{xiao2017fashionmnist,
  abstract = {We present Fashion-MNIST, a new dataset comprising of 28x28 grayscale images of 70,000 fashion products from 10 categories, with 7,000 images per category. The training set has 60,000 images and the test set has 10,000 images. Fashion-MNIST is intended to serve as a direct drop-in replacement for the original MNIST dataset for benchmarking machine learning algorithms, as it shares the same image size, data format and the structure of training and testing splits. The dataset is freely available at https://github.com/zalandoresearch/fashion-mnist},
  added-at = {2021-10-12T06:50:19.000+0200},
  author = {Xiao, Han and Rasul, Kashif and Vollgraf, Roland},
  biburl = {https://www.bibsonomy.org/bibtex/2de51af2f6c7d8b0f4cd84a428bb17967/andolab},
  description = {Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms},
  interhash = {0c81f9a6170118f14703b6796101ce40},
  intrahash = {de51af2f6c7d8b0f4cd84a428bb17967},
  keywords = {Fashion-MNIST Image_Classification_Benchmark},
  note = {cite arxiv:1708.07747Comment: Dataset is freely available at  https://github.com/zalandoresearch/fashion-mnist Benchmark is available at  http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/},
  timestamp = {2023-01-31T20:34:07.000+0100},
  title = {Fashion-MNIST: a Novel Image Dataset for Benchmarking Machine Learning Algorithms},
  url = {http://arxiv.org/abs/1708.07747},
  year = 2017
}



@misc{Seitzer2020FID,
  author={Maximilian Seitzer},
  title={{pytorch-fid: FID Score for PyTorch}},
  month={August},
  year={2020},
  note={Version 0.3.0},
  howpublished={\url{https://github.com/mseitzer/pytorch-fid}},
}
```

## Reference

- [Conditional GAN (cGAN)](https://nttuan8.com/bai-3-conditional-gan-cgan/)
- [Fashion MNIST Dataset in PNG](https://github.com/DeepLenin/fashion-mnist_png)
- [FID score in pytorch](https://github.com/mseitzer/pytorch-fid)