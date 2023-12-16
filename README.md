<h1 align="center"> Image Restoration Through Generalized Ornstein-Uhlenbeck Bridge </h1>
<div align="center"> 
Conghan Yue<sup>*</sup>, Zhengwei Peng, Junlong Ma, Shiyan Du, Pengxu Wei, Dongyu Zhang

<i>Department of Computer Science, Sun Yat-sen University</i> <br>
<sup>*</sup>yuech5@mail2.sysu.edu.cn
</div>
<h3 align="center"> [<a href="https://arxiv.org/abs/2302.05872">arXiv</a>]</h3>

Official PyTorch Implementations of GOUB, a diffusion bridge model that applies the Doob's *h*-transform to the generalized Ornstein-Uhlenbeck process. This model can address general image restoration tasks without the need for specific prior knowledge.

# Overview
<div align="center">
    <img src="figs/framework.png" alt="Framework">
</div>

# Visual Results
<div align="center">
    <img src="figs/ir.png" alt="Framework" width="60%"><br>
</div>

# Intallation
This code is developed with Python3, and we recommend python>=3.8 and PyTorch ==1.13.0. Install the dependencies with Anaconda and activate the environment with:

    conda env create --name GOUB python=3.8
    conda activate GOUB
    pip install -r requrements.txt

# Test
1. Prepare datasets.
2. Download pretrained checkpoints [here](https://www.baidu.com).
3. Modify options, including dataroot_GT, dataroot_LQ and pretrain_model_G.
4. `python test.py -opt=options/test.yml`

The Test results will be saved in `\results`.

# Train
1. Prepare datasets.
2. Modify options, including dataroot_GT, dataroot_LQ.
3. `python train.py -opt=options/train.yml`

The Training log will be saved in `\experiments`.

# Important Option Details
* `dataroot_GT`: Ground Truth (High-Quality) data path.
* `dataroot_LQ`: Low-Quality data path.
* `pretrain_model_G`: Pretraind model path.
* `GT_size, LQ_size`: Size of the data cropped during training.
* `niter`: Total training iterations.
* `val_freq`: Frequency of validation during training.
* `save_checkpoint_freq`: Frequency of saving checkpoint during training.