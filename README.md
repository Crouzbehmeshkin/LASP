# Look-Ahead Selective Plasticity for Continual Learning of Visual Tasks

Paper is under review. 
This code is based on [Co2L: Contrastive Continual Learning](https://github.com/chaht01/Co2L). 

# Reproducing Results
Our results were run on an Ubuntu 20.04.5 system, with two Nvidia RTX 3090 gpus (cuda version 11.2), and Python version 3.8. PyTorch version 1.12.1 with cuda 11.3 was used to implement main methods, while torchray version 1.0.0.2 was used to implement the extended excitation backprop. 

To set up the virtual environment, first install PyTorch:
```
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
Then proceed to install the requirements:
```
pip install -r requirements.txt
```

To measure accuracy for each continual learning setting and dataset,
two scripts need to be run. The first one is for the representation learning part and second one is for 
training a linear layer on frozen network (linear evaluation). For example, for training on SplitCIFAR10 using the Selective Distillation method:
```
python main.py --batch_size 512 --model resnet18 --dataset cifar10 --mem_size 200 --epochs 100 --start_epoch 500 --learning_rate 0.5 --temp 0.5 --current_temp 0.2 --past_temp 0.01  --cosine --syncBN --onlycurrent --emb_dim 128 --distill_type SD --selectiveness 1e-5 --distill_power 5 --trial 0;
python main_linear_buffer.py --dataset cifar10 --learning_rate 1 --target_task 4 --emb_dim 128 --ckpt ./save_SD_random_200_onlycurrent/cifar10_models/cifar10_SD_32_resnet18_lr_0.5_decay_0.0001_bsz_512_temp_0.5_trial_0_500_100_0.2_0.01_5.0_1e-05_cosine_warm/ --logpt ./save_SD_random_200_onlycurrent_cifar10/logs/cifar10_SD_32_resnet18_lr_0.5_decay_0.0001_bsz_512_temp_0.5_trial_0_500_100_0.2_0.01_5.0_1e-05_cosine_warm/;
```


# Issue
If you have trouble with NaN loss while training representation learning, you may find solutions from [SupCon issue page](https://github.com/HobbitLong/SupContrast/issues). Please check your training works perfectly on SupCon first. 
