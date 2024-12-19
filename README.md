<div align=center>
<img src="https://github.com/feizc/CAT/blob/main/asserts/logo.jpg?raw=true" width="300px">
</div>

<h2 align="center"> <a href="https://arxiv.org">Blending Custom Photos with Video Diffusion Transformers</a></h2>

<div align="center">
This repository is the official implementation of Ingredients.
</div>

## 📷 1. Gallery


## ⚙️ 2. Requirements and Installation 

We recommend the requirements as follows.

### Environment

```bash
conda create -n cat python=3.11.0
conda activate cat
pip install -r requirements.txt
```

### Download CKPTs

The weights are available at 🤗HuggingFace.


## 🗝️ 3. Inference 


## 🐳 4. Training 

First, setting hyperparameters:

- environment (e.g., cuda): deepspeed_configs
- training arguments (e.g., batchsize): train_single_rank.sh or train_multi_rank.sh 

Then, we run the following bash to start training:

```bash
# For single rank
bash train_single_rank.sh
# For multi rank
bash train_multi_rank.sh
```



## 👍 5. Acknowledgement

* This project wouldn't be possible without the following open-sourced repositories: [CogVideoX](https://github.com/THUDM/CogVideo), [ConsisID](https://github.com/PKU-YuanGroup/ConsisID). 











