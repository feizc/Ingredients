<div align=center>
<img src="https://github.com/feizc/Ingredients/blob/main/asserts/logo.jpg?raw=true" width="300px">
</div>

<h2 align="center"> <a href="https://arxiv.org">Blending Custom Photos with Video Diffusion Transformers</a></h2>

<div align="left">
This repository is the official implementation of Ingredients, a powerful way to customize video creations by incorporating multiple specific identity (ID) photos, with advanced video diffusion Transformers. 
This is a research project, and it is recommended to try advanced products: 
<a href="https://skyreels.ai/"><img src="https://img.shields.io/static/v1?label=Recommend&message=Application&color=orange&logo=demo"></a> &ensp;
</div>

## 📷 1. Gallery

<div align="center">
  <video src="https://github.com/user-attachments/assets/910220a4-6499-430d-8dde-bf86616eacab" width="70%"> </video>
</div>

## ⚙️ 2. Requirements and Installation 

We recommend the requirements as follows. 

```bash
conda create -n ingredients python=3.11.0
conda activate ingredients
pip install -r requirements.txt
```

The weights of model are available at 🤗HuggingFace.

## 🗝️ 3. Inference 
We provide the inference scripts ```inference.py``` for simple testing. Run the command as examples: 

```bash
python infer.py \
    --prompt "Two men in half bodies, are seated in a dimly lit room, possibly an office or meeting room, with a formal atmosphere." \
    --model_path "\path\to\model" \
    --seed 42 \
    --img_file_path 'asserts/0.jpg' 'asserts/1.jpg'
```

We also include the evaluation metrics and evaluation data for results comparison in multi-id customization tasks. 


## ⏰ 4. Training

Coming soon, including multi-stage training scripts and datasets. 


## 👍 5. Acknowledgement

* This project wouldn't be possible without the following open-sourced repositories: [CogVideoX](https://github.com/THUDM/CogVideo), [ConsisID](https://github.com/PKU-YuanGroup/ConsisID), and [Hunyuan Video](https://github.com/Tencent/HunyuanVideo). 


