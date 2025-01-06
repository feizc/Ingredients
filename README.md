<div align=center>
<img src="https://github.com/feizc/Ingredients/blob/main/asserts/logo.jpg?raw=true" width="300px">
</div>

<h2 align="center"> <a href="https://arxiv.org">Blending Custom Photos with Video Diffusion Transformers</a></h2>

[![arXiv](https://img.shields.io/badge/Arxiv-paper-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2501.01790)
[![Model](https://img.shields.io/badge/Huggingface-Model-yellow)](https://huggingface.co/feizhengcong/Ingredients)
[![Dataset](https://img.shields.io/badge/Huggingface-Dataset-blue)](https://huggingface.co/datasets/feizhengcong/Ingredients)




<div align="left">
This repository is the official implementation of Ingredients, a powerful way to customize video creations by incorporating multiple specific identity (ID) photos, with advanced video diffusion Transformers. 
This is a research project, and it is recommended to try advanced products: 
<a href="https://skyreels.ai/"><img src="https://img.shields.io/static/v1?label=Recommend&message=Application&color=orange&logo=demo"></a> &ensp;
</div>


## 📷 1. Gallery

<div align="center">
  <video src="https://github.com/user-attachments/assets/910220a4-6499-430d-8dde-bf86616eacab" width="70%"> </video>
</div>

## ⚙️ 2. Environments

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
    --seed 2025 \
    --img_file_path 'asserts/0.jpg' 'asserts/1.jpg'
```

We also include the evaluation metrics code at ```metric``` folder and evaluation data at [![Dataset](https://img.shields.io/badge/Huggingface-Dataset-blue)](https://huggingface.co/datasets/feizhengcong/Ingredients) for results comparison in multi-id customization tasks. 

Similar to [ConsisID](https://github.com/PKU-YuanGroup/ConsisID), Ingredients also has high requirements for prompt quality. 
We suggest referring to formation in the [link](https://github.com/PKU-YuanGroup/ConsisID?tab=readme-ov-file#prompt-refiner).


## ⏰ 4. Training

Coming soon, including multi-stage training scripts and multi-ID text-video datasets. 

You can prepare the video-text pair data as [formation](datasets.py) and our experiments can be repeated by simply run the training scripts as:

```bash
# For stage 1
bash train_face.sh
# For stage 2
bash train_router.sh
```



## 🚀 5. Cite

If you find this work useful for your research and applications, please cite us using this BibTeX:

```bibtex
@article{fei2025ingredients,
    title={Ingredients: Blending Custom Photos with Video Diffusion Transformers},
    author={Fei, Zhengcong and Li, Debang and Qiu, Di and Yu, Changqian and Fan, Mingyuan},
    journal={arXiv preprint arXiv:2501.01790},
    year={2025}
}
```
For any question, please feel free to open an issue. 


## Acknowledgement

This project wouldn't be possible without the following open-sourced repositories: [CogVideoX](https://github.com/THUDM/CogVideo), [ConsisID](https://github.com/PKU-YuanGroup/ConsisID), [Uniportrait](https://github.com/junjiehe96/UniPortrait), and [Hunyuan Video](https://github.com/Tencent/HunyuanVideo). 


