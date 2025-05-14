# TiSpell: A Semi-Masked Methodology for Tibetan Spelling Correction

**📄 Paper**: _TiSpell: A Semi-Masked Methodology for Tibetan Spelling Correction covering Multi-Level Error with Data Augmentation_

**🧑‍💻 Author**: Yutong Liu, Xiao Feng, Ziyue Zhang, Yongbin Yu*, Cheng Huang, Fan Gao, Xiangxiang Wang*, Ban Ma-bao, Manping Fan, Thupten Tsering, Gadeng Luosang, Renzeng Duojie, Nyima Tashi

**📦 Repository**: Official implementation of TiSpell.

---

## 🧠 Overview

**TiSpell** is a Tibetan spelling correction algorithm specifically designed for multi-level orthographic errors. It proposes a **semi-masked methodology** that jointly models character-level, syllable-level, and word-level errors. With integrated data augmentation strategies, TiSpell improves robustness and accuracy in real-world spelling correction tasks. It leverages pre-trained language models and introduces an end-to-end correction architecture.

---

## ✨ Features

- ✅ Handles character-, syllable-, and word-level spelling errors
- ✅ Combines semi-masked modeling with structural reconstruction
- ✅ Includes multiple data augmentation techniques (perturbation, phonetic substitution, etc.)
- ✅ Fully compatible with Huggingface Transformers for easy integration and customization

---

## 🗂️ Project Structure
```
TiSpell/
├── dataloader/ # Data loading utilities
├── dataset/ # Preprocessed and raw datasets
├── images/ # Visualizations
├── model/ # Model architecture
├── pretrained_models/ # Checkpoints and pre-trained weights
├── scripts/ # Training and evaluation scripts
├── LICENSE
├── README.md
├── compute_parameter.py # Parameter counting utility
├── data_analysis.py # Exploratory data analysis
├── infer.py # Inference script
├── metrics.py # Evaluation metrics
├── option.py # Argument parsing
├── plot.py # Visualization utilities
├── train.py # Training script
└── requirements.txt # Python dependencies
```

---

## 🚀 Quick Start

### 🔧 1. Install Dependencies

```bash
pip install -r requirements.txt
```
### 📁 2. Prepare Dataset
Download the Tibetan News Classification dataset from (Huggingface)[https://huggingface.co/datasets/UTibetNLP/tibetan_news_classification] and place it under the dataset/ directory. Ensure that the dataset is formatted in the following structure:
```
TiSpell/
└── dataset/
    └── tibetan_news_classification/
        ├── 政务类
        ├── 教育类
        ├── 文化类
        ├── 旅游类
        ├── 时政类
        ├── 民生类
        ├── 法律类
        ├── 科技类
        ├── 经济类
        └── 艺术类
            ├── 0.txt
            ├── 1.txt
            ├── 2.txt
            └── ...
```


### 🏋️‍♂️ 3. Train the Model
```
python main.py
```

## ⚙️ Configuration
You can customize training and evaluation parameters in option.py, including:
+ Learning rate / Batch size
+ Training epochs
+ Weight decay



## 📌 Citation
If you find TiSpell helpful in your research, please cite our work:
```
@misc{liu2025tispellsemimaskedmethodologytibetan,
      title={TiSpell: A Semi-Masked Methodology for Tibetan Spelling Correction covering Multi-Level Error with Data Augmentation}, 
      author={Yutong Liu and Feng Xiao and Ziyue Zhang and Yongbin Yu and Cheng Huang and Fan Gao and Xiangxiang Wang and Ma-bao Ban and Manping Fan and Thupten Tsering and Cheng Huang and Gadeng Luosang and Renzeng Duojie and Nyima Tashi},
      year={2025},
      eprint={2505.08037},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.08037}, 
}
```

## 📝 License
This project is licensed under the MIT License. See the LICENSE file for more details.
