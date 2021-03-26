---

<div align="center">    
 
# MEMO

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)

<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->
![CI testing](https://github.com/PyTorchLightning/deep-learning-project-template/workflows/CI%20testing/badge.svg?branch=master&event=push)


<!--  
Conference   
-->   
</div>
 
## Description   
Multiple Experts, Multiple Objectives

## How to run   
First, install dependencies   
```bash
# clone memo   
git clone https://github.com/feloundo/memo

# install memo   
cd memo
pip install -e .   
pip install -r requirements.txt
 ```   
 Next, navigate to any file and run it.   
 ```bash
# module folder
cd memo

# run module (example: mnist as your main contribution)   
python run_memo.py   
```

## Imports
This project is setup as a package which means you can now easily import any file into any other file like so:

```python
from memo.datasets.mnist import mnist
from memo.lit_classifier_main import LitClassifier
from pytorch_lightning import Trainer

# model
model = LitClassifier()


```

### Citation   
```
@article{Eloundou, Florentine,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={2021}
}
```   
