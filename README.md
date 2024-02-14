# TGEditor
Q: Does it need GPU, would CPU work, and how many?
It does not necessarily GPU. CPU will fully work, and GPU will expedite part of the program process.

Q: Is it in Python or Pytorch, what packages does it need?
It is in pytorch. Required packages include: numpy, pytorch, scipy, pandas, matplotlib, concurrent, yaml.

## Environment Setup
You can create a conda environment to easily run the code. For example, we can create a virtual environment named `tgeditor`:
```
conda create -n tgeditor python=3.8 -y
conda activate tgeditor
conda install --file requirements.txt
pip3 install config
```
If the command `conda install --file requirements.txt` fails in the previous step due to some package not found error, you could manually install all the required packages using the following commands:
```
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
conda install pyyaml matplotlib scipy pandas
pip install config
```

The next thing is to set up the necessary Jupyter lab installation
```
conda install ipykernel
ipython kernel install --user --name=tgeditor
```


The source code of the models are available in Python. We also provide an easy-to-use interactive Jupyter notebook file to guide you through each step, the data loading, training a model and testing the model, etc.
You can open a Jupyter notebook using
```
jupyter-lab
```
A browser tab should be automatically pop up. If not, click the link displayed in the terminal. Once the Jupyter is open, navigate to the project folder, and double click the `train.ipynb` file to open the Jupyter notebook. Once open, it will prompt you to select a kernel. Choose the `tgeditor` kernel we just created. If you accidentally close this prompt, you could also do so by clicking the  `No kernel` text button right next to a circle.

## File Organization
The file organization of the TGEditor repository is shown below. All the source code of the TGEditor is available in the `src` folder, which include the configuration files `config.yml`, and interactive training notebook `train.ipynb` and results visualization notebook `analysis.ipynb`. The model checkpoints are stored in the `model` folder, while some intermediate evaluation results are stored in `results` folder. All the other folders are dataset related folders.
```
TGEDITOR
├── src
│   ├── analysis.ipynb
│   ├── config.yml
│   ├── requirements.txt
│   ├── HTNE.py
│   ├── TGEditor.py
│   ├── TGEditor_fast.py
│   ├── train.ipynb
│   └── utils.py
├── README.md
├── model
│   ├── Model checkpoints
│   └── ....
├── results
│   ├── Evaulation results
│   └── ....
├── Chase
├── small_AML
├── medium_AML
└── docs
```


Please consider cite
```
@inproceedings{10.1145/3604237.3626883,
author = {Zhang, Shuaicheng and Zhu, Yada and Zhou, Dawei},
title = {TGEditor: Task-Guided Graph Editing for Augmenting Temporal Financial Transaction Networks},
year = {2023},
isbn = {9798400702402},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3604237.3626883},
doi = {10.1145/3604237.3626883},
abstract = {Recent years have witnessed a growth of research interest in designing powerful graph mining algorithms to discover and characterize the structural pattern of interests from financial transaction networks, motivated by impactful applications including anti-money laundering, identity protection, product promotion, and service promotion. However, state-of-the-art graph mining algorithms often suffer from high generalization errors due to data sparsity, data noisiness, and data dynamics. In the context of mining information from financial transaction networks, the issues of data sparsity, noisiness, and dynamics become particularly acute. Ensuring accuracy and robustness in such evolving systems is of paramount importance. Motivated by these challenges, we propose a fundamental transition from traditional mining to augmentation in the context of financial transaction networks. To navigate this paradigm shift, we introduce TGEditor, a versatile task-guided temporal graph augmentation framework. This framework has been crafted to concurrently preserve the temporal and topological distribution of input financial transaction networks, whilst leveraging the label information from pertinent downstream tasks, denoted as , inclusive of crucial downstream tasks like fraudulent transaction classification. In particular, to efficiently conduct task-specific augmentation, we propose two network editing operators that can be seamlessly optimized via adversarial training, while simultaneously capturing the dynamics of the data: Add operator aims to recover the missing temporal links due to data sparsity, and Prune operator is formulated to remove irrelevant/noisy temporal links due to data noisiness. Extensive results on financial transaction networks demonstrate that TGEditor 1) well preserves the data distribution of the original graph and 2) notably boosts the performance of the prediction models in the tasks of vertex classification and fraudulent transaction detection.},
booktitle = {Proceedings of the Fourth ACM International Conference on AI in Finance},
pages = {219–226},
numpages = {8},
keywords = {temporal graph augmentation, generative graph model, Financial transaction networks},
location = {<conf-loc>, <city>Brooklyn</city>, <state>NY</state>, <country>USA</country>, </conf-loc>},
series = {ICAIF '23}
}
```
