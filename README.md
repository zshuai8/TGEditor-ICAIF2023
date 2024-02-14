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