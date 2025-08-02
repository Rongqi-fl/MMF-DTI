#  MMF

[GitHub]([https://github.com/Rongqi-fl](https://github.com/Rongqi-fl/MMF-DTI/)

##  Dependencies
  
Quick install: `pip install -r requirements.txt`
  
Dependencies:
- python 3.8+
- pytorch >=1.2
- numpy
- sklearn
- tqdm
- prefetch_generator
- rdkit
- transformers
- torch-geometric
- fair-esm

##  Usage
  
`python main.py <dataset> [-m,--model] [-s,--seed] [-f,--fold]`
  
Parameters:
- `dataset` : `Davis` , `KIBA` 
- `-m` or `--model` :  *default:*`MMF`
- `-s` or `--seed` : set random seed, *optional*
- `-f` or `--fold` : set K-Fold number, *optional*
  
##  Project Structure
  
- DataSets: Data used in paper.
- Two modalities: sequence-based feature and structural feature.
- utils: A series of tools.
  - LLMData.py: Generates sequence-based feature.
  - GraphData.py: Generates structural feature.
  - DataSetFunction.py: Loads and processes training datasets.
- config.py: model config.
- LossFunction.py: Loss function used in paper.
- main.py: main file of project.
- model.py: Proposed model in paper.
- README.md: this file
- requirements.txt: dependencies file
- RunModel.py: Train, validation and test programs.
  

