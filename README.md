# Backdoored-COVID-Net
Backdoor attacks on a deep neural network-based system for COVID-19 detection from chest X-ray images


## Usage
### Requirements
- python 3.6.6
- tensorflow-gpu==1.15.0
- scikit-learn==0.23.2
- numpy==1.19.2
- opencv-python==4.4.0.44
- matplotlib==3.3.2
- Pillow==7.2.0

### See lindawangg/COVID-Net: Table of Contents for installation.
- Check the requirements
- Generate the COVIDx dataset
- Download the following datasets
  - covid-chestxray-dataset
    - Figure1-COVID-chestxray-dataset
    - rsna-pneumonia-detection-challenge
    - COVID-19-Radiography-Database
  - Use create_COVIDx.ipynb
- Download the COVID-Net models available
  - COVIDNet-CXR4-A


<pre>
# Directories
.
├── models
│   └── COVIDNet-CXR4-A
│ 
├── data
│   ├── train
│   └── test
│
├── labels
│ 
├── train.py
├── eval.py
├── data.py
└── poison.py
</pre>


### Training backdoored model
- training targeted to COVID-19 model
```
python train.py 
    --backdoor_attack = True
    --attack_type = 'targeted'
    --targeted_class = 2

```
### Training fine-tune model
```
python train.py 
    --backdoor_attack = False

```
