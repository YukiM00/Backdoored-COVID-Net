# Backdoored-COVID-Net

This repository contains the codes used in our study on [*Backdoor attacks to deep neural network-based system for COVID-19 detection from chest X-ray images*](https://doi.org/10.3390/app11209556).

## Terms of use
MIT licensed. Happy if you cite our paper when using the codes:

Matsuo Y & Takemoto K (2021) Backdoor attacks to deep neural network-based system for COVID-19 detection from chest X-ray images. Appl. Sci. 11, 9556. doi:10.3390/app11209556

## Usage
### Requirements
- python 3.6.6
- tensorflow-gpu==1.15.0
- scikit-learn==0.23.2
- numpy==1.19.2
- opencv-python==4.4.0.44
- matplotlib==3.3.2
- Pillow==7.2.0

### Image dataset and Model
See [lindawangg/COVID-Net: Table of Contents](https://github.com/lindawangg/COVID-Net#table-of-contents) for installation.
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


### Backdoored models
e.g., obtain the backdoored model for targeted attacks to *COVID-19*.
```
python train.py \
   --backdoor_attack = True \
   --attack_type = 'targeted' \
   --targeted_class = 2
```

### Fine-tuned models from backdoored models
```
python train.py \
   --backdoor_attack = False
```
