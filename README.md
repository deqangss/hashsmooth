# HashSmooth

This is the code repository for the paper [HashSmooth: Probabilistic Robustness Certificates Against Adversarial Examples in Security Classification](https://anonymous.url)

# Descriptions

We propose new robustness certification methods dedicated to security classification, which are model-agnostic and flexible to adapt discrete features (including for example, categorical vectors, bag-of-word frequencies, and sequences).
The framework joins locality-sensitive Hashing with randomized Smoothing (HashSmooth). 


# Models

All classification models are in the directory ```./security_classifies```. 
* In ```./security_classifies/drebin```, we build the malware detectors, for which all instructions can be found in the files of ```script_malware_detection.sh``` and ```script_malware_detection_malscan.sh```.
* For the CodeBERT and CodeGPT, both models are from the [Hugging Face website](https://huggingface.co/models). There are details in each model and each task. The directories of ```./security_classifies/CodeXGLUE``` and ```./security_classifies/CodeGPT``` is structured as

```
.
├── drebin
│   └── drebin_test.py   # train malware detectors
│   ├── drebin_cert_test.py # perform certificates
│   ├── drebin_adaptive_attack_test.py # l0 PGD attacks
│   ├── drebin_mimicry_test.py  # mimicry attacks
│   ├── script_malware_detection.sh  # scripts for performing instructions on the Drebin dataset
│   ├── script_malware_detection_malware.py # scripts for performing instructions on the Malscan dataset
├── CodeBERT (CodeXGLUE)
│   ├── Authorship-Attribution
│   │   └── code
│   │       ├── run.py
│   │       ├── run_smooth.py
│   │       ├── attack.py
│   │       └── README.md
│   │   └── dataset
│   ├── Clone Detection
│   │   └── code
│   │       ├── run.py
│   │       ├── run_smooth.py
│   │       ├── attack.py
│   │       └── README.md
│   │   └── dataset
├── CodeGPT 
│   ├── Authorship-Attribution
│   │   └── code
│   │       ├── run.py
│   │       ├── run_smooth.py
│   │       └── attack.py
│   │   └── dataset
│   ├── Clone Detection
│   │   └── code
│   │       ├── run.py
│   │       ├── run_smooth.py
│   │       └── attack.py
│   │   └── dataset
└── GraphCodeBERT
    ├── Authorship Attribution
    │   └── code
    │       ├── run.py
    │       ├── run_smooth.py
    │       └── attack.py
    │   └── dataset
    └── Clone Detection
        └── code
            ├── run.py
            ├── run_smooth.py
            └── attack.py
        └── dataset
```

Within each code directory, there is ```scripts_***.sh``` file telling instructions for running the code. Please read the ```README.md``` file at first.

## Datasets

### malware detection
  * We conduct the experiments upon the dataset [Drebin](https://www.sec.cs.tu-bs.de/~danarp/drebin/) and [Malcan](https://github.com/malscan-android/MalScan), respectively. 
    Both datasets are required to follow the policies of their own to obtain the apks. The sha256 checksums of benign apps in Drebin are available at [here](https://drive.google.com/drive/folders/1AHnNhtE2-YLWj8jeyciW52lFqFGdEmTB?usp=sharing). 
    These apks files can be downloaded from [Androzoo](https://androzoo.uni.lu/).
  * For reproducing the experimental results on the Drebin or Malscan dataset, we provide a portion of intermediate medium (e.g., vocabulary, dataset splitting info, etc.) available [here](https://drive.google.com/file/d/1JOiMzOjdgpyjM6WSmegpGmr6-32EEVYk/view?usp=share_link)
However, the data preprocessing step cannot be avoided, which means we need to download the apps and then conduct the experiments. We believe this is necessary because we attempt to generate the realistic attacks.

### Code analysis tasks
The datasets are from [the previous study](https://arxiv.org/pdf/2201.08698). We use their datasets that can be downloaded from this [Baidu Drive](
https://pan.baidu.com/s/19abfEewihl2Y-KwbW25shw?pwd=p847) with the extraction code p847. After decompressing this file, the folder structure is as follows.

```
.
└── CodeBERT
    └── Vulnerability Detection
        └── data
            ├── attack results
            │   ├── GA
            │   │   ├── attack_genetic_test_subs_0_400.csv
            │   │   ├── attack_gi_test_subs_0_400.log
            │   │   ├── ...
            │   └── MHM-LS
            │       ├── mhm_attack_ls_subs_0_400.log
            │       ├── mhm_attack_lstest_subs_0_400.csv
            │       ├── ...
            ├── dataset
            │   ├── test.jsonl
            │   ├── train.jsonl
            │   └── valid.jsonl
            └── substitutes
                ├── test_subs_0_400.jsonl
                ├── test_subs_1200_1600.jsonl
                ├── ...
```

# Requirements

Please see the file named ```requirements.txt```, which lists the necessary packages.

> [!WARNING]

> This is an ongoing repository.
