## Introduction

This repository contains source code that was used in 2024 Deepfake Audio Detection Competition held by National Program of Excellence in Software. (SW중심대학 디지털 경진대회_SW와 생성AI의 만남 : AI 부문)

## Environment Setup

We provide 2 separate environments for preprocessing and model training stage. We ran experiments in Ubuntu 20.04.5 LTS with CUDA 12.2, using single NVIDIA A100 Tensor Core GPU.

### Dependencies
For linux, libsox-dev is required.
```bash
sudo apt-get update
sudo apt install libsox-dev
```

### Requirements
```bash
conda create -n StripedMarlin python=3.8
conda activate StripedMarlin
pip install -r requirements.txt

# Or
conda env create --file StripedMarlin.yaml
```

```bash
conda env create --file modelscope.yaml
```


## Train Dataset Preparation


### (1) Noise Generation (Text-to-Audio)
1. Generate text prompts using LLaMA-v2. Note that we clean up the outputs by removing duplicates or repeated words. Example output and fine-grained output can be found at `preprocessing/example/llama2_output_raw.txt` and `preprocessing/example/llama2_output_finegrained.txt`, respectively.
   ```bash
   python preprocessing/llama.py
   ```
2. Generate audios based on the outputs using AudioLDM. You can also use command-line inside the bash script.
   ```bash
   bash preprocessing/generate_noise.sh path_to_finegrained_list.txt
   ```

### (2) Noise Generation (Vocal Removal)
1. Extract background noises from unlabeled data.
   ```bash
   python preprocessing/inference.py -i path/to/unlabeled_data --save_option --noise-only
   ```

2. As some outputs could still include human voices, run vad on the outputs.
   ```bash
   python preprocessing/run_vad.py 
   python preprocessing/filter_voices.py
   ```
3. Filter some noises that are too small.
   ```bash
   python preprocessing/filter_silences.py
   ```

### Clustering Preparation
We provide code for visualizing clusters of voices using audio embeddings. 

1. Extract embeddings
   ```bash
   cd preprocessing/clustering
   python extract_embeddings.py -i path/to/train_data
   ```
2. Run Clustering
   ```bash
   python run_clustering.py
   cd ../..
   ```

After you obtain the `cluster_labels.csv`, combine the labels with the original `train.csv` and name it as `train_w_cl.csv`.


## Training

### Data path
Set data paths as below:
- train data
- noise data
- train_cluster data

### Run
You can use `train_models.py` and configs in `configs/*.yaml`. Here we provide training scripts for 2 best models.

```bash
python train_models.py --config configs/WavLM.yaml
python train_models.py --config configs/AASIST.yaml
```

## Test

For running inference, we have several preprocessing steps for better results. If you would like to skip this and directly reproduce the results, see [Guidlines for Reproducing Private Score](###guidelines-for-reproducing-private-score).

### Preprocessing (Noise removal)
   ```bash
    python preprocessing/inference.py -i path/to/test_data --save_option --vocal-only
   ```

### Inference
Before you run inference, set the checkpoint path of model config to ... We provide inference scripts of best models as examples.

```bash
python inference.py --config configs/AASIST.yaml --mode vr
python inference.py --config configs/WavLM.yaml --mode raw
```
### Ensemble
After obtaining inference csv files, we enxembl  

## Results

### Guidlines for Reproducing Private Score
This part describes directly reproducing private score.
Download 2 checkpoints and set the path to AASIST_best.yaml and WavLM_best.yaml.

1. For AASIST, run:
   ```bash
   python inference.py --config configs/AASIST_best.yaml --mode vr
   ```
2. For WavLM, run:
   ```bash
   python inference.py --config configs/WavLM_best.yaml --mode raw
   ```
3. You will see 2 submission files created. Ensemble them by:
   ```bash
   python ensemble.py path_to_csv_file1 path_to_csv_file2
   ```
4. You will see 1 ensembled csv file created. Post-process the result using VAD:
   ```bash
   python rewrite.py path_to_csv_file preprocessing/example/vad.csv path_to_output_file.csv
   ```
5. Finally, ensemble result from 3 and 4:
   ```bash
   python ensemble.py csv_after_step3.csv csv_after_step4.csv
   ```


## Acknowledgments

This project is built upon repository [audio-deepfake-adversarial-attacks](https://github.com/piotrkawa/audio-deepfake-adversarial-attacks). Thanks to the contributors of the great codebase.