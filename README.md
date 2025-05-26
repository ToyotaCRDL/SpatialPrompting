# SpatialPrompting

[**SpatialPrompting: Keyframe-driven Zero-Shot Spatial Reasoning with Off-the-Shelf Multimodal Large Language Models**](http://arxiv.org/abs/2505.04911)
*Shun Taguchi, Hideki Deguchi, Takumi Hamazaki, and Hiroyuki Sakai*

This repository contains the source code for the [paper](http://arxiv.org/abs/2505.04911). Our method leverages keyframe-driven prompt generation to perform zero-shot spatial reasoning in 3D environments using off-the-shelf multimodal large language models (LLMs) without the need for 3D-specific fine-tuning.

## Relation to the Paper

The provided code implements the following core components from the paper:

- **Spatial Feature Extraction**
  The script `extract_features.py` extracts spatial embeddings from RGB images, depth maps, and camera pose information. These embeddings represent the scene and are saved in a specified `.npz` file.

- **Interactive Spatial Question Answering**
  The script `spatialqa.py` loads the spatial features and interacts with an LLM to answer spatial questions. It integrates keyframe data and corresponding camera poses into prompts for the LLM.

- **Dataset Prediction and Evaluation**
  - `predict_scanqa.py` and `predict_sqa3d.py` generate predictions on the ScanQA and SQA3D datasets, respectively.
  - `score_scanqa.py` and `score_sqa3d.py` evaluate the predictions using metrics such as EM@1, BLEU, METEOR, ROUGE-L, CIDEr, and SPICE.

## Dataset Organization

All dataset files should be placed under a unified base directory. For example, if your base directory is `/path/to/your/data`, the folder structure should be organized as follows:

```
/path/to/your/data
└── data
    ├── ScanNet
    ├── ScanQA
    └── SQA3D
```

Please extract .sens files of the ScanNet.
When running the scripts, specify the base path using the `--base_path` argument. 

## Environment Setup and Dependencies

The code requires the following major dependencies:

- **Python 3.x**
- **PyTorch** and **torchvision**
- **NumPy**
- **Pillow (PIL)**
- **CLIP** ([clip](https://github.com/openai/CLIP]))
- **OpenAI Python SDK** (`openai`)
- **Google Generative AI SDK** (`google.generativeai`)
- **EasyDict**
- Standard libraries (e.g., argparse, glob, json, collections, re)

Additionally, set the following environment variables for API access:

```bash
export OPENAI_API_KEY="your_openai_api_key"
export GOOGLE_API_KEY="your_google_api_key"
```

## Execution

Each script is designed to be run from the command line with various options.

### 1. Generate Spatial Embeddings

Run `extract_features.py` to extract spatial features from your dataset:

```bash
python extract_features.py \
  --base_path /path/to/your/data \
  --dataset scannet \
  --env scene0050_00 \
  --model vitl336
```

This script reads RGB, depth, and pose files from the specified directory and saves the computed embeddings to an `.npz` file.

### 2. Interactive Spatial QA

Run `spatialqa.py` to launch an interactive session that answers spatial questions based on the loaded spatial features:

```bash
python spatialqa.py \
  --llm gpt-4o-2024-11-20 \
  --feature /path/to/spatial_feature.npz \
  --image_num 30
```

Follow the on-screen prompts to enter your questions.

### 3. Prediction on ScanQA Dataset

Use `predict_scanqa.py` to generate predictions for the ScanQA dataset. Options allow for few-shot/zero-shot mode, merging settings, and whether to include camera pose information:

```bash
python predict_scanqa.py \
  --base_path /path/to/your/data \
  --llm gpt-4o-2024-11-20 \
  --model vitl336 \
  --image_num 30
```

The results will be saved in JSONL format.

### 4. Prediction on SQA3D Dataset

Run `predict_sqa3d.py` for generating predictions on the SQA3D dataset. Adjust options as needed (e.g., few-shot mode):

```bash
python predict_sqa3d.py \
  --base_path /path/to/your/data \
  --llm gpt-4o-2024-11-20 \
  --model vitl336 \
  --image_num 30 \
```

### 5. Evaluation

Evaluate the prediction results using the provided scoring scripts:

- **ScanQA Evaluation**

  ```bash
  python score_scanqa.py --base_path /path/to/your/data --pred /path/to/prediction.jsonl [--use_spice]
  ```

- **SQA3D Evaluation**

  ```bash
  python score_sqa3d.py --base_path /path/to/your/data --pred /path/to/prediction.jsonl
  ```
