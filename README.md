
<div align="center" style="font-family: charter;">
<h1><img src="imgs/icon.png" width="5%"/>&nbsp;<i>DIG</i>:</br><span style="font-size: 0.9em;">Adapting Frame Selection to Query Types for Long-Form Video Understanding</span></h1> 

[![arXiv](https://img.shields.io/badge/arXiv-DIG-b31b1b.svg?logo=arxiv)](https://arxiv.org/abs/2512.04000) [![License](https://img.shields.io/badge/License-MIT-green.svg)]() [![Python](https://img.shields.io/badge/Python-3.10-blue.svg)]()

<div>
    <a href="https://jialuo-li.github.io/" target="_blank">Jialuo Li</a><sup>1,2,*</sup>,</span>
    <a target="_blank">Bin Li</a><sup>2</sup>, </span>
    <a target="_blank">Jiahao Li</a><sup>2</sup>,</span>
    <a target="_blank">Yan Lu</a><sup>2</sup>,</span>
</div>

<div>
    <sup>1</sup>Tsinghua University&emsp;
    <sup>2</sup>Microsoft Research Asia&emsp;
    <sup>*</sup>Work done during Jialuo's internship at MSRA&emsp;
</div>

<p></p>

<img src="imgs/teaser.jpg" width="100%"/>

<p align="justify"><i>Overview of the DIG Framework. An LLM first classifies the input query as either global or localized. Global queries trigger uniform sampling across the entire video. Conversely, localized queries utilize CAFS and reward assignment to generate a reward distribution; this distribution is used to construct a refined video for targeted uniform sampling. The selected frames are subsequently processed by the LMM for final inference.</i></p>

</div>  


---

## 📰 News
* **[2026-02-21]** 🎉 Exciting news! Our paper has been accepted to **CVPR 2026**!

---

## 🚀 Quick Start

### 1. Installation

Set up a clean environment to avoid conflicts.

```bash
# Create and activate conda environment
conda create -n dig python=3.10 -y
conda activate dig

# Clone the repository
git clone git@github.com:Jialuo-Li/DIG.git
cd DIG

# Install dependencies
bash scripts/install.sh
```

### 2. Data Preparation

Download the supported benchmarks and organize them in the `data/` directory.

| Dataset | Link | Description |
| :--- | :--- | :--- |
| **MLVU** | [Hugging Face](https://huggingface.co/datasets/sy1998/MLVU_dev) | Multi-Task Long Video Understanding |
| **LongVideoBench** | [Hugging Face](https://huggingface.co/datasets/longvideobench/LongVideoBench) | Long-context video QA |
| **VideoMME** | [Hugging Face](https://huggingface.co/datasets/lmms-lab/Video-MME) | Comprehensive video evaluation |

**Directory Structure:**
```text
data/
├── mlvu/
│   ├── 1.mp4
│   └── ...
├── longvideobench/
│   └── videos/
│       ├── 1.mp4
│       └── ...
└── videomme/
    └── data/
        ├── 1.mp4
        └── ...
```

### 3. Inference

We provide pre-computed **query types**, **r-frame indices**, and **reward values** from *Qwen2.5-VL-7B/32B* and *Qwen3-VL-8B* in the `rewards/` directory. This allows you to directly evaluate DIG's performance.

**1. Set the Target Model**
```bash
export MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct 
# Supported: Qwen/Qwen2.5-VL-32B-Instruct, Qwen/Qwen3-VL-8B-Instruct
```

**2. Video Refinement (Key Frame Selection)**
Extract the optimal keyframes based on the pre-computed rewards.
```bash
# Usage: bash scripts/video_refinement.sh <dataset>
bash scripts/video_refinement.sh mlvu # Options: longvideobench, videomme
```

**3. Run Evaluation**
Evaluate using the `lmms-eval` framework.
```bash
# Usage: bash scripts/eval/qwen25vl.sh <dataset> <method>

# For Qwen2.5-VL
bash scripts/eval/qwen25vl.sh mlvu DIG 

# For Qwen3-VL
bash scripts/eval/qwen3vl.sh mlvu DIG 
```
**Supported Methods:**
 *   `DIG`: Uses DIG pipeline.
 *   `UNI`: Uses standard uniform sampling.

---

## 🛠️ Full Pipeline Workflow

If you wish to run the entire DIG process from scratch, please follow these steps.

### Step 1: Query Identification
The LLM is used to analyzes the user query as `Global` or `Localized`.

```bash
# 1. Launch the LLM Server
export MODEL_NAME=Qwen/Qwen3-Next-80B-A3B-Instruct
bash scripts/launch_llm.sh

# 2. Run Identification
bash scripts/query_identification.sh mlvu # Options: longvideobench, videomme
```

### Step 2: Content-Aware Frame Selection (CAFS)
We use **DINOv2** to extract features and select representative "r-frames" from the video.

```bash
bash scripts/cafs.sh mlvu # Options: longvideobench, videomme
```

### Step 3: Reward Assignment
The LMM is used to score the r-frames based on their relevance to the user query.

```bash
# 1. Launch the LMM Server (e.g., Qwen2.5-VL-7B)
export MODEL_NAME=Qwen/Qwen2.5-VL-7B-Instruct
bash scripts/launch_mllm.sh

# 2. Assign Rewards
bash scripts/reward_assignment.sh mlvu # Options: longvideobench, videomme
# Results are saved to the 'rewards/' directory.
```

### Step 4: Video Refinement
Use the generated rewards to construct the final frame input for inference.
*(See the "Inference" section above for this step)*

---

## 📂 Project Structure

```text
DIG/
├── data/                   # Dataset storage
├── lmms-eval/              # Evaluation framework
├── pipeline/               # Core DIG implementation
│   ├── cafs.py             # Content-Aware Frame Selection
│   ├── query_identification.py # Global vs. Localized classification
│   ├── reward_assignment.py    # Frame relevance scoring
│   └── video_refinement.py     # Final frame selection 
├── rewards/                # Pre-computed metadata & rewards
├── scripts/                # execution scripts
│   ├── eval/               # Evaluation launchers
│   ├── launch_llm.sh       # vLLM Server for Query Identification
│   └── launch_mllm.sh      # vLLM Server for Reward Assignment
├── utils.py                # Dataset loader & Prompt templates
└── requirements.txt        # Python dependencies
```

## 🤝 Citation
If you find DIG useful for your research or projects, we would greatly appreciate it if you could cite our work:

```bibtex
@misc{li2025dividegroundadaptingframe,
      title={Divide, then Ground: Adapting Frame Selection to Query Types for Long-Form Video Understanding}, 
      author={Jialuo Li and Bin Li and Jiahao Li and Yan Lu},
      year={2025},
      eprint={2512.04000},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.04000}, 
}
```
