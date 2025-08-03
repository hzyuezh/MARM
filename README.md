#  MARM
Code of our paper: "MARM: Medical Adaptive Reasoning Model".

# Environments
This repository contains the codebase for SFT and RL based on LLaMA-Factory (https://github.com/hiyouga/LLaMA-Factory) and Unsloth (https://unsloth.ai/?ref=producthunt). We use two separate conda environments for each stage.


# Training
File "data/train1.jsonl" is used for training in the SFT stage, while File "data/train2.jsonl" is used for training in the RL stage.


# Evaluate
The "eval.py "script is used to evaluate model performance, while the "statistic.py" script is used to compute and summarize experimental results.

```yaml
cd evaluate
python eval.py
python statistic.py
```
