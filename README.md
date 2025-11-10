#  MARM
Data and Code of our paper: "MARM: Medical Adaptive Reasoning Model".

# Environments
This repository contains the codebase for SFT and RL based on LLaMA-Factory (https://github.com/hiyouga/LLaMA-Factory) and Unsloth (https://unsloth.ai/?ref=producthunt). We use two separate conda environments for each stage.


# Training
File "data/train_sft.jsonl" is used for training in the SFT stage, while File "data/train_rl.jsonl" is used for training in the RL stage.


# Reasoning
The "run.py" script is used to run inference on the trained model. 


# Evaluate
The "eval.py "script is used to evaluate model performance, while the "statistic.py" script is used to compute and summarize experimental results.

```yaml
cd evaluate
python eval.py
python statistic.py
```
