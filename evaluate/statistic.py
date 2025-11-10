import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import json
from tqdm import tqdm
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
from scipy.optimize import linear_sum_assignment
import numpy as np

output_file_template = "/XXX/eval_res/{file_name}.json"
all_output_file = "/XXX/eval_res/all.xlsx"

CONFIGS = [
    {
        "file_name":"XXX",
        "tokenizer":"/XXX/tokenizer/XXX"
    }
]

clinical_bert_path = "/XXX/ClinicalBERT"
clinical_bert_tokenizer = AutoTokenizer.from_pretrained(clinical_bert_path)
clinical_bert_model = AutoModel.from_pretrained(clinical_bert_path)
clinical_bert_model.eval()
device = "cuda" if torch.cuda.is_available() else "cpu"
clinical_bert_model.to(device)

def load_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_json(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data

def save_json(file_path, data):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

def save_excel(file_path, data_list, columns=None):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df = pd.DataFrame(data_list, columns=columns)
    df.to_excel(file_path, index=False)


def get_clinical_bert_embeddings(texts, tokenizer, model, device="cuda", max_length=512):
    if not texts:
        return np.array([]).reshape(0, model.config.hidden_size)
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        return_attention_mask=True
    ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        token_embeddings = outputs.last_hidden_state
        attention_mask = inputs["attention_mask"]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embeddings = sum_embeddings / sum_mask
    return embeddings.cpu().numpy()

def get_bert_pair_acc(disease1: str, disease2: str, threshold: float = 0.7) -> int:
    emb1 = get_clinical_bert_embeddings([disease1], clinical_bert_tokenizer, clinical_bert_model, device=device)
    emb2 = get_clinical_bert_embeddings([disease2], clinical_bert_tokenizer, clinical_bert_model, device=device)
    sim = cosine_similarity(emb1, emb2)[0][0]
    return 1 if sim >= threshold else 0

def get_bert_set_acc(set1: list, set2: list, threshold: float = 0.7) -> float:
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0

    emb1 = get_clinical_bert_embeddings(set1, clinical_bert_tokenizer, clinical_bert_model, device=device)
    emb2 = get_clinical_bert_embeddings(set2, clinical_bert_tokenizer, clinical_bert_model, device=device)

    sim_matrix = cosine_similarity(emb1, emb2) 
    cost_matrix = 1 - sim_matrix  
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    matches = 0
    for i, j in zip(row_ind, col_ind):
        if sim_matrix[i, j] >= threshold:
            matches += 1

    total = max(len(set1), len(set2))
    return matches / total

def get_bert_set_tp_fp_fn(
    pred_list, 
    gold_list, 
    threshold=0.7, 
    device="cuda"
):
    if not pred_list and not gold_list:
        return 0, 0, 0
    if not pred_list:
        return 0, 0, len(gold_list)
    if not gold_list:
        return 0, len(pred_list), 0

    pred_embs = get_clinical_bert_embeddings(pred_list, clinical_bert_tokenizer, clinical_bert_model, device=device)
    gold_embs = get_clinical_bert_embeddings(gold_list, clinical_bert_tokenizer, clinical_bert_model, device=device)
    sim_matrix = cosine_similarity(pred_embs, gold_embs)
    cost_matrix = 1 - sim_matrix
    pred_indices, gold_indices = linear_sum_assignment(cost_matrix)

    tp = 0
    for p_idx, g_idx in zip(pred_indices, gold_indices):
        if sim_matrix[p_idx, g_idx] >= threshold:
            tp += 1

    fp = len(pred_list) - tp
    fn = len(gold_list) - tp
    return tp, fp, fn

def get_direct_answer(c1, c2, max_options=26):
    def normalize(ans):
        ans = ans.strip().lower()
        if ans.isalpha() and len(ans) == 1:
            idx = ord(ans) - ord("a")
            if 0 <= idx < max_options:
                return str(idx)
        return ans

    return 1.0 if normalize(c1) == normalize(c2) else 0.0

def compute_f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    return round(f1*100, 2)


for config in tqdm(CONFIGS, desc="Processing configurations"):
    file_name = config["file_name"]
    model_path = config["tokenizer"]
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    input_file = f"/XXX/llm_eval/{file_name}.jsonl"
    output_file = output_file_template.format(file_name=file_name)
    if os.path.exists(output_file):
        continue
    
    data_list = load_jsonl(input_file)
    
    temp_dict_acc = {}
    temp_dict_tp_fp_fn = {
        "diagnosis": {"tp": 0, "fp": 0, "fn": 0, "token": 0, "total": 0},
        "clinical_lab_diagnosis": {"tp": 0, "fp": 0, "fn": 0, "token": 0, "total": 0}
    }
    f1_datasets = ["diagnosis-cn","diagnosis-en","clinical_lab_diagnosis"]

    for item in tqdm(data_list, desc=f"Evaluating {file_name}",leave=False):
        dataset = item.get("dataset")
        think = item["llm_res"].get("think", "")
        tokenized = tokenizer(think, add_special_tokens=False)  # 不加特殊token
        token = len(tokenized["input_ids"])
        
        if dataset in ["medqa-cn","medqa-us"]:
            dataset = "medqa"

        if dataset not in f1_datasets and dataset not in temp_dict_acc:
            temp_dict_acc[dataset] = {
                "acc": 0,
                "token": 0,
                "total": 0
            }

        acc = 0
        if dataset in ["agentclinic"]:
            if "error" in item.get("llm_eval", {}):
                acc = 0
                continue
            llm_eval = item.get("llm_eval", {}).get("diagnosis")
            gold_answer = item.get("output", "")
            acc = get_bert_pair_acc(llm_eval, gold_answer)
            
        elif dataset in ["clinical_lab_guidance"]:
            # print(item)
            llm_res = str(item.get("llm_res", {}).get("answer", "") or "").strip()
            gold_answer = item.get("output", "").strip()
            acc = 1 if llm_res == gold_answer else 0
            
        elif dataset in ["medjourney_dr"]:
            if "error" in item.get("llm_eval", {}):
                acc = 0
                continue
            llm_eval = item.get("llm_eval").get("judge", "").strip()
            acc = 1 if llm_eval == "correct" else 0
                        
        elif dataset in ["medjourney_dp"]:
            llm_res = str(item.get("llm_res", {}).get("answer", "") or "").strip()
            gold_answer = item.get("output", "").strip()
            acc = get_bert_pair_acc(llm_res, gold_answer)
            
        elif dataset in ["pubmedqa-pqal","triage-cn"]:
            llm_res = str(item.get("llm_res", {}).get("answer", "") or "").strip()
            gold_answer = item.get("output", "").strip()
            acc = 1 if llm_res == gold_answer else 0
            
        elif dataset in ["medmcqa","medqa","mmlu-pro"]:
            llm_res = str(item.get("llm_res", {}).get("answer", "") or "").strip()
            gold_answer = item.get("output", "").strip()
            acc = get_direct_answer(llm_res, gold_answer, max_options=5)
            
        if dataset not in f1_datasets:
            temp_dict_acc[dataset]["acc"] += acc
            temp_dict_acc[dataset]["token"] += token
            temp_dict_acc[dataset]["total"] += 1
        
        tp,fp,fn = 0,0,0        
        if dataset in ["diagnosis-cn"]:
            if "error" in item.get("llm_eval", {}):
                tp,fp,fn = 0,0,0
                dataset = "diagnosis"
                continue
            llm_res = item.get("llm_eval", {}).get("resident_diagnosis", [])
            gold_answer = item.get("llm_eval", {}).get("reference_diagnosis", [])
            tp, fp, fn = get_bert_set_tp_fp_fn(llm_res, gold_answer)
            llm_res2 = item.get("llm_eval", {}).get("resident_differential_diagnosis", [])
            gold_answer2 = item.get("llm_eval", {}).get("reference_differential_diagnosis", [])
            tp2, fp2, fn2 = get_bert_set_tp_fp_fn(llm_res2, gold_answer2)
            tp += tp2
            fp += fp2
            fn += fn2
            dataset = "diagnosis"
        elif dataset in ["diagnosis-en"]:
            if "error" in item.get("llm_eval", {}):
                tp,fp,fn = 0,0,0
                dataset = "diagnosis"
                continue
            llm_res = item.get("llm_eval", {}).get("resident_diagnosis", [])
            gold_answer = item.get("llm_eval", {}).get("reference_diagnosis", [])
            tp, fp, fn = get_bert_set_tp_fp_fn(llm_res, gold_answer)
            dataset = "diagnosis"
        elif dataset in ["clinical_lab_diagnosis"]:
            if "error" in item.get("llm_eval", {}):
                tp,fp,fn = 0,0,0
                continue
            
            llm_res = item.get("llm_eval", {}).get("resident_diagnosis", [])
            gold_answer = item.get("llm_eval", {}).get("reference_diagnosis", [])
            tp, fp, fn = get_bert_set_tp_fp_fn(llm_res, gold_answer)
        if dataset in ["diagnosis","clinical_lab_diagnosis"]:
            temp_dict_tp_fp_fn[dataset]["tp"] += tp
            temp_dict_tp_fp_fn[dataset]["fp"] += fp
            temp_dict_tp_fp_fn[dataset]["fn"] += fn
            temp_dict_tp_fp_fn[dataset]["token"] += token
            temp_dict_tp_fp_fn[dataset]["total"] += 1

    acc_dict = {}
    token_dict = {}
    f1_dict = {}
    
    for dataset, stats in temp_dict_acc.items():
        total = stats["total"]
        avg_acc = stats["acc"] / total if total > 0 else 0
        avg_token = stats["token"] / total if total > 0 else 0
        acc_dict[dataset] = round(avg_acc * 100, 2)
        token_dict[dataset] = round(avg_token, 2)
        
    for dataset, stats in temp_dict_tp_fp_fn.items():
        tp = stats["tp"]
        fp = stats["fp"]
        fn = stats["fn"]
        f1 = compute_f1(tp, fp, fn)
        f1_dict[dataset] = f1
        total = stats["total"]
        avg_token = stats["token"] / total if total > 0 else 0
        token_dict[dataset] = round(avg_token, 2)

    template_keys = [
        "model",
        "f1_diagnosis",
        "acc_medqa",
        "acc_medmcqa",
        "acc_pubmedqa-pqal",
        "acc_triage-cn",
        "acc_id_avg",
        "token_medqa",
        "token_medmcqa",
        "token_pubmedqa-pqal",
        "token_triage-cn",
        "token_diagnosis",
        "token_id_avg",
        "f1_clinical_lab_diagnosis",
        "acc_mmlu-pro",
        "acc_agentclinic",
        "acc_clinical_lab_guidance",
        "acc_medjourney_dr",
        "acc_medjourney_dp",
        "acc_ood_avg",
        "token_mmlu-pro",
        "token_agentclinic",
        "token_clinical_lab_guidance",
        "token_clinical_lab_diagnosis",
        "token_medjourney_dr",
        "token_medjourney_dp",
        "token_ood_avg"
    ]

    final_dict = {}
    for key in template_keys:
        if key == "model":
            final_dict[key] = file_name
        else:
            final_dict[key] = 0.0
            
    for dataset, f1 in f1_dict.items():
        f1_key = f"f1_{dataset}"
        if f1_key in final_dict:
            final_dict[f1_key] = f1
            
    for dataset, acc in acc_dict.items():
        acc_key = f"acc_{dataset}"
        if acc_key in final_dict:
            final_dict[acc_key] = acc

    for dataset, token in token_dict.items():
        token_key = f"token_{dataset}"
        if token_key in final_dict:
            final_dict[token_key] = token

    id_datasets = ["medqa", "medmcqa", "pubmedqa-pqal", "triage-cn"]
    ood_datasets = ["mmlu-pro", "agentclinic", "clinical_lab_guidance", "medjourney_dr", "medjourney_dp"]
    
    id_token_datasets = ["medqa", "medmcqa", "pubmedqa-pqal", "triage-cn", "diagnosis"]
    ood_token_datasets = ["mmlu-pro", "agentclinic", "clinical_lab_guidance", "clinical_lab_diagnosis", "medjourney_dr", "medjourney_dp"]

    final_dict["acc_id_avg"] = round(sum(final_dict[f"acc_{d}"] for d in id_datasets) / len(id_datasets),2)
    final_dict["token_id_avg"] = round(sum(final_dict[f"token_{d}"] for d in id_token_datasets) / len(id_token_datasets),2)

    final_dict["acc_ood_avg"] = round(sum(final_dict[f"acc_{d}"] for d in ood_datasets) / len(ood_datasets),2)
    final_dict["token_ood_avg"] = round(sum(final_dict[f"token_{d}"] for d in ood_token_datasets) / len(ood_token_datasets),2)

    save_json(output_file, final_dict)
    

output_list = []
for config in CONFIGS:
    input_file = output_file_template.format(file_name=config["file_name"])
    final_dict = load_json(input_file)
    output_list.append(final_dict)

save_excel(all_output_file, output_list)