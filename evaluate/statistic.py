import json
import os
from collections import defaultdict
import pandas as pd

input_dir = ""
output_file = ""


def load_jsonl(file_path):
    data_list = []
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            data_list.append(json.loads(line.strip()))
    return data_list


def analyze_files(input_dir):
    results = {}
    file_names = []

    # for filename in os.listdir(input_dir):
    for filename in file_names:
        file_path = os.path.join(input_dir, filename)
        if not os.path.isfile(file_path) or not filename.endswith(".jsonl"):
            continue

        print(f"Processing file: {filename}")
        data_list = load_jsonl(file_path)

        dataset_stats = defaultdict(lambda: {"acc": [], "tokens": []})

        for item in data_list:
            dataset = item.get("dataset")
            acc = item.get("acc")
            tokens = item.get("tokens")

            if dataset is None or acc is None or tokens is None:
                continue

            dataset_stats[dataset]["acc"].append(acc)
            dataset_stats[dataset]["tokens"].append(tokens)

        avg_results = {}
        for ds, values in dataset_stats.items():
            avg_acc = sum(values["acc"]) / len(values["acc"]) if values["acc"] else 0
            avg_tokens = (
                sum(values["tokens"]) / len(values["tokens"]) if values["tokens"] else 0
            )
            avg_results[ds] = {"acc": avg_acc, "tokens": avg_tokens}

        results[filename] = avg_results

    return results


def generate_table(results, output_file):
    dataset_order = [
        "medqa-cn",
        "medqa-us",
        "medmcqa",
        "pubmedqa-pqal",
        "triage",
        "diagnosis",
        "mmlu-pro",
        "diagnosis-agentclinic",
        "triage-clinicallab",
        "diagnosis-clinicallab",
        "triage-medjourney",
        "diagnosis-medjourney",
    ]

    acc_data = {}
    tokens_data = {}

    for filename, file_data in results.items():
        acc_row = {}
        tokens_row = {}
        for ds in dataset_order:
            acc_value = file_data.get(ds, {}).get("acc", None)
            token_value = file_data.get(ds, {}).get("tokens", None)

            if isinstance(acc_value, (int, float)):
                acc_row[ds] = acc_value * 100
            else:
                acc_row[ds] = acc_value

            tokens_row[ds] = token_value

        acc_data[filename] = acc_row
        tokens_data[filename] = tokens_row

    df_acc = pd.DataFrame(acc_data).T
    df_tokens = pd.DataFrame(tokens_data).T

    with pd.ExcelWriter(output_file) as writer:
        df_acc.to_excel(writer, sheet_name="Accuracy")
        df_tokens.to_excel(writer, sheet_name="tokens")

    print(f"Saved")


# 主流程
if __name__ == "__main__":
    results = analyze_files(input_dir)
    generate_table(results, output_file)
