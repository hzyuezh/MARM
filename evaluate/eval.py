import json
import os
from time import sleep
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import re
from transformers import AutoTokenizer
import pandas as pd
from collections import defaultdict
from threading import Semaphore
from openai import OpenAI
from copy import deepcopy
import sys


FILE_NAME = ""

input_file = f"{FILE_NAME}.jsonl"
output_file = f"{FILE_NAME}.jsonl"


MODEL_PATH = (
    "/root/nas/code/adaptive-reasoning/9_eval/tokenizer/Meta-Llama-3.1-8B-Instruct"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

MAX_WORKERS = 20
BATCH_SIZE = 1000

API_KEY = ""
BASE_URL = ""
MODEL_NAME = "qwen/qwen3-235b-a22b"

prompt = r"""
你是一名专业的医生，你的任务是根据正确的诊断结果和鉴别诊断结果，判断模型生成的诊断结果和鉴别诊断结果是否正确。
如果模型回答正确，请返回{"answer": "true"}，如果模型回答错误，请返回{"answer": "false"}。只能选择一个。

### 输出格式：
{"answer": "true | false"}
"""


def llm(content):
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

    for attempt in range(3):
        try:
            completion = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": content},
                ],
                temperature=0.01,
                extra_body={"chat_template_kwargs": {"thinking": False}},
            )
            res = completion.choices[0].message.content
            match = re.search(r"\{.*\}", res, re.DOTALL)
            if match:
                json_str = match.group(0)
            json_dict = json.loads(json_str)
            answer = json_dict.get("answer", "")
            return answer
        except Exception as e:
            if attempt <= 2:
                sleep(2)
                print(f"[WARNING] {e} (尝试 {attempt + 1}/3)")
            else:
                print(f"[ERROR] {e}")
                return "error"
    return "error"


def load_jsonl(file_path):
    data_list = []
    if not os.path.exists(file_path):
        return data_list
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data = json.loads(line.strip())
            data_list.append(data)
    return data_list


def save_jsonl(file_path, data_list, mode):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, mode, encoding="utf-8") as f:
        content = "".join(
            json.dumps(data, ensure_ascii=False) + "\n"
            for data in tqdm(
                data_list,
                desc=f"Saving {os.path.basename(file_path)}",
                total=len(data_list),
            )
        )
        f.write(content)


def py_eval(data, max_options=26):
    data = deepcopy(data)

    predict = data["predict"]
    predict_answer = str(predict["answer"]).strip().lower()
    predict_reasoning = predict["reasoning"]

    gold_answer = str(data["gold_answer"]).strip()

    if predict_answer.isalpha():
        if len(predict_answer) == 1:
            letter_index = ord(predict_answer) - ord("a")
            if 0 <= letter_index < max_options:
                predict_answer = str(letter_index)

    acc = 1.0 if predict_answer == gold_answer else 0.0

    tokenized = tokenizer(predict_reasoning, add_special_tokens=False)
    tokens = len(tokenized["input_ids"])
    data["acc"] = acc
    data["tokens"] = tokens

    return data


def llm_eval(data):
    data = deepcopy(data)

    predict = data["predict"]
    predict_answer = predict["answer"]
    predict_reasoning = predict["reasoning"]

    gold_answer = data["gold_answer"]

    tokenized = tokenizer(predict_reasoning, add_special_tokens=False)
    tokens = len(tokenized["input_ids"])

    text = f"""
    ### 以下是正确答案：
    {gold_answer}
    
    ### 以下是模型回复:
    {predict_answer}
    """

    content = prompt + text

    res = llm(content)
    if res == "error":
        data["acc"] = "error"
        data["tokens"] = tokens
        return data
    else:
        if res == "true":
            data["acc"] = 1.0
            data["tokens"] = tokens
        elif res == "false":
            data["acc"] = 0.0
            data["tokens"] = tokens
        return data


def process_item(data):
    data = deepcopy(data)
    if (
        data["dataset"] == "diagnosis"
        or data["dataset"] == "diagnosis-agentclinic"
        or data["dataset"] == "diagnosis-clinicallab"
        or data["dataset"] == "diagnosis-medjourney"
        or data["dataset"] == "triage-medjourney"
    ):
        data = llm_eval(data)
    else:
        data = py_eval(data)
    return data


def main():
    try:
        input_list = load_jsonl(input_file)
        output_list = load_jsonl(output_file)

        error_list = []
        error_ids = set()

        done_list = []
        done_ids = set()

        for item in output_list:
            if item.get("acc") == "error":
                if item["id"] not in error_ids:
                    error_list.append(item)
                    error_ids.add(item["id"])
            else:
                if item["id"] not in done_ids:
                    done_list.append(item)
                    done_ids.add(item["id"])

        data_list = [
            item
            for item in input_list
            if item["id"] in error_ids or item["id"] not in done_ids
        ]

        print(f"[INFO] 总计: {len(input_list)}")
        print(f"[INFO] 完成: {len(done_ids)}")
        print(f"[INFO] 错误: {len(error_ids)}")
        print(f"[INFO] 待处理: {len(data_list)}")

        all_results = done_list
        results_buffer = []

        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_data = {
                executor.submit(process_item, data): data for data in data_list
            }
            for future in tqdm(as_completed(future_to_data), total=len(future_to_data)):
                try:
                    result = future.result()
                    all_results.append(result)
                    results_buffer.append(result)

                    if len(results_buffer) >= BATCH_SIZE:
                        mode = "a" if os.path.exists(output_file) else "w"
                        save_jsonl(output_file, results_buffer, mode)
                        results_buffer.clear()
                except Exception as exc:
                    print(f"[ERROR] 处理任务出错: {exc}")

            # 保存剩余数据
            all_results = sorted(all_results, key=lambda x: x["id"])
            save_jsonl(output_file, all_results, mode="w")
            print(f"[INFO] 已完成最后一批 {len(all_results)} 条数据保存。")

        print(f"全部任务已完成")
    except KeyboardInterrupt:
        print("\n[WARNING] 检测到 Ctrl+C 中断请求，正在尝试保存当前进度...")
        all_results = sorted(all_results, key=lambda x: x["id"])
        save_jsonl(output_file, all_results, mode="w")
        print(f"[INFO] 已保存 {len(all_results)} 条数据。")
        # 退出程序
        sys.exit(0)


if __name__ == "__main__":
    main()
