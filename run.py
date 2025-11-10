from openai import OpenAI
import json
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from copy import deepcopy
import sys
import re


def llm_base(content, llm_config):
    client = llm_config["client"]
    model_name = llm_config["model_name"]
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": content}],
            temperature=0.01,
            frequency_penalty=0.3,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f" API 调用失败: {e}")
        raise e


file_name = "XXX"
input_file = "/XXX/test.jsonl"
output_file = f"/XXX/llm_res/{file_name}.jsonl"

BATCH_SIZE = 100
MAX_WORKERS = 64

LLM_CONFIGS = [
    {
        "model_name": "qwen",
        "api_key": "xxx",  # 替换为你的实际 API Key
        "base_url": "XXX",
        "func": llm_base,
    },
]

def extract_answer(text):
    text = text.strip()
    
    pattern_answer = re.compile(
        r'^(?P<prefix>.*?)<ANSWER>(?P<answer>.*?)</ANSWER>$',
        re.DOTALL
    )
    pattern_short_cot = re.compile(
        r'^(?P<prefix>.*?)<SHORT_COT>(?P<cot>.*?)</SHORT_COT>\s*<ANSWER>(?P<answer>.*?)</ANSWER>$',
        re.DOTALL
    )
    pattern_long_cot = re.compile(
        r'^(?P<prefix>.*?)<LONG_COT>(?P<cot>.*?)</LONG_COT>\s*<ANSWER>(?P<answer>.*?)</ANSWER>$',
        re.DOTALL
    )

    # 尝试匹配三种模式（顺序不重要，但互斥）
    match = pattern_short_cot.fullmatch(text)
    if match:
        groups = match.groupdict()
        return {
            "valid": True,
            "pattern": "short",
            "prefix": groups["prefix"],
            "cot": groups["cot"],
            "answer": groups["answer"]
        }

    match = pattern_long_cot.fullmatch(text)
    if match:
        groups = match.groupdict()
        return {
            "valid": True,
            "pattern": "long",
            "prefix": groups["prefix"],
            "cot": groups["cot"],
            "answer": groups["answer"]
        }

    match = pattern_answer.fullmatch(text)
    if match:
        groups = match.groupdict()
        return {
            "valid": True,
            "pattern": "low",
            "prefix": groups["prefix"],
            "cot": None,
            "answer": groups["answer"]
        }

    # 都不匹配
    return {
        "valid": False,
        "pattern": "invalid",
        "prefix": None,
        "cot": None,
        "answer": None
    }

def llm_json(content, llm_config):
    llm = llm_config["func"]
    max_retries = 3    
    for attempt in range(max_retries+1):
        try:
            llm_res = llm(content, llm_config)
            extracted_answer = extract_answer(llm_res)
            if not extracted_answer["valid"]:
                raise ValueError("LLM 输出格式无效，未能提取答案。")
            
            prefix = extracted_answer["prefix"]
            cot = extracted_answer["cot"]
            
            if cot:
                think = prefix + cot
            else:
                think = prefix

            answer = extracted_answer["answer"]
            final_dict = {
                "think": think.strip(),
                "answer": answer,
                "answer_raw": llm_res
            }
            return final_dict
        except Exception as e:
            print(f"处理记录时发生异常: {e}")
            if attempt < max_retries:
                print(f"第{attempt}次, 重试中...")
                print("错误内容:", llm_res)
            else:
                print("达到最大重试次数，跳过此记录。")
                return {"error": str(e),  "think": "","answer": "", "answer_raw": llm_res}


def load_jsonl(file_path):
    data_list = []
    if not os.path.exists(file_path):
        return data_list
    with open(file_path, "r", encoding="utf-8") as file:
        for line in file:
            try:
                data = json.loads(line.strip())
                data_list.append(data)
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON: {e}")
    return data_list


def save_jsonl(file_path, data_list, mode="w"):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, mode, encoding="utf-8") as file:
        for data in data_list:
            json_line = json.dumps(data, ensure_ascii=False)
            file.write(json_line + "\n")

en_prompt = """{instruction}{input}"""

cn_prompt = """{instruction}{input}"""

agentclinic_prompt = en_prompt
clinical_lab_diagnosis_prompt = """{instruction}

## 注意事项
- 请你罗列所有可能的初步诊断。

## 以下是相关信息
{input}
"""
clinical_lab_guidance_prompt = cn_prompt
diagnosis_cn_prompt = cn_prompt
diagnosis_en = en_prompt
medjourney_dp_prompt = cn_prompt
medjourney_dr_prompt = cn_prompt
medmcqa_prompt = en_prompt
medqa_cn_prompt = cn_prompt
medqa_us_prompt = en_prompt
mmlu_pro_prompt = en_prompt
pubmedqa_pqal_prompt = en_prompt
triage_cn_prompt = cn_prompt



def process_item(item, llm_config):
    item = deepcopy(item)
    
    dataset = item.get("dataset")
    instruction = item.get("instruction")
    input_content = item.get("input")
    prompt_map = {
        "agentclinic": agentclinic_prompt,
        "clinical_lab_diagnosis": clinical_lab_diagnosis_prompt,
        "clinical_lab_guidance": clinical_lab_guidance_prompt,
        "diagnosis-cn": diagnosis_cn_prompt,
        "diagnosis-en": diagnosis_en,
        "medjourney_dp": medjourney_dp_prompt,
        "medjourney_dr": medjourney_dr_prompt,
        "medmcqa": medmcqa_prompt,
        "medqa-cn": medqa_cn_prompt,
        "medqa-us": medqa_us_prompt,
        "mmlu-pro": mmlu_pro_prompt,
        "pubmedqa-pqal": pubmedqa_pqal_prompt,
        "triage-cn": triage_cn_prompt,
    }
    prompt = prompt_map.get(dataset).format(
        instruction=instruction, input=input_content
    )
    
    llm_res = llm_json(prompt, llm_config)
    item["llm_res"] = llm_res

    return item


def main():
    input_list = load_jsonl(input_file)

    for llm_config in LLM_CONFIGS:
        client = OpenAI(
            api_key=llm_config["api_key"],
            base_url=llm_config["base_url"],
        )
        llm_config["client"] = client

        res_list = []
        res_buffer = []

        # don't need
        done_list = []
        done_ids = set()

        try:
            output_list = load_jsonl(output_file)
            
            for item in output_list:
                if item.get("llm_res", "") and item["id"] not in done_ids:
                    # and "error" not in item["llm_res"]
                    done_list.append(item)
                    done_ids.add(item["id"])

            data_list = [
                item
                for item in input_list
                if item["id"] not in done_ids
            ]


            print(f"[INFO] 总计: {len(input_list)}")
            print(f"[INFO] 完成: {len(done_ids)}")
            print(f"[INFO] 待处理: {len(data_list)}")

            res_list = done_list

            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                future_to_data = {
                    executor.submit(process_item, data, llm_config): data
                    for data in data_list
                }
                for future in tqdm(
                    as_completed(future_to_data), total=len(future_to_data)
                ):
                    try:
                        result = future.result()
                        res_buffer.append(result)

                        if len(res_buffer) >= BATCH_SIZE:
                            mode = "a" if os.path.exists(output_file) else "w"
                            res_list.extend(res_buffer)
                            save_jsonl(output_file, res_buffer, mode)
                            res_buffer.clear()
                    except Exception as exc:
                        print(f"[ERROR] 处理任务出错: {exc}")

            if res_buffer:
                res_list.extend(res_buffer)
            res_list = sorted(res_list, key=lambda x: x["id"])
            save_jsonl(output_file, res_list, mode="w")
            print(f"✅ {file_name}已完成")
        except KeyboardInterrupt:
            print("\n[WARNING] 检测到 Ctrl+C 中断请求，正在尝试保存当前进度...")
            if res_buffer:
                res_list.extend(res_buffer)
            res_list = sorted(res_list, key=lambda x: x["id"])
            save_jsonl(output_file, res_list, mode="w")
            print(f"[INFO] 已保存 {len(res_list)} 条数据。")
            sys.exit(0)


if __name__ == "__main__":
    main()
