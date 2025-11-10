from openai import OpenAI
import json
import os
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
from copy import deepcopy
import sys
import re
from threading import Semaphore
import shutil
import sys
import time

def llm_base(content, llm_config):
    client = llm_config["client"]
    model_name = llm_config["model_name"]
    try:
        completion = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": content}],
            temperature=0.01,
            extra_body={"chat_template_kwargs": {"thinking": False}}
        )

        think = ""
        answer = completion.choices[0].message.content
        
        return think, answer

    except Exception as e:
        print(f" API 调用失败: {e}")
        raise e

BATCH_SIZE = 2000
MAX_WORKERS = 64

LLM_CONFIGS = [
    {
        "model_name": "qwen/qwen3-235b-a22b",
        "api_key": "XXX",  # 替换为你的实际 API Key
        "base_url": "XXX",
        "func": llm_base,
    }
]

filenames = [
    "XXX",
    "XXX2"
]

input_file_template = "/XXX/llm_res/{filename}.jsonl"
output_file_template = (
    "/XXX/llm_eval/{filename}.jsonl"
)


def extract_json(text):
    start = text.find('{')
    if start == -1:
        raise ValueError("未找到 '{'")
    depth = 0
    for i in range(start, len(text)):
        if text[i] == '{':
            depth += 1
        elif text[i] == '}':
            depth -= 1
            if depth == 0:
                candidate = text[start:i+1]
                try:
                    candidate_json = json.loads(candidate)
                    return candidate_json
                except json.JSONDecodeError:
                    print("JSONDecodeError for candidate:", candidate)
                    continue  # 继续找下一个可能的结束
    raise ValueError("未找到合法的 JSON 对象")

def llm_json(content, llm_config):
    llm = llm_config["func"]
    max_retries = 5
    think, answer = "", ""
    for attempt in range(max_retries+1):
        try:
            _, answer = llm(content, llm_config)
            res_json = extract_json(answer)
            return res_json
        except Exception as e:
            print(f"处理记录时发生异常: {e}")
            if attempt < max_retries:
                print(f"第{attempt}次, 重试中...")
                print("错误内容:", answer)
            else:
                print("达到最大重试次数，跳过此记录。")
                return {"error": str(e)}

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

agentclinic_prompt = """You are a professional physician. Based on the resident doctor's response, extract the diagnosis they consider most likely.

## Output Format
- Please only output a single JSON object, do not output any other content.
- The output JSON object must be on a single line, and all newline characters in its string values must be escaped as \\n.
- The JSON object you output must be able to be correctly parsed by json.loads().
{{
    "diagnosis": "Most likely diagnosis"
}}

## Below is the resident doctor's response
{llm_answer}
"""

clinical_lab_diagnosis_prompt = """
你是一名专业的医生，请你将参考初步诊断，与实习医生的初步诊断，各自整理为一个列表。

## 输出格式
- 请你仅输出一个 JSON 对象，不要输出其他内容。
- 输出的 JSON 对象必须是单行的，其字符串值中的所有换行符都必须被转义为 \\n。
- 请你输出的 JSON 对象可以被 json.loads() 正确解析。
{{
    "reference_diagnosis": ["诊断1", "诊断2", ...], 
    "resident_diagnosis": ["诊断1", "诊断2",...]
}}

## 以下是参考初步诊断
{reference_answer}

## 以下是实习医生的初步诊断
{llm_answer}
"""

diagnosis_cn_prompt = """
你是一名专业的医生，请你将参考答案的诊断和鉴别诊断，与实习医生的诊断和鉴别诊断，各自整理为一个列表。

## 输出格式
- 请你仅输出一个 JSON 对象，不要输出其他内容。
- 输出的 JSON 对象必须是单行的，其字符串值中的所有换行符都必须被转义为 \\n。
- 请你输出的 JSON 对象可以被 json.loads() 正确解析。
{{
    "reference_diagnosis": ["诊断1",...],
    "reference_differential_diagnosis": ["鉴别诊断1", "鉴别诊断2", ...],
    "resident_diagnosis": ["诊断1",...],
    "resident_differential_diagnosis": ["鉴别诊断1", "鉴别诊断2", ...]
}}

## 以下是参考诊断和鉴别诊断
{reference_answer}

## 以下是实习医生的诊断和鉴别诊断
{llm_answer}
"""

diagnosis_en_prompt = """You are a professional physician. Please organize the diagnoses from the reference answer and the resident doctor into separate lists.

## Output Format
- Please only output a single JSON object, do not output any other content.
- The output JSON object must be on a single line, and all newline characters in its string values must be escaped as \\n.
- The JSON object you output must be able to be correctly parsed by json.loads().
{{
    "reference_diagnosis": ["Diagnosis 1", ...],
    "resident_diagnosis": ["Diagnosis 1", ...]
}}

## Here is the reference diagnosis
{reference_answer}

## Here is the resident doctor's diagnosis
{llm_answer}
"""

medjourney_dr_prompt = """你是一名专业的医生，请你根据参考答案，评估实习医生所给的导诊科室是否正确。如果正确，回答{{"judge": "correct"}}，如果不正确，回答{{"judge": "error"}}。语义正确即可。

## 输出格式
- 请你仅输出一个 JSON 对象，不要输出其他内容。
- 输出的 JSON 对象必须是单行的，其字符串值中的所有换行符都必须被转义为 \\n。
- 请你输出的 JSON 对象可以被 json.loads() 正确解析。
{{"judge": "correct" or "error"}}

## 以下是参考答案
{reference_answer}

## 以下是实习医生的导诊科室
{llm_answer}
"""



def process_item(item, llm_config):
    item = deepcopy(item)
    llm_config = deepcopy(llm_config)
    
    dataset = item.get("dataset")

    try:
        client = OpenAI(
            api_key=llm_config["api_key"],
            base_url=llm_config["base_url"],
        )
        llm_config["client"] = client
        

        llm_answer = None
        if "error" not in item.get("llm_res", {}):
            llm_answer = item["llm_res"]["answer"]
        else:
            llm_answer = item["llm_res"].get("answer_raw", "")
            
        reference_answer = item.get("output", "")
            
        prompt = None
        if dataset in ["agentclinic"]:
            prompt = agentclinic_prompt.format(
                llm_answer=llm_answer,
            )
        elif dataset in ["clinical_lab_diagnosis"]:
            prompt = clinical_lab_diagnosis_prompt.format(
                reference_answer=reference_answer,
                llm_answer=llm_answer,
            )
        elif dataset in ["diagnosis-cn"]:
            prompt = diagnosis_cn_prompt.format(
                reference_answer=reference_answer,
                llm_answer=llm_answer,
            )
        elif dataset in ["diagnosis-en"]:
            prompt = diagnosis_en_prompt.format(
                reference_answer=reference_answer,
                llm_answer=llm_answer,
            )
        elif dataset in ["medjourney_dr"]:
            prompt = medjourney_dr_prompt.format(
                reference_answer=reference_answer,
                llm_answer=llm_answer,
            )
            
        llm_eval = llm_json(prompt, llm_config)
        item["llm_eval"] = llm_eval
        return item
    except Exception as e:
        print(f"[ERROR] 处理记录时发生异常: {e}")
        return item


def main():

    for filename in filenames:
        llm_config = deepcopy(LLM_CONFIGS[0])
        input_file = input_file_template.format(filename=filename)
        output_file = output_file_template.format(filename=filename)
        input_list = load_jsonl(input_file)
        
        res_list = []
        res_buffer = []

        done_list = []
        done_ids = set()

        try:
            todo_dataset = [
                "agentclinic",
                "clinical_lab_diagnosis",
                "diagnosis-cn",
                "diagnosis-en",
                "medjourney_dr"
            ]
            
            for item in input_list:
                if item["dataset"] not in todo_dataset:
                    done_list.append(item)
                    done_ids.add(item["id"])
            
            output_list = load_jsonl(output_file)

            for item in output_list:
                if item["dataset"] in todo_dataset and item.get("llm_eval",{})  and "error" not in item.get("llm_eval",{}) and item["id"] not in done_ids:
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
            print(f"✅ 已完成 {filename}")
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
