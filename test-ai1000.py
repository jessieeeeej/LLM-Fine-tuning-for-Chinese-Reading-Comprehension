import os
import pandas as pd
from unsloth import FastLanguageModel
import torch
from transformers import TextStreamer
from datasets import Dataset

# 定義格式化函數
def formatting_prompts_func(examples):
    texts = []
    for article, question, option1, option2, option3, option4 in zip(
        examples["文章"], 
        examples["問題"], 
        examples["選項1"], 
        examples["選項2"], 
        examples["選項3"], 
        examples["選項4"] 
        #examples["正確答案"], 
        #examples["資料來源"]
    ):
        instruction = f"Please read the article and answer the following multiple-choice question with a number\nQuestion: {question}\n1: {option1}\n2: {option2}\n3: {option3}\n4: {option4}"
        input_text = f"Article: {article}\n"
        response = ""
        text = alpaca_prompt.format(instruction, input_text, response) + EOS_TOKEN
        texts.append(text)
    return { "text": texts }

# 讀取 Excel 檔案
excel_path = "C:/Users/keysi/Desktop/HW2-llama3/AI1000.xlsx"
df = pd.read_excel(excel_path)
df["選項1"] = df["選項1"].astype(str)
df["選項2"] = df["選項2"].astype(str)
df["選項3"] = df["選項3"].astype(str)
df["選項4"] = df["選項4"].astype(str)
#df["正確答案"] = df["正確答案"].astype(str)
dataset = Dataset.from_pandas(df)

# 設定 Alpaca 格式
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{}
### Input:
{}
### Response:
{}"""

# 設定模型與 Tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "lora_model3", # 加載訓練後的LoRA模型
    max_seq_length = 2048,
    dtype = None,
    load_in_4bit = True,
)
tokenizer.padding_side = "left"
EOS_TOKEN = tokenizer.eos_token # 必須添加 EOS_TOKEN

# 應用格式化函數
formatted_dataset = dataset.map(formatting_prompts_func, batched=True)

FastLanguageModel.for_inference(model)

# 定義 TextStreamer
text_streamer = TextStreamer(tokenizer)

# 生成文本
for example in formatted_dataset["text"]:
    inputs = tokenizer([example], return_tensors="pt", padding=True, truncation=True).to("cuda")
    outputs = model.generate(**inputs, max_new_tokens=2)  # 限制輸出長度
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # 確保輸出只包含數字
    answer = ''.join(filter(str.isdigit, answer))
    print(answer)
