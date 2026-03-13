import pandas as pd
import json

# 讀取 Excel 檔案
excel_path = "C:/Users/keysi/Desktop/HW2-llama3/AI.xlsx"
df = pd.read_excel(excel_path)

# 轉換選項和正確答案的格式
df["選項1"] = df["選項1"].astype(str)
df["選項2"] = df["選項2"].astype(str)
df["選項3"] = df["選項3"].astype(str)
df["選項4"] = df["選項4"].astype(str)

# 假設 convert_output 是一個將正確答案轉換為適當格式的函數
def convert_output(answer):
    try:
        return int(answer)
    except ValueError:
        return -1

df["正確答案"] = df["正確答案"].apply(convert_output)
df = df[df["正確答案"] != -1]
df = df.drop(columns=["資料來源"])

### Step 2: 格式化數據
# 定義 Alpaca Prompt 模板

# 定義 EOS_TOKEN
from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("FlagAlpha/Llama3-Chinese-8B-Instruct")
EOS_TOKEN = tokenizer.eos_token

def formatting_prompts_func(examples):
    texts = []
    for article, question, option1, option2, option3, option4, answer in zip(
        examples["文章"], 
        examples["問題"], 
        examples["選項1"], 
        examples["選項2"], 
        examples["選項3"], 
        examples["選項4"], 
        examples["正確答案"]
    ):
        instruction = f"請閱讀文章並回答選擇題，只輸出選項的數字\n文章: {article}\n問題: {question}\n選項1: {option1}\n選項2: {option2}\n選項3: {option3}\n選項4: {option4}"
        #input_text = f""
        #response = ""
        #text = alpaca_prompt.format(instruction, input_text, response) + EOS_TOKEN
        
        texts.append(instruction)
    return {"text": texts}

# 格式化數據
formatted_data = formatting_prompts_func(df)

### Step 3: 將數據寫入 JSON 文件
# 定義輸出 JSON 文件路徑
output_json_path = "formatted_data.json"

# 寫入 JSON 文件
with open(output_json_path, 'w', encoding='utf-8') as f:
    json.dump(formatted_data, f, ensure_ascii=False, indent=4)

print(f"Data successfully written to {output_json_path}")
