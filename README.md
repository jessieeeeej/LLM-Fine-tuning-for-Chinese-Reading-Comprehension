## Environment
- Python 3.11.8
- OS: Windows 11
- GPU: NVIDIA GeForce RTX 4060 Ti
- 使用的Python套件：
    - torch
    - transformers
    - datasets
    - pandas
    - unsloth

## Model description
- basic模型:
    1. **unsloth/llama-3-8b-Instruct-bnb-4bit**
        - 基於LLama架構進行的微調，專門針對指令生成進行優化
        - 支持4-bit精度，加載速度快，適合快速微調
    2. **llamafamily/llama3-chinese-8b-instruct**
        - 專為中文指令生成優化的LLama模型
        - 針對中文語言進行了更深度的微調，能夠更準確地生成符合語境的中文回應
          
- dataset: 包含13000多筆的data

## Fine-tuning
1. 模型：
    - `unsloth/llama-3-8b-Instruct-bnb-4bit` 和 `llamafamily/llama3-chinese-8b-instruct`
    - 最大序列長度：2048
    - 4-bit參數
```python
max_seq_length = 2048
dtype = None
load_in_4bit = True
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit", 
    max_seq_length = max_seq_length, 
    dtype = dtype,     
    load_in_4bit = load_in_4bit,
)
```

2. 準備訓練data：
    - 定義格式，將文章、問題、選項和正確答案組合成訓練樣本
    - 從Excel檔案中讀取數據，並將每個選項轉換為字串格式
    - 使用格式函數將數據集轉換為模型可用的格式
```python
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{}
### Input:
{}
### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token  # 必須添加 EOS_TOKEN
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
        instruction = f"Please read the article and answer the following multiple-choice question with a number\nQuestion: {question}\n1: {option1}\n2: {option2}\n3: {option3}\n4: {option4}"
        input_text = f"Article: {article}\n"
        response = str(answer)
        text = alpaca_prompt.format(instruction, input_text, response) + EOS_TOKEN
        texts.append(text)
    return {"text": texts}

df = pd.read_excel(excel_path)
df["選項1"] = df["選項1"].astype(str)
df["選項2"] = df["選項2"].astype(str)
df["選項3"] = df["選項3"].astype(str)
df["選項4"] = df["選項4"].astype(str)
df["正確答案"] = df["正確答案"].astype(str)

dataset = Dataset.from_pandas(df)
formatted_dataset = dataset.map(formatting_prompts_func, batched=True)
```

3. 訓練：
    - 使用SFTTrainer進行訓練
    - 訓練數據集和評估數據集均為格式化後的數據集
```python
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=100,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=formatted_dataset,
    eval_dataset=formatted_dataset,
)

trainer.train()
```

## Evaluate
1. 模型：
    - 訓練後的LoRA模型(lora_model)

2. 準備data：
    - 從Excel檔案中讀取。將每個選項轉換為字串格式
    - 使用格式化函數將數據集轉換為模型可用的格式

3. 定義生成文本的設定：
    - 最大序列長度：2048
    - 定義結束標誌(EOS_TOKEN)

## 心得
- 問題: `unsloth/llama-3-8b-Instruct-bnb-4bit`微調後的模型輸出無法控制在單一選項數值，容易會包含其他文字輸出。可能是因為模型設計主要是為了快速微調，且基礎模型是英文的，即使訓練數據集有13000多筆對模型來說依然不夠多，雖然改用英文下指令有提升一點點，訓練時損loss值一直卡在2.0左右，並且測試訓練好的模型輸出還是很難穩定

- 解決: 改用`llamafamily/llama3-chinese-8b-instruct`生成答案效果明顯提升
    - 此模型是專為中文優化的模型，能夠更好地理解和生成中文內容，避免了由於語言差異導致的上下文理解錯誤
    - 該模型在更大量的中文dataset上進行過微調，能更好理解中文語意，提升生成答案的準確性
    - 相比通用的快速微調模型，該模型進行了更深的微調，在處理選擇題這類任務時能夠提供更準確的輸出


