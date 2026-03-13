import os
#os.environ["CC"] = "gcc"
from unsloth import FastLanguageModel
import torch
import pandas as pd
from trl import SFTTrainer
from transformers import TrainingArguments, get_scheduler, EarlyStoppingCallback
from datasets import load_dataset, Dataset

#加载模型
max_seq_length = 2048
dtype = None
load_in_4bit = True
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/llama-3-8b-Instruct-bnb-4bit", 
    #model_name = "unsloth/llama-3-8b-bnb-4bit", 
    max_seq_length = max_seq_length, 
    dtype = dtype,     
    load_in_4bit = load_in_4bit,
)

def convert_output(output):
    try:
        return int(output)
    except ValueError:
        return -1

#准备训练数据
alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction:
{}
### Input:
{}
### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # 必须添加 EOS_TOKEN
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
    print(texts[0])
    return {"text": texts}

#hugging face数据集路径
excel_path = "C:/Users/keysi/Desktop/HW2-llama3/AI.xlsx"
df = pd.read_excel(excel_path)
df["選項1"] = df["選項1"].astype(str)
df["選項2"] = df["選項2"].astype(str)
df["選項3"] = df["選項3"].astype(str)
df["選項4"] = df["選項4"].astype(str)
df["正確答案"] = df["正確答案"].apply(convert_output)
df = df[df["正確答案"] != -1] 
df = df.drop(columns=["資料來源"])
"""
print(df.head(2))
dataset = Dataset.from_pandas(df)
dataset = dataset.map(formatting_prompts_func, batched = True,)

#设置训练参数
model = FastLanguageModel.get_peft_model(
    model,
    r = 32,     # lora_rank
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, 
    bias = "none",    
    use_gradient_checkpointing = True,
    random_state = 3407,
    max_seq_length = max_seq_length,
    use_rslora = False,  
    loftq_config = None, 
)

trainer = SFTTrainer(
    model = model,
    train_dataset = dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    tokenizer = tokenizer,
    args = TrainingArguments(
        per_device_train_batch_size = 16,
        gradient_accumulation_steps = 4,
        warmup_steps = 10,
        max_steps = 200,
        learning_rate = 1e-4,
        fp16 = not torch.cuda.is_bf16_supported(),
        bf16 = torch.cuda.is_bf16_supported(),
        logging_steps = 1,
        output_dir = "outputs3",
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
    ),
)
#开始训练
trainer.train()

#保存微调模型
model.save_pretrained("lora_model3") 

#合并模型，保存为16位hf
model.save_pretrained_merged("outputs3", tokenizer, save_method = "merged_16bit",)

#合并模型，并量化成4位gguf
#model.save_pretrained_gguf("model", tokenizer, quantization_method = "q4_k_m")
"""

# 拆分數據集
dataset = Dataset.from_pandas(df)
dataset = dataset.train_test_split(test_size=0.1)
train_dataset = dataset['train']
eval_dataset = dataset['test']

# 格式化數據
train_dataset = train_dataset.map(formatting_prompts_func, batched=True)
eval_dataset = eval_dataset.map(formatting_prompts_func, batched=True)

# 設置訓練參數
model = FastLanguageModel.get_peft_model(
    model,
    r=32,  # lora_rank
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing=True,
    random_state=3407,
    max_seq_length=max_seq_length,
    use_rslora=False,
    loftq_config=None
)

training_args = TrainingArguments(
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,  # 增加梯度累積步數
    warmup_steps=10,  # 增加熱身步數
    max_steps=50,  # 增加訓練步數
    learning_rate=5e-5,  # 保持較低的學習率
    fp16=not torch.cuda.is_bf16_supported(),
    bf16=torch.cuda.is_bf16_supported(),
    logging_steps=5,  # 增加log頻率
    output_dir="outputs4",
    optim="adamw_8bit",
    weight_decay=0.01,
    evaluation_strategy="steps",  # 評估策略
    eval_steps=10,  # 設置評估間隔
    seed=3407,
    max_grad_norm=1.0,  # 設置梯度剪裁
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    #eval_dataset=eval_dataset,
    dataset_text_field="text",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_args,
    #callbacks=[EarlyStoppingCallback(early_stopping_patience=10)]
)


# 開始訓練
trainer.train()

# 保存微調模型
model.save_pretrained("./unsloth/lora_model4")
tokenizer.save_pretrained("./unsloth/lora_model4")

# 合併模型，保存為16位hf
model.save_pretrained_merged("./unsloth/outputs4", tokenizer, save_method="merged_16bit")