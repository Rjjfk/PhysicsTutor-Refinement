import json
from transformers import AutoTokenizer
from datasets import Dataset

# 加载ChatGLM-6B分词器（与模型保持一致）
tokenizer = AutoTokenizer.from_pretrained("../model/base/", trust_remote_code=True)

# 读取教师思维链JSON
with open("../data/raw/teacher_chain.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# 转换为模型训练样本（input-output对）
train_samples = []
for item in raw_data:
    # 输入：物理题描述
    input_text = f"解析物理题：{item['物理题描述']}"
    # 输出：完整教师思维链（含阶段、易错点）
    output_text = "\n".join([
        f"【{stage['阶段']}】{stage['思考步骤']}\n知识锚点：{stage['知识锚点']}\n易错点：{stage['学生易错点']}"
        for stage in item['教师思维链']
    ])

    # Tokenize并截断（控制长度，避免显存爆炸）
    tokenized = tokenizer(
        input_text,
        output_text,
        max_length=1024,  # 适配ChatGLM-6B上下文
        truncation=True,
        padding="max_length"
    )
    train_samples.append({
        "input_ids": tokenized["input_ids"],
        "labels": tokenized["input_ids"]  # 因果语言模型，输入=输出（自回归训练）
    })

# 保存为Hugging Face Dataset
dataset = Dataset.from_list(train_samples)
dataset.save_to_disk("../data/processed/train_dataset")

print(f"✅ 成功转换 {len(train_samples)} 条样本，保存至 ../data/processed/train_dataset")