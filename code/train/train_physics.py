import os
import json
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import get_peft_model, LoraConfig, TaskType


def train_physics_model(base_model_path, train_data_path, output_dir):
    # 加载基础模型
    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)

    # 加载训练数据
    dataset = load_dataset("json", data_files=train_data_path)["train"]

    # 配置LoRA参数（仅训练部分层）
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]  # 针对ChatGLM-6B
    )

    model = get_peft_model(model, peft_config)

    # 数据预处理
    def preprocess_function(examples):
        inputs = [f"解析物理题：{q}" for q in examples["input"]]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True)

        # 添加标签（与输入相同，用于自回归训练）
        labels = tokenizer(examples["output"], max_length=1024, truncation=True)
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # 加载训练参数
    with open("../../config/training_args.json", 'r') as f:
        training_args_dict = json.load(f)["physics_train"]

    training_args = TrainingArguments(
        output_dir=output_dir,
        **training_args_dict
    )

    # 初始化训练器
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    # 开始训练
    trainer.train()

    # 保存模型权重
    model.save_pretrained(output_dir)
    print(f"物理解题模型已保存至 {output_dir}")


if __name__ == "__main__":
    train_physics_model(
        base_model_path="../../model/base",
        train_data_path="../../data/processed/physics_train",
        output_dir="../../model/physics_lora"
    )