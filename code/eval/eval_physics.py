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


def train_elysia_model(physics_model_path, train_data_path, output_dir):
    # 加载已训练的物理解题模型
    model = AutoModelForCausalLM.from_pretrained(
        physics_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(physics_model_path)

    # 加载角色化训练数据
    dataset = load_dataset("json", data_files=train_data_path)["train"]

    # 配置LoRA参数（仅训练顶层）
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=4,  # 比物理训练更小，微调风格而非知识
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )

    model = get_peft_model(model, peft_config)

    # 数据预处理
    def preprocess_function(examples):
        inputs = [f"用爱莉希雅的风格解析：{q}" for q in examples["input"]]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True)

        labels = tokenizer(examples["output"], max_length=1024, truncation=True)
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True)

    # 加载训练参数
    with open("../../config/training_args.json", 'r') as f:
        training_args_dict = json.load(f)["elysia_train"]

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
    print(f"角色化模型已保存至 {output_dir}")


if __name__ == "__main__":
    train_elysia_model(
        physics_model_path="../../model/physics_lora",
        train_data_path="../../data/processed/elysia_train",
        output_dir="../../model/elysia_lora"
    )