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


def merge_adversarial_data(original_data_path, adversarial_data_paths, min_quality_score=0.7):
    """
    合并原始角色数据与对抗生成的优质数据
    :param original_data_path: 原始角色训练数据路径
    :param adversarial_data_paths: 对抗生成数据路径列表
    :param min_quality_score: 最低质量分数阈值（来自判别器评估）
    :return: 合并并去重后的数据集
    """
    # 加载原始数据
    with open(original_data_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)

    # 加载并筛选对抗生成数据
    adversarial_data = []
    for path in adversarial_data_paths:
        with open(path, 'r', encoding='utf-8') as f:
            gen_data = json.load(f)
            # 筛选高质量生成数据（假设数据中包含quality_score字段）
            filtered = [item for item in gen_data if item.get('quality_score', 0) >= min_quality_score]
            adversarial_data.extend(filtered)

    # 合并并去重（按input字段）
    combined_dict = {item["input"]: item for item in original_data + adversarial_data}
    combined_data = list(combined_dict.values())

    print(f"合并完成：原始数据{len(original_data)}条，优质对抗数据{len(adversarial_data)}条，合并后{len(combined_data)}条")
    return combined_data


def train_elysia_model(physics_model_path, train_data_path, output_dir, adversarial_data_paths=None):
    # 加载已训练的物理解题模型
    model = AutoModelForCausalLM.from_pretrained(
        physics_model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(physics_model_path)

    # 如果提供了对抗数据路径，则合并数据
    if adversarial_data_paths and len(adversarial_data_paths) > 0:
        merged_data = merge_adversarial_data(train_data_path, adversarial_data_paths)
        # 保存合并后的数据用于后续查看
        merged_data_path = os.path.join(output_dir, "merged_training_data.json")
        with open(merged_data_path, 'w', encoding='utf-8') as f:
            json.dump(merged_data, f, ensure_ascii=False, indent=2)
        # 将合并后的数据转换为数据集
        dataset = load_dataset("json", data_files=merged_data_path)["train"]
    else:
        # 直接加载原始训练数据
        dataset = load_dataset("json", data_files=train_data_path)["train"]

    # 配置LoRA参数（针对角色微调优化）
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=8,  # 角色微调使用较大的秩以更好捕捉风格特征
        lora_alpha=32,
        lora_dropout=0.15,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # 增加更多模块以更好学习角色风格
        bias="none",
        inference_mode=False
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()  # 打印可训练参数比例

    # 数据预处理：强化爱莉希雅风格提示
    def preprocess_function(examples):
        # 更具体的角色引导提示
        inputs = [f"作为崩坏三的爱莉希雅，用活泼优雅的语气解析物理题，要包含完整逻辑链和角色特征：{q}"
                  for q in examples["input"]]
        model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding="max_length")

        # 处理输出，确保包含角色风格标记
        outputs = [f"~♪ 爱莉希雅的解析时间到啦～ {ans}" for ans in examples["output"]]
        labels = tokenizer(outputs, max_length=1024, truncation=True, padding="max_length")
        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True, remove_columns=dataset.column_names)

    # 加载训练参数
    with open("../../config/training_args.json", 'r') as f:
        training_args_dict = json.load(f)["elysia_train"]

    # 调整训练参数以适应对抗数据
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

    # 保存模型权重和训练配置
    model.save_pretrained(output_dir)
    with open(os.path.join(output_dir, "peft_config.json"), 'w') as f:
        json.dump(peft_config.to_dict(), f, indent=2)
    print(f"角色化模型已保存至 {output_dir}")

    return output_dir


if __name__ == "__main__":
    # 对抗生成的数据路径列表（可根据实际训练轮次调整）
    adversarial_data_paths = [
        "../../model/adversarial/generator_loop_0/fake_data.json",
        "../../model/adversarial/generator_loop_1/fake_data.json",
        "../../model/adversarial/generator_loop_2/fake_data.json"
    ]

    # 过滤存在的文件路径
    valid_adversarial_paths = [p for p in adversarial_data_paths if os.path.exists(p)]

    train_elysia_model(
        physics_model_path="../../model/physics_lora",
        train_data_path="../../data/processed/elysia_train",
        output_dir="../../model/elysia_adv_lora",
        adversarial_data_paths=valid_adversarial_paths  # 传入有效的对抗数据路径
    )
