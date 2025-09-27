# code/train/train_merge.py（核心代码框架）
import os
import json
import torch
import argparse
from datasets import load_from_disk, concatenate_datasets
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
from peft import PeftModel, LoraConfig, get_peft_model
from train_physics import TrainingLogger, TrainingStateManager  # 复用日志组件


def load_dual_lora_models(physics_model_path, role_model_path, base_model_path):
    """加载物理和角色的LoRA权重，合并到基础模型"""
    # 1. 加载基础模型
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path, torch_dtype=torch.float16, device_map="auto"
    )
    # 2. 先加载物理LoRA
    model = PeftModel.from_pretrained(base_model, physics_model_path)
    # 3. 再加载角色LoRA（关键：保留物理权重，叠加角色权重）
    model = PeftModel.from_pretrained(model, role_model_path)
    return model


def train_merge(args):
    logger = TrainingLogger(args.log_file)
    logger.log("===== 开始物理+角色合并微调 =====")

    # 1. 加载配置
    with open(args.config_path, "r") as f:
        config = json.load(f)["merge_train"]

    # 2. 加载双模型权重
    model = load_dual_lora_models(
        physics_model_path=config["physics_self_model_path"],
        role_model_path=config["role_self_model_path"],
        base_model_path=config["base_model_path"]
    )
    tokenizer = AutoTokenizer.from_pretrained(config["base_model_path"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. 准备混合训练数据（物理题+角色对话，比例7:3）
    physics_dataset = load_from_disk(config["physics_data_path"])
    role_dataset = load_from_disk(config["role_data_path"])
    # 按比例采样，确保数据量均衡
    physics_sample = physics_dataset.shuffle().select(range(min(3000, len(physics_dataset))))
    role_sample = role_dataset.shuffle().select(range(int(len(physics_sample) * 0.3)))
    merged_dataset = concatenate_datasets([physics_sample, role_sample]).shuffle()

    # 4. 配置合并微调参数（极低学习率，仅1轮）
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=config["batch_size"],
        num_train_epochs=1,  # 仅微调1轮，避免破坏平衡
        learning_rate=3e-5,  # 比自训练更低
        logging_steps=5,
        fp16=True,
        report_to="none"
    )

    # 5. 微调（冻结大部分层，仅微调交叉注意力层）
    model = get_peft_model(model, LoraConfig(
        task_type="CAUSAL_LM",
        r=2,
        lora_alpha=8,
        target_modules=["q_proj", "v_proj"],  # 交叉注意力层，负责特征融合
        lora_dropout=0.05
    ))
    logger.log(f"合并微调可训练参数：{model.print_trainable_parameters()}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=merged_dataset,
        tokenizer=tokenizer
    )
    trainer.train()

    # 6. 保存最终模型
    model.save_pretrained(args.output_dir)
    logger.log(f"合并微调完成，模型保存至：{args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--log_file", type=str, required=True)
    parser.add_argument("--config_path", type=str, default="../../config/merge_config.json")
    args = parser.parse_args()
    train_merge(args)