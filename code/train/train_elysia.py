import os
import json
import time
import argparse
import torch
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import get_peft_model, LoraConfig, TaskType


# --------------------------
# 新增：日志工具类（与后端对接）
# --------------------------
class TrainingLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.log("训练日志初始化完成", "info")

    def log(self, content, log_type="info"):
        """记录日志：终端打印 + 文件写入"""
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        log_line = f"{timestamp} [{log_type.upper()}] {content}"
        print(log_line)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")


# --------------------------
# 新增：训练状态管理类（与前端交互）
# --------------------------
class TrainingStateManager:
    def __init__(self, state_file, total_epochs):
        self.state_file = state_file
        self.state = {
            "status": "init",  # init/training/paused/completed/failed
            "epoch": 0,
            "total_epochs": total_epochs,
            "current_loss": 0.0,
            "best_loss": float("inf"),
            "physics_accuracy": 0.0,
            "role_accuracy": 0.0,
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_time": "00:00:00",
            "epoch_history": []
        }
        self.save()

    def update(self, epoch, current_loss, physics_acc=0.0, role_acc=0.0):
        """更新训练状态"""
        self.state["status"] = "training"
        self.state["epoch"] = epoch
        self.state["current_loss"] = round(current_loss, 4)
        self.state["best_loss"] = round(min(self.state["best_loss"], current_loss), 4)
        self.state["physics_accuracy"] = round(physics_acc, 2)
        self.state["role_accuracy"] = round(role_acc, 2)

        # 计算耗时
        start = datetime.strptime(self.state["start_time"], "%Y-%m-%d %H:%M:%S")
        elapsed = datetime.now() - start
        self.state["elapsed_time"] = str(elapsed).split(".")[0]

        # 记录轮次历史
        self.state["epoch_history"].append({
            "epoch": epoch,
            "loss": self.state["current_loss"],
            "physics_accuracy": physics_acc,
            "role_accuracy": role_acc,
            "time": self.state["elapsed_time"]
        })

        self.save()

    def set_status(self, status):
        """设置训练状态（如完成/失败）"""
        self.state["status"] = status
        self.save()

    def save(self):
        """保存状态到文件"""
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)


# --------------------------
# 新增：训练回调类（跟踪训练进度）
# --------------------------
class ElysiaTrainingCallback(TrainerCallback):
    def __init__(self, state_manager, logger):
        self.state_manager = state_manager
        self.logger = logger

    def on_epoch_end(self, args, state, control, **kwargs):
        """每轮结束时更新状态"""
        if state.epoch is not None:
            current_epoch = int(state.epoch)
            current_loss = state.log_history[-1].get("loss", 0.0)
            self.state_manager.update(current_epoch, current_loss)
            self.logger.log(f"轮次 {current_epoch} 结束，当前损失: {current_loss:.4f}")


# --------------------------
# 原有核心功能：数据合并（保留并优化）
# --------------------------
def merge_adversarial_data(original_data_path, adversarial_data_paths, min_quality_score=0.7, logger=None):
    """合并原始角色数据与对抗生成的优质数据"""
    log = logger.log if logger else print

    # 加载原始数据
    if not os.path.exists(original_data_path):
        raise FileNotFoundError(f"原始数据文件不存在: {original_data_path}")

    with open(original_data_path, 'r', encoding='utf-8') as f:
        original_data = json.load(f)
    log(f"加载原始角色数据: {len(original_data)} 条")

    # 加载并筛选对抗生成数据
    adversarial_data = []
    for path in adversarial_data_paths:
        if not os.path.exists(path):
            log(f"对抗数据文件不存在，跳过: {path}", "warning")
            continue

        with open(path, 'r', encoding='utf-8') as f:
            gen_data = json.load(f)

        # 筛选高质量生成数据
        filtered = [item for item in gen_data if item.get('quality_score', 0) >= min_quality_score]
        adversarial_data.extend(filtered)
        log(f"加载对抗数据 {path}: 原始{len(gen_data)}条，筛选后{len(filtered)}条")

    # 合并并去重（按input字段）
    combined_dict = {item["input"]: item for item in original_data + adversarial_data}
    combined_data = list(combined_dict.values())

    log(f"数据合并完成：原始{len(original_data)}条 + 对抗{len(adversarial_data)}条 → 去重后{len(combined_data)}条")
    return combined_data


# --------------------------
# 主训练函数（整合状态管理）
# --------------------------
def train_elysia_model(
        physics_model_path,
        train_data_path,
        output_dir,
        state_file,
        log_file,
        adversarial_data_paths=None,
        config_path=None
):
    # 初始化日志和状态管理器
    logger = TrainingLogger(log_file)
    try:
        # 加载训练配置（优先使用传入的配置文件）
        training_args_dict = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                training_args_dict = json.load(f).get("elysia_train", {})
                logger.log(f"从配置文件加载参数: {config_path}")

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        logger.log(f"模型输出目录: {output_dir}")

        # 加载模型和分词器
        logger.log(f"加载基础物理解题模型: {physics_model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            physics_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(physics_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # 处理训练数据
        if adversarial_data_paths and len(adversarial_data_paths) > 0:
            merged_data = merge_adversarial_data(
                train_data_path,
                adversarial_data_paths,
                logger=logger
            )
            # 保存合并后的数据
            merged_data_path = os.path.join(output_dir, "merged_training_data.json")
            with open(merged_data_path, 'w', encoding='utf-8') as f:
                json.dump(merged_data, f, ensure_ascii=False, indent=2)
            dataset = load_dataset("json", data_files=merged_data_path)["train"]
            logger.log(f"使用合并数据训练，共 {len(dataset)} 条样本")
        else:
            dataset = load_dataset("json", data_files=train_data_path)["train"]
            logger.log(f"使用原始数据训练，共 {len(dataset)} 条样本")

        # 配置LoRA参数（角色微调优化）
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,  # 角色微调使用较大的秩以更好捕捉风格特征
            lora_alpha=32,
            lora_dropout=0.15,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
            inference_mode=False
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
        logger.log(f"LoRA配置完成，可训练参数占比: {model.print_trainable_parameters()}")

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

        # 处理数据集
        logger.log("开始预处理数据集...")
        tokenized_dataset = dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        logger.log("数据集预处理完成")

        # 初始化训练状态管理器
        total_epochs = training_args_dict.get("num_train_epochs", 10)
        state_manager = TrainingStateManager(state_file, total_epochs)

        # 配置训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=total_epochs,
            per_device_train_batch_size=training_args_dict.get("per_device_train_batch_size", 4),
            gradient_accumulation_steps=training_args_dict.get("gradient_accumulation_steps", 4),
            learning_rate=training_args_dict.get("learning_rate", 2e-5),
            warmup_ratio=training_args_dict.get("warmup_ratio", 0.1),
            logging_steps=training_args_dict.get("logging_steps", 10),
            save_steps=training_args_dict.get("save_steps", 500),
            fp16=training_args_dict.get("fp16", True),
            report_to=training_args_dict.get("report_to", "none"),  # 不使用wandb等外部报告
            save_total_limit=3,
            load_best_model_at_end=True,
            metric_for_best_model="loss"
        )

        # 初始化训练器
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
            callbacks=[ElysiaTrainingCallback(state_manager, logger)]  # 添加自定义回调
        )

        # 开始训练
        logger.log("开始爱莉希雅角色模型训练...")
        state_manager.set_status("training")
        trainer.train()

        # 训练完成
        model.save_pretrained(output_dir)
        with open(os.path.join(output_dir, "peft_config.json"), 'w') as f:
            json.dump(peft_config.to_dict(), f, indent=2)

        state_manager.set_status("completed")
        logger.log(f"训练完成！模型已保存至 {output_dir}", "success")
        return output_dir

    except Exception as e:
        logger.log(f"训练过程出错: {str(e)}", "error")
        if 'state_manager' in locals():
            state_manager.set_status("failed")
        raise e


# --------------------------
# 新增：参数解析（与后端对接）
# --------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="爱莉希雅角色模型训练脚本（支持状态跟踪）")
    parser.add_argument("--config", type=str, help="训练配置文件路径")
    parser.add_argument("--state_file", type=str, required=True, help="训练状态保存文件路径")
    parser.add_argument("--log_file", type=str, required=True, help="训练日志文件路径")
    parser.add_argument("--physics_model_path", type=str, help="物理模型路径")
    parser.add_argument("--train_data_path", type=str, help="训练数据路径")
    parser.add_argument("--output_dir", type=str, help="模型输出目录")
    parser.add_argument("--adversarial_paths", type=str, help="对抗数据路径（逗号分隔）")
    return parser.parse_args()


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()

    # 处理对抗数据路径
    adversarial_paths = []
    if args.adversarial_paths:
        adversarial_paths = args.adversarial_paths.split(",")

    # 调用训练函数
    train_elysia_model(
        physics_model_path=args.physics_model_path or "../../model/physics_lora",
        train_data_path=args.train_data_path or "../../data/processed/elysia_train",
        output_dir=args.output_dir or "../../model/elysia_adv_lora",
        state_file=args.state_file,
        log_file=args.log_file,
        adversarial_data_paths=adversarial_paths,
        config_path=args.config
    )
