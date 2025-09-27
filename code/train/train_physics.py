import os
import json
import torch
import argparse
from datetime import datetime
from datasets import load_from_disk
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
# 日志工具：对接后端日志系统（与角色训练保持一致）
# --------------------------
class TrainingLogger:
    def __init__(self, log_file):
        self.log_file = log_file
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        self.log("物理模型训练日志初始化完成", "info")

    def log(self, content, log_type="info"):
        timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        log_line = f"{timestamp} [{log_type.upper()}] {content}"
        print(log_line)
        with open(self.log_file, "a", encoding="utf-8") as f:
            f.write(log_line + "\n")


# --------------------------
# 训练状态管理：对接前端监控（字段与前端对齐）
# --------------------------
class TrainingStateManager:
    def __init__(self, state_file, total_epochs):
        self.state_file = state_file
        self.state = {
            "status": "init",  # init/training/paused/completed/failed
            "epoch": 0,
            "total_epochs": total_epochs,
            "current_loss": 0.0,
            "val_loss": 0.0,  # 新增：验证集损失（评估泛化能力）
            "best_val_loss": float("inf"),
            "physics_accuracy": 0.0,  # 物理逻辑准确率（阶段+公式覆盖）
            "start_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_time": "00:00:00",
            "epoch_history": []  # 轮次历史记录（供前端表格展示）
        }
        self.save()

    def update(self, epoch, current_loss, val_loss=0.0, physics_acc=0.0):
        """更新训练状态（每轮结束调用）"""
        self.state["status"] = "training"
        self.state["epoch"] = epoch
        self.state["current_loss"] = round(current_loss, 4)
        self.state["val_loss"] = round(val_loss, 4)
        self.state["best_val_loss"] = round(min(self.state["best_val_loss"], val_loss), 4)
        self.state["physics_accuracy"] = round(physics_acc, 2)

        # 计算累计训练时间
        start_dt = datetime.strptime(self.state["start_time"], "%Y-%m-%d %H:%M:%S")
        elapsed = datetime.now() - start_dt
        self.state["elapsed_time"] = str(elapsed).split(".")[0]  # 去除毫秒

        # 记录轮次历史
        self.state["epoch_history"].append({
            "epoch": epoch,
            "train_loss": self.state["current_loss"],
            "val_loss": self.state["val_loss"],
            "physics_accuracy": physics_acc,
            "time": self.state["elapsed_time"]
        })

        self.save()

    def set_status(self, status):
        """设置训练状态（如完成/失败）"""
        self.state["status"] = status
        self.save()

    def save(self):
        """保存状态到JSON文件（后端读取）"""
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(self.state, f, ensure_ascii=False, indent=2)


# --------------------------
# 训练回调：每轮结束更新状态+评估物理准确率
# --------------------------
class PhysicsTrainingCallback(TrainerCallback):
    def __init__(self, state_manager, logger, val_dataset, tokenizer):
        self.state_manager = state_manager
        self.logger = logger
        self.val_dataset = val_dataset  # 验证集（用于计算物理准确率）
        self.tokenizer = tokenizer

    def on_epoch_end(self, args, state, control, **kwargs):
        """每轮训练结束后执行：更新状态+评估物理逻辑准确率"""
        if state.epoch is None:
            return  # 跳过未明确轮次的情况

        current_epoch = int(state.epoch)
        # 获取当前轮次训练损失（取最后一个日志的训练损失）
        train_loss = state.log_history[-1].get("loss", 0.0)
        # 获取验证集损失（若有验证日志）
        val_loss = state.log_history[-1].get("eval_loss", 0.0)

        # 计算物理准确率：阶段覆盖率 + 公式覆盖率（简化评估，聚焦核心）
        physics_acc = self._calc_physics_accuracy(kwargs["model"])

        # 更新状态和日志
        self.state_manager.update(current_epoch, train_loss, val_loss, physics_acc)
        self.logger.log(
            f"轮次{current_epoch}结束 | 训练损失：{train_loss:.4f} | 验证损失：{val_loss:.4f} | 物理准确率：{physics_acc:.2f}%"
        )

    def _calc_physics_accuracy(self, model):
        """计算物理准确率：随机抽取10个验证样本，评估阶段和公式覆盖"""
        model.eval()
        device = next(model.parameters()).device
        sample_size = min(10, len(self.val_dataset))  # 抽样10个样本（平衡速度与准确性）
        correct_count = 0

        # 定义关键评估指标（根据物理题场景调整）
        required_stages = ["审题闭环", "建模闭环", "计算闭环", "迭代闭环"]
        required_formulas = ["动量守恒", "动能守恒", "v0²=2*g*l", "s=v0*t+(1/2)*a*t²"]

        with torch.no_grad():
            for idx in range(sample_size):
                sample = self.val_dataset[idx]
                # 构建输入（仅用raw_input，避免tokenized后的截断影响）
                input_text = sample["raw_input"]
                inputs = self.tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)

                # 模型生成解析
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.1,  # 低温度保证稳定性
                    do_sample=False
                )
                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                # 评估：阶段覆盖 + 公式覆盖
                stage_covered = sum(1 for stage in required_stages if stage in generated)
                formula_covered = sum(1 for formula in required_formulas if formula in generated)
                # 满足≥3个阶段+≥2个公式，视为正确
                if stage_covered >= 3 and formula_covered >= 2:
                    correct_count += 1

        model.train()
        # 计算准确率（百分比）
        return (correct_count / sample_size) * 100 if sample_size > 0 else 0.0


# --------------------------
# 主训练函数：加载数据+配置模型+启动训练
# --------------------------
def train_physics_model(
        base_model_path,
        train_data_dir,
        val_data_dir,
        output_dir,
        state_file,
        log_file,
        config_path
):
    # 初始化日志
    logger = TrainingLogger(log_file)
    try:
        # 1. 加载训练配置（从physics_config.json读取）
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"训练配置文件不存在: {config_path}")
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        logger.log(f"从配置文件加载参数：{config_path}")

        # 2. 加载基础模型和分词器（ChatGLM-6B）
        logger.log(f"加载基础模型：{base_model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.float16,  # FP16节省显存
            device_map="auto"  # 自动分配设备（GPU优先）
        )
        tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.log("模型和分词器加载完成")

        # 3. 加载预处理后的训练集和验证集（Dataset格式）
        logger.log(f"加载训练集：{train_data_dir}")
        logger.log(f"加载验证集：{val_data_dir}")
        train_dataset = load_from_disk(train_data_dir)
        val_dataset = load_from_disk(val_data_dir)
        logger.log(f"数据集规模：训练集{len(train_dataset)}条，验证集{len(val_dataset)}条")

        # 4. 配置LoRA（参数高效微调，聚焦物理逻辑学习）
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,  # 秩：物理训练用8（比角色训练大，确保知识学习充分）
            lora_alpha=32,  # 缩放因子：与r匹配
            lora_dropout=0.1,  # dropout：防止过拟合
            target_modules=["q_proj", "v_proj"],  # ChatGLM-6B关键注意力层
            bias="none",
            inference_mode=False
        )
        model = get_peft_model(model, peft_config)
        trainable_info = model.print_trainable_parameters()  # 打印可训练参数占比
        logger.log(f"LoRA配置完成，可训练参数：{trainable_info}")

        # 5. 配置训练参数（从config读取，覆盖默认值）
        total_epochs = config.get("num_train_epochs", 5)
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=total_epochs,
            per_device_train_batch_size=config.get("per_device_train_batch_size", 2),
            per_device_eval_batch_size=config.get("per_device_train_batch_size", 2),
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
            learning_rate=config.get("learning_rate", 2e-4),
            warmup_ratio=config.get("warmup_ratio", 0.1),
            logging_steps=config.get("logging_steps", 10),
            save_steps=config.get("save_steps", 500),
            save_total_limit=config.get("save_total_limit", 3),
            fp16=config.get("fp16", torch.cuda.is_available()),  # GPU可用则启用FP16
            evaluation_strategy="epoch",  # 每轮结束评估验证集
            eval_delay=0,
            report_to="none",  # 不使用wandb等外部工具
            load_best_model_at_end=True,  # 训练结束加载最优模型（按val_loss）
            metric_for_best_model="eval_loss",
            greater_is_better=False  # val_loss越小越好
        )

        # 6. 初始化训练状态管理器
        state_manager = TrainingStateManager(state_file, total_epochs)
        state_manager.set_status("training")  # 设置状态为训练中

        # 7. 初始化训练器（含自定义回调）
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            data_collator=data_collator,
            callbacks=[
                PhysicsTrainingCallback(
                    state_manager=state_manager,
                    logger=logger,
                    val_dataset=val_dataset,
                    tokenizer=tokenizer
                )
            ]
        )

        # 8. 启动训练
        logger.log(f"开始物理模型训练（共{total_epochs}轮）")
        trainer.train()

        # 9. 训练完成：保存模型和配置
        model.save_pretrained(output_dir)
        peft_config.save_pretrained(output_dir)
        # 保存训练配置（方便后续复现）
        with open(os.path.join(output_dir, "training_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, ensure_ascii=False, indent=2)

        # 更新状态为完成
        state_manager.set_status("completed")
        logger.log(f"物理模型训练完成！模型保存至：{output_dir}", "success")
        return output_dir

    except Exception as e:
        # 训练失败：更新状态并记录错误
        logger.log(f"训练过程出错：{str(e)}", "error")
        if "state_manager" in locals():
            state_manager.set_status("failed")
        raise e


# --------------------------
# 命令行参数解析（统一入口）
# --------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="物理模型LoRA微调脚本（支持前端监控）")
    parser.add_argument("--config", type=str, default="../config/physics_config.json",
                        help="物理训练配置文件路径")
    parser.add_argument("--state_file", type=str, required=True,
                        help="训练状态保存文件路径（如../logs/physics_state.json）")
    parser.add_argument("--log_file", type=str, required=True,
                        help="训练日志文件路径（如../logs/physics_train.log）")
    parser.add_argument("--base_model_path", type=str, default="../../model/base",
                        help="基础模型（ChatGLM-6B）路径")
    parser.add_argument("--train_data_dir", type=str, default="../../data/processed/physics_train/train_dataset",
                        help="预处理后的训练集路径")
    parser.add_argument("--val_data_dir", type=str, default="../../data/processed/physics_train/val_dataset",
                        help="预处理后的验证集路径")
    parser.add_argument("--output_dir", type=str, default="../../model/physics_lora",
                        help="模型输出目录")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train_physics_model(
        base_model_path=args.base_model_path,
        train_data_dir=args.train_data_dir,
        val_data_dir=args.val_data_dir,
        output_dir=args.output_dir,
        state_file=args.state_file,
        log_file=args.log_file,
        config_path=args.config
    )