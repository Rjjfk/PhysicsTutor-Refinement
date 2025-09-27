import os
import json
import time
import argparse
import torch
from datetime import datetime
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import get_peft_model, LoraConfig, TaskType
from rouge import Rouge  # 新增：用于计算物理逻辑一致性（需安装：pip install rouge）


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
# 新增：准确率评估函数（真实计算）
# --------------------------
def evaluate_accuracy(model, val_dataset, tokenizer, device):
    """
    评估两项准确率：
    1. 物理准确率：生成的物理逻辑与标准答案的一致性（用ROUGE-L评分）
    2. 角色准确率：生成是否包含爱莉希雅风格特征（关键词匹配）
    """
    model.eval()
    rouge = Rouge()
    physics_scores = []
    role_matches = 0
    total_samples = min(50, len(val_dataset))  # 每次评估取50个样本，平衡速度与准确性

    # 角色风格关键词（爱莉希雅标志性元素）
    role_keywords = ["～♪", "呀", "哦", "呢", "啦", "舞蹈", "旋律", "音符", "花瓣", "华丽"]

    with torch.no_grad():
        for idx in range(total_samples):
            sample = val_dataset[idx]
            input_text = sample["input"]
            instruction = sample["instruction"]  # 物理逻辑标准答案
            prompt = f"作为崩坏三的爱莉希雅，用活泼优雅的语气解析物理题，要包含完整逻辑链和角色特征：{input_text}"

            # 生成模型输出
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                do_sample=True
            )
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()

            # 1. 计算物理准确率（ROUGE-L：衡量生成逻辑与标准答案的相似度）
            try:
                # 提取生成文本中的物理逻辑部分（去除角色语气词）
                generated_logic = generated_text.replace("～♪", "").replace("呀", "").replace("哦", "").strip()
                # 用ROUGE-L评分（100分制）
                score = rouge.get_scores(generated_logic, instruction)[0]["rouge-l"]["f"] * 100
                physics_scores.append(score)
            except:
                physics_scores.append(0.0)  # 解析失败时记0分

            # 2. 计算角色准确率（包含≥2个关键词则视为匹配）
            keyword_count = sum(1 for kw in role_keywords if kw in generated_text)
            if keyword_count >= 2:
                role_matches += 1

    # 计算平均准确率
    avg_physics_acc = sum(physics_scores) / len(physics_scores) if physics_scores else 0.0
    role_acc = (role_matches / total_samples) * 100  # 百分比

    model.train()
    return avg_physics_acc, role_acc


# --------------------------
# 新增：训练回调类（跟踪训练进度）
# --------------------------
class ElysiaTrainingCallback(TrainerCallback):
    def __init__(self, state_manager, logger, val_dataset, model, tokenizer, device):
        self.state_manager = state_manager
        self.logger = logger
        self.val_dataset = val_dataset  # 验证集，用于计算准确率
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def on_epoch_end(self, args, state, control, **kwargs):
        """每轮结束时更新状态+评估准确率"""
        if state.epoch is not None:
            current_epoch = int(state.epoch)
            current_loss = state.log_history[-1].get("loss", 0.0)

            # 评估真实准确率
            avg_physics_acc, role_acc = evaluate_accuracy(
                self.model, self.val_dataset, self.tokenizer, self.device
            )

            # 更新状态
            self.state_manager.update(current_epoch, current_loss, avg_physics_acc, role_acc)
            self.logger.log(
                f"轮次 {current_epoch} 结束 | 损失: {current_loss:.4f} | "
                f"物理准确率: {avg_physics_acc:.2f}% | 角色准确率: {role_acc:.2f}%"
            )


# --------------------------
# 原有核心功能：数据合并（保留并优化）
# --------------------------
def merge_adversarial_data(original_data_path, adversarial_data_paths, min_quality_score=0.7, logger=None):
    """合并原始角色数据与对抗生成的优质数据"""
    log = logger.log if logger else print

    # 加载原始数据（适配新字段：input/instruction/output）
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

        # 筛选高质量生成数据（需包含input/instruction/output字段和quality_score）
        filtered = [
            item for item in gen_data
            if item.get('quality_score', 0) >= min_quality_score
               and all(k in item for k in ["input", "instruction", "output"])
        ]
        adversarial_data.extend(filtered)
        log(f"加载对抗数据 {path}: 原始{len(gen_data)}条，筛选后{len(filtered)}条")

    # 合并并去重（按input字段去重，避免重复样本）
    combined_dict = {item["input"]: item for item in original_data + adversarial_data}
    combined_data = list(combined_dict.values())

    log(f"数据合并完成：原始{len(original_data)}条 + 对抗{len(adversarial_data)}条 → 去重后{len(combined_data)}条")
    return combined_data


# --------------------------
# 主训练函数（整合状态管理+真实评估）
# --------------------------
def train_elysia_model(
        physics_model_path,
        train_data_path,
        val_data_path,  # 新增：验证集路径，用于评估准确率
        output_dir,
        state_file,
        log_file,
        adversarial_data_paths=None,
        config_path=None
):
    # 初始化日志
    logger = TrainingLogger(log_file)
    try:
        # 加载训练配置（优先使用传入的配置文件）
        training_args_dict = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                training_args_dict = json.load(f)
                logger.log(f"从配置文件加载参数: {config_path}")

        # 确定设备（GPU优先）
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.log(f"使用设备: {device}")

        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
        logger.log(f"模型输出目录: {output_dir}")

        # 加载模型和分词器（基于物理微调后的模型）
        logger.log(f"加载基础物理解题模型: {physics_model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            physics_model_path,
            torch_dtype=torch.float16 if device.type == "cuda" else torch.float32,
            device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(physics_model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        logger.log("模型和分词器加载完成")

        # 处理训练数据（支持合并对抗数据）
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
            # 划分训练集和验证集（若未指定单独验证集）
            if not val_data_path or not os.path.exists(val_data_path):
                logger.log("未指定验证集，从合并数据中划分（按preprocess_params.train_val_split）")
                split_ratio = training_args_dict.get("preprocess_params", {}).get("train_val_split", 0.9)
                merged_dataset = Dataset.from_list(merged_data)
                split_dataset = merged_dataset.train_test_split(test_size=1 - split_ratio, seed=42)
                train_dataset = split_dataset["train"]
                val_dataset = split_dataset["test"]
            else:
                train_dataset = Dataset.from_list(merged_data)
                val_dataset = load_dataset("json", data_files=val_data_path)["train"]
            logger.log(f"使用合并数据训练：训练集{len(train_dataset)}条，验证集{len(val_dataset)}条")
        else:
            # 直接加载原始训练集和验证集
            train_dataset = load_dataset("json", data_files=train_data_path)["train"]
            val_dataset = load_dataset("json", data_files=val_data_path)["train"] if (
                        val_data_path and os.path.exists(val_data_path)) else None
            if not val_dataset:
                logger.log("未找到验证集，从训练集中划分10%作为验证集")
                split_dataset = train_dataset.train_test_split(test_size=0.1, seed=42)
                train_dataset = split_dataset["train"]
                val_dataset = split_dataset["test"]
            logger.log(f"使用原始数据训练：训练集{len(train_dataset)}条，验证集{len(val_dataset)}条")

        # 配置LoRA参数（角色微调优化：多目标模块，捕捉风格特征）
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=8,
            lora_alpha=32,
            lora_dropout=0.15,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            bias="none",
            inference_mode=False
        )
        model = get_peft_model(model, peft_config)
        trainable_info = model.print_trainable_parameters()  # 打印可训练参数占比
        logger.log(f"LoRA配置完成，可训练参数信息: {trainable_info}")

        # 数据预处理：适配新字段（input/instruction/output）+ 强化角色引导
        def preprocess_function(examples):
            # 构建输入prompt（包含角色引导+物理场景+逻辑提示）
            inputs = [
                f"作为崩坏三的爱莉希雅，用活泼优雅的语气解析物理题，要包含完整逻辑链和角色特征（如～♪、舞蹈比喻等）：{examples['input']}\n物理逻辑要求：{examples['instruction']}"
                for _, examples in examples.items()  # 适配Dataset的字典格式
            ]
            # 处理输入tokenize
            model_inputs = tokenizer(
                inputs,
                max_length=training_args_dict.get("preprocess_params", {}).get("max_length", 1024),
                truncation=True,
                padding="max_length"
            )
            # 处理输出tokenize（包含角色风格标记）
            outputs = [f"~♪ 爱莉希雅的解析时间到啦～ {ans}" for ans in examples["output"]]
            labels = tokenizer(
                outputs,
                max_length=training_args_dict.get("preprocess_params", {}).get("max_length", 1024),
                truncation=True,
                padding="max_length"
            )
            # 设置labels（-100表示不计算该位置损失）
            model_inputs["labels"] = [
                [-100 if token == tokenizer.pad_token_id else label for token, label in zip(input_ids, label_ids)]
                for input_ids, label_ids in zip(model_inputs["input_ids"], labels["input_ids"])
            ]
            return model_inputs

        # 执行预处理
        logger.log("开始预处理训练数据集...")
        tokenized_train = train_dataset.map(
            preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        # 预处理验证集（用于评估，无需标签）
        tokenized_val = val_dataset.map(
            lambda x: tokenizer(
                x["input"],
                max_length=1024,
                truncation=True,
                padding="max_length"
            ),
            batched=True,
            remove_columns=val_dataset.column_names
        )
        logger.log("数据集预处理完成")

        # 初始化训练状态管理器
        total_epochs = training_args_dict.get("num_train_epochs", 3)
        state_manager = TrainingStateManager(state_file, total_epochs)

        # 配置训练参数
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=total_epochs,
            per_device_train_batch_size=training_args_dict.get("per_device_train_batch_size", 2),
            per_device_eval_batch_size=training_args_dict.get("per_device_train_batch_size", 2),
            gradient_accumulation_steps=training_args_dict.get("gradient_accumulation_steps", 4),
            learning_rate=training_args_dict.get("learning_rate", 1e-4),
            warmup_ratio=training_args_dict.get("warmup_ratio", 0.1),
            logging_steps=training_args_dict.get("logging_steps", 10),
            save_steps=training_args_dict.get("save_steps", 500),
            save_total_limit=training_args_dict.get("save_total_limit", 3),
            fp16=training_args_dict.get("fp16", device.type == "cuda"),
            report_to="none",
            load_best_model_at_end=True,
            metric_for_best_model="loss",
            evaluation_strategy="epoch"  # 每轮结束评估验证集损失
        )

        # 初始化训练器
        data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_val,
            data_collator=data_collator,
            callbacks=[
                ElysiaTrainingCallback(
                    state_manager=state_manager,
                    logger=logger,
                    val_dataset=val_dataset,  # 原始验证集（用于准确率评估）
                    model=model,
                    tokenizer=tokenizer,
                    device=device
                )
            ]
        )

        # 开始训练
        logger.log("开始爱莉希雅角色风格微调训练...")
        state_manager.set_status("training")
        trainer.train()

        # 训练完成：保存模型和配置
        model.save_pretrained(output_dir)
        peft_config.save_pretrained(output_dir)
        with open(os.path.join(output_dir, "training_config.json"), 'w', encoding='utf-8') as f:
            json.dump(training_args_dict, f, ensure_ascii=False, indent=2)

        # 更新状态为完成
        state_manager.set_status("completed")
        logger.log(f"训练完成！模型已保存至 {output_dir}", "success")
        return output_dir

    except Exception as e:
        logger.log(f"训练过程出错: {str(e)}", "error")
        if 'state_manager' in locals():
            state_manager.set_status("failed")
        raise e


# --------------------------
# 参数解析（支持验证集路径+对抗数据路径）
# --------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="爱莉希雅角色风格微调脚本（支持前端监控）")
    parser.add_argument("--config", type=str, help="训练配置文件路径（如elysia_config.json）")
    parser.add_argument("--state_file", type=str, required=True,
                        help="训练状态保存文件路径（如../logs/elysia_state.json）")
    parser.add_argument("--log_file", type=str, required=True, help="训练日志文件路径（如../logs/elysia_train.log）")
    parser.add_argument("--physics_model_path", type=str, default="../../model/physics_lora", help="物理微调后模型路径")
    parser.add_argument("--train_data_path", type=str, default="../../data/processed/elysia/train.json",
                        help="训练数据路径")
    parser.add_argument("--val_data_path", type=str, default="../../data/processed/elysia/val.json",
                        help="验证数据路径（可选）")
    parser.add_argument("--output_dir", type=str, default="../../model/elysia_adv_lora", help="模型输出目录")
    parser.add_argument("--adversarial_paths", type=str, help="对抗生成数据路径（逗号分隔，可选）")
    return parser.parse_args()


if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    # 处理对抗数据路径（逗号分隔转列表）
    adversarial_paths = args.adversarial_paths.split(",") if args.adversarial_paths else []
    # 调用训练函数
    train_elysia_model(
        physics_model_path=args.physics_model_path,
        train_data_path=args.train_data_path,
        val_data_path=args.val_data_path,
        output_dir=args.output_dir,
        state_file=args.state_file,
        log_file=args.log_file,
        adversarial_data_paths=adversarial_paths,
        config_path=args.config
    )