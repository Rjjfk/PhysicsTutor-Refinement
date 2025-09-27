import os
import json
import torch
import argparse
from datasets datasets
import Dataset, load_from_disk
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    TrainingArguments, Trainer, DataCollatorForLanguageModeling
)
from peft import PeftModel, get_peft_model, LoraConfig, TaskType
# 复用项目核心组件
from train_physics import TrainingLogger, TrainingStateManager
from discriminator import DualDiscriminator
from code.eval.eval_physics import evaluate_formula_accuracy, evaluate_step_accuracy
from code.eval.eval_elysia import evaluate_role_consistency


def load_combined_config(task_type, default_config_path, self_train_config_path):
    """加载并合并默认配置和自训练配置，按任务类型筛选参数"""
    with open(default_config_path, "r", encoding="utf-8") as f:
        default_config = json.load(f)
    with open(self_train_config_path, "r", encoding="utf-8") as f:
        self_train_config = json.load(f)

    # 合并基础参数与任务专属参数
    task_config = {
        **default_config,  # 通用参数（batch_size, learning_rate等）
        **default_config[task_type],  # 任务专属基础参数
        "loop_count": self_train_config["loop_count"],
        "generate_batch_size": self_train_config["generate_batch_size"],
        "generate_strategy": self_train_config["generate_strategy"],
        "filter_thresholds": self_train_config["filter_thresholds"][task_type]
    }
    return task_config


def generate_self_training_samples(model, tokenizer, task_type, config, logger):
    """
    基于种子数据生成新样本
    - 物理：生成多样化物理题解析
    - 角色：生成爱莉希雅风格对话与解析
    """
    generated_samples = []
    device = next(model.parameters()).device
    gen_strategy = config["generate_strategy"]
    num_samples = config["generate_num_samples"]
    logger.log(f"开始生成{task_type}自训练样本：总数{num_samples}，生成策略{gen_strategy}")

    # 种子数据（覆盖核心场景）
    if task_type == "physics":
        seed_questions = [
            "质量为m的小球以速度v0水平抛出，重力加速度为g，求t秒后的竖直位移和速度大小。",
            "两个带电量分别为q1和q2的点电荷，相距r，库仑力大小是多少？若距离变为2r，力如何变化？",
            "劲度系数为k的弹簧，拉伸x距离时弹性势能是多少？若拉伸量加倍，势能如何变化？",
            "质量为M的木块静止在光滑水平面上，质量为m的子弹以速度v射入并嵌入其中，求共同速度。",
            "单摆摆长为l，重力加速度为g，其周期公式是什么？若在月球上（g'=g/6），周期如何变化？"
        ]
    else:  # role
        seed_questions = [
            "爱莉希雅，用你的方式给我讲讲动量守恒定律吧～♪",
            "如果小球从斜面滑下来撞到另一个球，你会怎么解析这道题呀？",
            "为什么物理里的公式都这么复杂呢？用你的风格解释一下嘛～",
            "解完这道题我们去跳舞吧！先告诉我圆周运动的向心力公式是什么～",
            "用飞花和水晶的比喻，给我讲讲能量守恒定律好不好？"
        ]

    # 生成样本（循环直到达到目标数量）
    while len(generated_samples) < num_samples:
        # 随机选择种子问题
        seed_idx = len(generated_samples) % len(seed_questions)
        input_text = seed_questions[seed_idx]

        # 生成文本
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=gen_strategy["max_new_tokens"],
                do_sample=True,
                temperature=gen_strategy["temperature"],
                top_p=gen_strategy["top_p"],
                no_repeat_ngram_size=3  # 避免重复短语
            )

        gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        generated_samples.append({
            "input": input_text,
            "generated_output": gen_text
        })

        # 进度提示
        if len(generated_samples) % 50 == 0:
            logger.log(f"已生成{len(generated_samples)}/{num_samples}个样本")

    return generated_samples


def filter_high_quality_samples(samples, task_type, model, discriminator, tokenizer, config, logger):
    """
    多维度筛选优质样本
    - 物理：公式正确性、步骤完整性、逻辑一致性
    - 角色：风格特征覆盖率、物理内容准确性、长度合规性
    """
    high_quality = []
    device = next(discriminator.parameters()).device
    thresholds = config["filter_thresholds"]
    logger.log(f"开始筛选{task_type}样本，阈值：{thresholds}")

    for idx, sample in enumerate(samples):
        gen_text = sample["generated_output"]
        input_text = sample["input"]

        # 1. 基础长度筛选
        if len(gen_text) < thresholds["min_length"] or len(gen_text) > thresholds["max_length"]:
            continue

        # 2. 任务专属筛选
        if task_type == "physics":
            # 物理：公式数量+步骤数量+判别器分数
            formula_coverage, _ = evaluate_formula_accuracy(gen_text, gen_text)
            step_count = evaluate_step_accuracy(gen_text)["步骤总数"]

            # 判别器逻辑一致性评分
            input_ids = tokenizer(
                gen_text,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(device)["input_ids"]
            logic_score = discriminator(input_ids).item()

            # 满足所有条件
            if (formula_coverage >= thresholds["min_formula_count"] and
                    step_count >= thresholds["min_step_count"] and
                    logic_score >= thresholds["logic_consistency_score"]):
                high_quality.append({
                    "raw_input": input_text,
                    "raw_output": gen_text,
                    "source": "self-generated",
                    "metrics": {
                        "formula_coverage": formula_coverage,
                        "step_count": step_count,
                        "logic_score": logic_score
                    }
                })

        else:  # role
            # 角色：风格覆盖率+物理准确性+关键词数量
            style_score = evaluate_role_consistency(gen_text)["风格特征覆盖率"]

            # 检查物理内容准确性（复用物理评估）
            physics_text = gen_text  # 从角色回复中提取物理部分
            if "解析" in gen_text:
                physics_text = gen_text.split("解析")[1].strip()
            physics_accuracy = evaluate_formula_accuracy(physics_text, physics_text)[0] / 100  # 转为0-1

            # 判别器风格一致性评分
            input_ids = tokenizer(
                gen_text,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).to(device)["input_ids"]
            style_consistency = discriminator(input_ids).item()

            # 满足所有条件
            if (style_score >= thresholds["style_feature_coverage"] and
                    physics_accuracy >= thresholds["physics_accuracy"] and
                    style_consistency >= thresholds["style_consistency_score"]):
                high_quality.append({
                    "raw_input": input_text,
                    "raw_output": gen_text,
                    "source": "self-generated",
                    "metrics": {
                        "style_score": style_score,
                        "physics_accuracy": physics_accuracy,
                        "style_consistency": style_consistency
                    }
                })

        # 进度提示
        if (idx + 1) % 100 == 0:
            logger.log(f"已筛选{idx + 1}/{len(samples)}个样本，优质率：{len(high_quality) / (idx + 1):.2f}")

    logger.log(f"筛选完成：原始{len(samples)}个，优质{len(high_quality)}个（保留率{len(high_quality) / len(samples):.2f}）")
    return high_quality


def increment_finetune(model, tokenizer, dataset, config, output_dir, logger):
    """用优质自生成样本进行增量微调（低学习率，少轮次）"""
    # 配置LoRA（仅微调关键层，最小化参数更新）
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=2,  # 低秩矩阵维度（比对抗训练更小）
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=["o_proj"] if config["task_type"] == "physics" else ["fc_out"],
        bias="none",
        inference_mode=False
    )
    model = get_peft_model(model, peft_config)
    logger.log(f"增量微调LoRA配置完成，可训练参数：{model.print_trainable_parameters()}")

    # 训练参数（低学习率，少轮次）
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=config["batch_size"],
        num_train_epochs=config["epochs"],
        learning_rate=float(config["learning_rate"]),
        logging_steps=5,
        fp16=True,
        report_to="none",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss"
    )

    # 数据collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # 因果语言模型不需要掩码
    )

    # 初始化Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )

    # 开始微调
    logger.log(f"开始增量微调：{config['epochs']}轮，批次大小{config['batch_size']}")
    trainer.train()

    # 保存微调后的模型
    model.save_pretrained(output_dir)
    logger.log(f"增量微调完成，模型保存至：{output_dir}")
    return model


def self_train_loop(args):
    """自训练主流程：多轮生成→筛选→微调循环"""
    # 1. 初始化日志和配置
    logger = TrainingLogger(args.log_file)
    logger.log(f"===== 启动{args.task}自训练 =====")

    # 加载合并配置
    config = load_combined_config(
        task_type=args.task,
        default_config_path=args.default_config,
        self_train_config_path=args.self_train_config
    )
    config["task_type"] = args.task  # 记录任务类型
    logger.log(f"加载配置完成：{json.dumps(config, indent=2, ensure_ascii=False)}")

    # 2. 加载基础模型和判别器
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log(f"使用设备：{device}")

    # 加载基础模型（对抗训练后的模型）
    if args.task == "physics":
        # 物理模型：直接加载对抗增强后的LoRA模型
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        # 加载物理判别器
        discriminator = DualDiscriminator(
            tokenizer_path=args.base_model_path,
            task_type="physics"
        ).to(device)
    else:
        # 角色模型：先加载物理基础模型，再叠加角色对抗LoRA
        base_physics_path = config.get("base_physics_model_path", "../../model/physics_adversarial_lora")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_physics_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        base_model = PeftModel.from_pretrained(base_model, args.base_model_path)
        # 加载角色判别器
        discriminator = DualDiscriminator(
            tokenizer_path=args.base_model_path,
            task_type="role"
        ).to(device)

    # 加载判别器权重
    disc_weight_path = os.path.join(args.base_model_path, "discriminator.pth")
    if os.path.exists(disc_weight_path):
        discriminator.load_state_dict(torch.load(disc_weight_path, map_location=device))
        logger.log(f"加载判别器权重：{disc_weight_path}")
    else:
        logger.log(f"警告：未找到判别器权重{disc_weight_path}，使用随机初始化")

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. 多轮自训练循环
    current_model = base_model
    current_output_dir = args.output_dir
    os.makedirs(current_output_dir, exist_ok=True)

    for loop in range(config["loop_count"]):
        logger.log(f"\n===== 自训练循环 {loop + 1}/{config['loop_count']} =====")
        loop_output_dir = os.path.join(current_output_dir, f"loop_{loop + 1}")
        os.makedirs(loop_output_dir, exist_ok=True)

        # 3.1 生成自训练样本
        generated_samples = generate_self_training_samples(
            model=current_model,
            tokenizer=tokenizer,
            task_type=args.task,
            config=config,
            logger=logger
        )

        # 3.2 筛选高质量样本
        high_quality_samples = filter_high_quality_samples(
            samples=generated_samples,
            task_type=args.task,
            model=current_model,
            discriminator=discriminator,
            tokenizer=tokenizer,
            config=config,
            logger=logger
        )

        # 若优质样本不足，终止循环
        if len(high_quality_samples) < config["generate_batch_size"]:
            logger.log(f"优质样本数量不足（{len(high_quality_samples)}），终止自训练循环")
            break

        # 3.3 合并原始数据与自生成数据
        logger.log(f"合并原始数据（{config['data_path']}）与自生成数据")
        original_dataset = load_from_disk(config["data_path"])
        self_dataset = Dataset.from_list(high_quality_samples)

        # 按8:2比例合并（原始数据为主，自生成数据为辅）
        combined_dataset = original_dataset.train_test_split(test_size=0.2)[0].concatenate(
            self_dataset.train_test_split(test_size=0.2)[0]
        ).shuffle(seed=42 + loop)  # 每轮随机种子不同

        # 3.4 增量微调
        current_model = increment_finetune(
            model=current_model,
            tokenizer=tokenizer,
            dataset=combined_dataset,
            config=config,
            output_dir=loop_output_dir,
            logger=logger
        )

        # 3.5 保存本轮优质样本
        with open(os.path.join(loop_output_dir, "high_quality_samples.json"), "w", encoding="utf-8") as f:
            json.dump(high_quality_samples, f, ensure_ascii=False, indent=2)

    # 4. 保存最终模型软链接
    final_model_path = os.path.join(current_output_dir, "final_model")
    if os.path.exists(final_model_path):
        os.remove(final_model_path)
    os.symlink(os.path.basename(loop_output_dir), final_model_path)
    logger.log(f"自训练完成，最终模型路径：{final_model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="物理/角色任务自训练循环脚本")
    parser.add_argument("--task", type=str, required=True, choices=["physics", "role"],
                        help="任务类型：physics（物理）或role（角色）")
    parser.add_argument("--base_model_path", type=str, required=True,
                        help="对抗训练后的基础模型路径（如model/physics_adversarial_lora）")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="自训练模型保存根目录")
    parser.add_argument("--log_file", type=str, required=True,
                        help="训练日志文件路径")
    parser.add_argument("--default_config", type=str,
                        default="../../config/default_config.json",
                        help="默认配置文件路径")
    parser.add_argument("--self_train_config", type=str,
                        default="../../config/self_train_config.json",
                        help="自训练专用配置文件路径")
    args = parser.parse_args()

    self_train_loop(args)
