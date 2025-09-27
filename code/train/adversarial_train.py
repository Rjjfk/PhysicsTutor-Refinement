import os
import json
import torch
import torch.optim as optim
import argparse
from datasets import load_from_disk, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, get_peft_model, LoraConfig
# 导入统一组件和双分支判别器
from train_physics import TrainingLogger, TrainingStateManager
from discriminator import DualDiscriminator


# --------------------------
# 任务专属：错误样本生成器（物理/角色共用函数，按任务类型生成）
# --------------------------
def generate_error_samples(task_type, correct_samples, tokenizer):
    """
    生成错误样本：
    - physics：物理逻辑错误（公式错、步骤缺）
    - role：风格偏离（物理正确，无爱莉希雅特征）
    """
    error_samples = []
    if task_type == "physics":
        # 物理错误样本生成（复用之前的逻辑）
        for sample in correct_samples:
            question = sample["raw_input"].split("：")[-1].strip()
            correct_answer = sample["raw_output"]
            # 公式参数错误（l→L）
            wrong_answer = correct_answer.replace("v0²=2*g*l", "v0²=2*g*L")
            error_samples.append({
                "input": f"解析物理题：{question}",
                "output": wrong_answer,
                "label": 0,
                "is_adversarial": True
            })
    else:
        # 角色错误样本生成（复用之前的逻辑）
        with open("../../config/elysia_config.json", "r") as f:
            style_features = json.load(f)["role_adversarial"]["style_features"]
        for sample in correct_samples:
            question = sample["raw_input"]
            correct_answer = sample["raw_output"]
            # 移除风格特征
            wrong_answer = correct_answer
            for feat in style_features:
                wrong_answer = wrong_answer.replace(feat, "")
            error_samples.append({
                "input": question,
                "output": wrong_answer,
                "label": 0,
                "is_adversarial": True
            })
    return error_samples


# --------------------------
# 核心：双任务对抗训练主函数
# --------------------------
def run_adversarial_train(args):
    # 1. 初始化日志和配置
    logger = TrainingLogger(args.log_file)
    with open(args.config_path, "r") as f:
        config = json.load(f)
    adv_config = config["physics_adversarial"] if args.task == "physics" else config["role_adversarial"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.log(f"开始{args.task}对抗训练，设备：{device}，配置：{adv_config}")

    # 2. 加载基础模型（物理/角色）
    if args.task == "physics":
        # 物理任务：加载基础物理模型（train_physics.py输出）
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path, torch_dtype=torch.float16, device_map="auto"
        )
        # 对抗LoRA：仅微调输出层（保护物理逻辑）
        peft_config = LoraConfig(
            task_type="CAUSAL_LM", r=4, lora_alpha=16, target_modules=["o_proj"], lora_dropout=0.1
        )
    else:
        # 角色任务：加载基础角色模型（train_elysia.py输出，需先加载物理模型）
        base_physics_model = AutoModelForCausalLM.from_pretrained(
            config["base_physics_model_path"], torch_dtype=torch.float16, device_map="auto"
        )
        base_model = PeftModel.from_pretrained(base_physics_model, args.base_model_path)
        # 对抗LoRA：仅微调风格层（保护物理逻辑）
        peft_config = LoraConfig(
            task_type="CAUSAL_LM", r=4, lora_alpha=16, target_modules=["fc_out"], lora_dropout=0.1
        )
    model = get_peft_model(base_model, peft_config)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 3. 准备对抗数据（正确样本+错误样本）
    correct_dataset = load_from_disk(args.correct_data_dir)
    correct_samples = [
        {
            "raw_input": sample["raw_input"],
            "raw_output": sample["raw_output"],
            "label": 1,
            "is_adversarial": False
        } for sample in correct_dataset
    ]
    # 生成错误样本（1:1比例）
    error_samples = generate_error_samples(args.task, correct_samples, tokenizer)
    # 合并为最终对抗数据集
    all_samples = correct_samples + error_samples
    adv_dataset = Dataset.from_list(all_samples).shuffle(seed=42)
    logger.log(f"对抗数据准备完成：正确样本{len(correct_samples)}，错误样本{len(error_samples)}")

    # 4. 初始化判别器和优化器
    discriminator = DualDiscriminator(
        tokenizer_path=args.base_model_path,
        task_type=args.task
    ).to(device)
    gen_optim = optim.Adam(model.parameters(), lr=adv_config["gen_lr"])
    disc_optim = optim.Adam(discriminator.parameters(), lr=adv_config["disc_lr"])
    bce_loss = torch.nn.BCELoss()

    # 5. 对抗训练循环（生成器-判别器交替）
    total_epochs = adv_config["num_train_epochs"]
    batch_size = adv_config["per_device_train_batch_size"]
    num_batches = len(adv_dataset) // batch_size
    state_manager = TrainingStateManager(args.state_file, total_epochs)
    state_manager.set_status("training")

    try:
        for epoch in range(total_epochs):
            model.train()
            discriminator.train()
            total_gen_loss = 0.0
            total_disc_loss = 0.0

            for batch_idx in range(num_batches):
                # 取批次数据
                batch = adv_dataset[batch_idx*batch_size : (batch_idx+1)*batch_size]
                inputs = tokenizer(
                    [item["raw_input"] for item in batch],
                    padding=True, truncation=True, max_length=512, return_tensors="pt"
                ).to(device)
                labels = torch.tensor([item["label"] for item in batch], dtype=torch.float32).unsqueeze(1).to(device)

                # --------------------------
                # 步骤1：训练判别器
                # --------------------------
                disc_optim.zero_grad()
                # 判别真实样本
                real_outputs = tokenizer(
                    [item["raw_output"] for item in batch],
                    padding=True, truncation=True, max_length=1024, return_tensors="pt"
                ).to(device)
                real_pred = discriminator(real_outputs["input_ids"])
                disc_real_loss = bce_loss(real_pred, labels)
                # 判别生成样本
                gen_outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
                fake_pred = discriminator(gen_outputs["sequences"])
                disc_fake_loss = bce_loss(fake_pred, torch.zeros_like(labels))
                # 更新判别器
                disc_loss = (disc_real_loss + disc_fake_loss) / 2
                disc_loss.backward()
                disc_optim.step()
                total_disc_loss += disc_loss.item()

                # --------------------------
                # 步骤2：训练生成器
                # --------------------------
                gen_optim.zero_grad()
                # 对抗损失（欺骗判别器）
                gen_outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
                fake_pred = discriminator(gen_outputs["sequences"])
                adv_loss = bce_loss(fake_pred, torch.ones_like(labels))
                # 任务专属损失（物理：逻辑损失；角色：风格损失）
                if args.task == "physics":
                    # 物理损失：与正确公式的匹配度
                    physics_loss = torch.tensor(0.0).to(device)
                    gen_texts = tokenizer.batch_decode(gen_outputs["sequences"], skip_special_tokens=True)
                    for text, item in zip(gen_texts, batch):
                        if item["label"] == 1:  # 仅对正确样本计算逻辑损失
                            logic_score = discriminator.check_physics_logic(text)
                            physics_loss += (1 - logic_score)  # 逻辑越差，损失越大
                else:
                    # 角色损失：与风格特征的匹配度
                    role_loss = torch.tensor(0.0).to(device)
                    gen_texts = tokenizer.batch_decode(gen_outputs["sequences"], skip_special_tokens=True)
                    for text, item in zip(gen_texts, batch):
                        if item["label"] == 1:  # 仅对正确样本计算风格损失
                            style_score = discriminator.check_role_style(text)
                            role_loss += (1 - style_score)  # 风格越差，损失越大
                # 总生成器损失（对抗损失权重+任务损失权重）
                task_loss = physics_loss if args.task == "physics" else role_loss
                gen_loss = adv_loss * adv_config["adv_weight"] + task_loss * (1 - adv_config["adv_weight"])
                gen_loss.backward()
                gen_optim.step()
                total_gen_loss += gen_loss.item()

            # 6. 每轮更新状态和日志
            avg_gen_loss = total_gen_loss / num_batches
            avg_disc_loss = total_disc_loss / num_batches
            # 任务专属评估指标（物理：逻辑准确率；角色：风格覆盖率）
            if args.task == "physics":
                eval_metric = f"物理逻辑准确率：{100 - avg_gen_loss*10:.2f}%"
            else:
                eval_metric = f"角色风格覆盖率：{100 - avg_gen_loss*10:.2f}%"
            state_manager.update(epoch+1, avg_gen_loss, avg_disc_loss, eval_metric)
            logger.log(f"轮次{epoch+1}/{total_epochs} | 生成器损失：{avg_gen_loss:.4f} | 判别器损失：{avg_disc_loss:.4f} | {eval_metric}")

        # 7. 保存模型和判别器
        model.save_pretrained(args.output_dir)
        torch.save(discriminator.state_dict(), os.path.join(args.output_dir, "discrimin