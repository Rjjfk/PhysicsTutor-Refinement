import os
import json
import torch
from datasets import load_dataset
from .train_physics import train_physics_model  # 复用生成器训练逻辑
from .discriminator import PhysicsDiscriminator
from transformers import AutoModelForCausalLM, AutoTokenizer

def adversarial_train_loop(config):
    # 初始化生成器（基于物理模型）和判别器
    generator = AutoModelForCausalLM.from_pretrained(config["base_generator_path"])
    generator_tokenizer = AutoTokenizer.from_pretrained(config["base_generator_path"])
    discriminator = PhysicsDiscriminator(config["discriminator_base"])
    discriminator.to(generator.device)

    # 加载真实数据和无标注问题
    real_data = load_dataset("json", data_files=config["real_data_path"])["train"]
    unlabeled_questions = [item["question"] for item in load_dataset("json", data_files=config["unlabeled_data_path"])["train"]]

    for loop in range(config["loop_count"]):
        print(f"\n=== 对抗循环 {loop+1}/{config['loop_count']} ===")

        # Step 1: 生成器生成伪数据
        generator.eval()
        fake_data = []
        with torch.no_grad():
            for i in range(0, len(unlabeled_questions), config["generate_batch_size"]):
                batch_questions = unlabeled_questions[i:i+config["generate_batch_size"]]
                inputs = generator_tokenizer(
                    [f"解析物理题：{q}" for q in batch_questions],
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(generator.device)
                outputs = generator.generate(
                    **inputs,
                    max_length=1024,
                    temperature=config["gen_temperature"]
                )
                fake_answers = [generator_tokenizer.decode(o, skip_special_tokens=True) for o in outputs]
                fake_data.extend([{"input": q, "output": a} for q, a in zip(batch_questions, fake_answers)])

        # Step 2: 训练判别器区分真实/生成数据
        sampled_real = real_data.shuffle(seed=loop).select(range(min(len(fake_data), len(real_data))))
        discriminator = discriminator.train_discriminator(
            real_data=sampled_real,
            fake_data=fake_data,
            epochs=config["disc_epochs"]
        )

        # Step 3: 对抗训练生成器（欺骗判别器）
        generator.train()
        adv_optimizer = torch.optim.Adam(generator.parameters(), lr=config["gen_adv_lr"])
        criterion = torch.nn.BCELoss()

        # 用判别器的反馈优化生成器
        fake_texts = [f"物理解析：{item['output']}" for item in fake_data]
        inputs = generator_tokenizer(fake_texts, padding=True, truncation=True, return_tensors="pt").to(generator.device)
        disc_inputs = discriminator.tokenizer(fake_texts, padding=True, truncation=True, return_tensors="pt").to(generator.device)

        for _ in range(config["gen_adv_steps"]):
            adv_optimizer.zero_grad()
            # 生成器输出
            gen_outputs = generator(** inputs, labels=inputs["input_ids"])
            # 判别器对生成结果的判断
            disc_outputs = discriminator(disc_inputs["input_ids"], disc_inputs["attention_mask"])
            # 生成器损失：让判别器误以为是真实数据（目标=1）
            adv_loss = criterion(disc_outputs, torch.ones_like(disc_outputs))
            # 联合损失（生成损失+对抗损失）
            total_loss = gen_outputs.loss + config["adv_weight"] * adv_loss
            total_loss.backward()
            adv_optimizer.step()
            print(f"生成器对抗训练，Loss: {total_loss.item():.4f}")

        # Step 4: 保存本轮模型
        generator.save_pretrained(f"{config['output_root']}/generator_loop_{loop}")
        discriminator.save_pretrained(f"{config['output_root']}/discriminator_loop_{loop}")

    print("对抗式训练完成！")


if __name__ == "__main__":
    with open("../../config/adversarial_config.json", 'r') as f:
        config = json.load(f)
    adversarial_train_loop(config)