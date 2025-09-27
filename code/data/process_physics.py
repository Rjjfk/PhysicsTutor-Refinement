import json
import os
import argparse
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from datasets import Dataset


def process_physics_data(config_path: str):
    """
    处理物理教师思维链数据：
    1. 从physics_config.json读取参数（与训练脚本联动）
    2. 格式化数据为"题干→分步解析+解题闭环"结构（保留物理逻辑）
    3. 划分训练集/验证集（评估泛化能力）
    4. 生成Hugging Face Dataset格式（供train_physics.py直接加载）
    """
    # 1. 加载配置参数（确保与物理训练配置统一）
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"物理训练配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    preprocess_params = config.get("preprocess_params", {})

    # 解析配置（含默认值，避免参数缺失）
    RAW_DATA_PATH = preprocess_params.get("raw_data_path", "../data/raw/teacher_chain.json")
    OUTPUT_DIR = preprocess_params.get("processed_data_dir", "../data/processed/physics_train")
    MAX_LENGTH = preprocess_params.get("max_length", 1024)  # 与ChatGLM-6B上下文匹配
    TRAIN_VAL_SPLIT = preprocess_params.get("train_val_split", 0.9)  # 9:1划分
    TOKENIZER_PATH = preprocess_params.get("tokenizer_path", "../model/base")  # 基础模型路径

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"=== 物理数据预处理参数 ===")
    print(f"原始思维链数据：{RAW_DATA_PATH}")
    print(f"输出目录：{OUTPUT_DIR}")
    print(f"最大序列长度：{MAX_LENGTH}")
    print(f"训练/验证划分：{TRAIN_VAL_SPLIT}:{1 - TRAIN_VAL_SPLIT}")

    # 2. 加载原始教师思维链数据
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"教师思维链数据不存在: {RAW_DATA_PATH}")
    with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    print(f"\n加载原始数据：共{len(raw_data)}条物理题思维链")

    # 3. 初始化分词器（与基础模型一致，补充pad_token）
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"分词器加载完成：{TOKENIZER_PATH}")

    # 4. 格式化数据（保留物理逻辑关键信息+解题闭环）
    formatted_samples = []
    for idx, item in enumerate(raw_data, 1):
        # 检查核心字段完整性
        required_fields = ["物理题描述", "教师思维链", "解题闭环逻辑"]
        if not all(field in item for field in required_fields):
            print(f"跳过无效样本{idx}：缺失字段{[f for f in required_fields if f not in item]}")
            continue

        # 构建输入：明确任务指令+题干（引导模型聚焦物理解题）
        input_text = f"请严格按照物理规范分步骤解析以下问题，需包含「阶段标签、思考步骤、知识锚点、易错点」，最后总结解题闭环逻辑：\n{item['物理题描述']}"

        # 构建输出：分步思维链 + 解题闭环总结
        output_segments = []
        # 处理每阶段思维链
        for stage in item["教师思维链"]:
            stage_required = ["阶段", "思考步骤", "知识锚点", "学生易错点"]
            if not all(f in stage for f in stage_required):
                print(f"样本{idx}的{stage.get('阶段', '未知阶段')}字段缺失，跳过该阶段")
                continue
            # 格式化单阶段内容（增强可读性，帮助模型区分模块）
            stage_text = (
                    f"【{stage['阶段']}】\n"
                    f"思考步骤：\n" + "\n".join([f"- {step}" for step in stage["思考步骤"]]) + "\n"
                                                                                              f"知识锚点：{stage['知识锚点']}\n"
                                                                                              f"易错提醒：{stage['学生易错点']}"
            )
            output_segments.append(stage_text)

        # 补充解题闭环总结（新增：让模型学习通用解题框架）
        output_segments.append(f"【解题闭环总结】\n{item['解题闭环逻辑']}")
        output_text = "\n\n".join(output_segments)  # 阶段间空行分隔，避免混淆

        # 5. Tokenize处理（区分输入/输出，仅计算输出损失）
        full_text = f"{input_text}\n\n### 解析开始 ###\n{output_text}"
        tokenized = tokenizer(
            full_text,
            max_length=MAX_LENGTH,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        # 标记输入部分（设为-100，不参与损失计算）
        sep_tokens = tokenizer.encode("### 解析开始 ###", add_special_tokens=False)
        sep_pos = None
        # 找到分隔符位置（确保输入/输出边界正确）
        for i in range(len(tokenized["input_ids"][0]) - len(sep_tokens) + 1):
            if tokenized["input_ids"][0][i:i + len(sep_tokens)].tolist() == sep_tokens:
                sep_pos = i + len(sep_tokens)
                break
        # 保底逻辑：若未找到分隔符，前1/3设为输入
        if sep_pos is None:
            sep_pos = len(tokenized["input_ids"][0]) // 3

        # 构建labels：输入部分设为-100
        labels = tokenized["input_ids"].clone()
        labels[0, :sep_pos] = -100

        # 保存样本（含原始文本，方便调试）
        formatted_samples.append({
            "input_ids": tokenized["input_ids"][0].tolist(),
            "attention_mask": tokenized["attention_mask"][0].tolist(),
            "labels": labels[0].tolist(),
            "raw_input": input_text,
            "raw_output": output_text
        })

    # 检查有效样本数量
    if len(formatted_samples) == 0:
        raise ValueError("无有效预处理样本，请检查原始数据格式")
    print(
        f"\n数据格式化完成：有效样本{len(formatted_samples)}条（过滤{len(raw_data) - len(formatted_samples)}条无效样本）")

    # 6. 划分训练集和验证集（固定随机种子，确保结果可复现）
    train_samples, val_samples = train_test_split(
        formatted_samples,
        test_size=1 - TRAIN_VAL_SPLIT,
        random_state=42
    )
    print(f"数据集划分：训练集{len(train_samples)}条，验证集{len(val_samples)}条")

    # 7. 保存为Dataset格式（供train_physics.py加载）
    train_dataset = Dataset.from_list(train_samples)
    val_dataset = Dataset.from_list(val_samples)

    train_save_path = os.path.join(OUTPUT_DIR, "train_dataset")
    val_save_path = os.path.join(OUTPUT_DIR, "val_dataset")
    train_dataset.save_to_disk(train_save_path)
    val_dataset.save_to_disk(val_save_path)

    print(f"\n=== 预处理完成 ===")
    print(f"训练集保存至：{train_save_path}")
    print(f"验证集保存至：{val_save_path}")


if __name__ == "__main__":
    # 支持命令行指定配置文件（与物理训练脚本统一入口）
    parser = argparse.ArgumentParser(description="物理教师思维链数据预处理脚本")
    parser.add_argument("--config", type=str, default="../config/physics_config.json",
                        help="物理训练配置文件路径（默认：../config/physics_config.json）")
    args = parser.parse_args()

    process_physics_data(args.config)