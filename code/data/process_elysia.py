import json
import os
import argparse
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split


def process_elysia_data(config_path):
    """
    处理爱莉希雅角色风格数据：
    1. 加载原始数据和配置参数
    2. 格式化数据结构（适配input/instruction/output字段）
    3. 截断过长序列（按config中的max_length）
    4. 划分训练集和验证集（按config中的train_val_split）
    """
    # 1. 加载配置参数
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    preprocess_params = config.get("preprocess_params", {})

    # 配置参数（从config读取，默认值确保兼容性）
    RAW_DATA_PATH = preprocess_params.get("raw_data_path", "../data/raw/elysia_role_data.json")
    OUTPUT_DIR = preprocess_params.get("processed_data_dir", "../data/processed/elysia")
    MAX_LENGTH = preprocess_params.get("max_length", 1024)
    TRAIN_VAL_SPLIT = preprocess_params.get("train_val_split", 0.9)
    TOKENIZER_PATH = preprocess_params.get("tokenizer_path", "../model/base")  # 基础模型的tokenizer路径

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"预处理参数加载完成：")
    print(f"  原始数据路径: {RAW_DATA_PATH}")
    print(f"  输出目录: {OUTPUT_DIR}")
    print(f"  最大序列长度: {MAX_LENGTH}")
    print(f"  训练/验证集比例: {TRAIN_VAL_SPLIT}:{1 - TRAIN_VAL_SPLIT}")

    # 2. 加载原始数据（适配新字段：input/instruction/output）
    if not os.path.exists(RAW_DATA_PATH):
        raise FileNotFoundError(f"原始角色数据文件不存在: {RAW_DATA_PATH}")
    with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)
    print(f"加载原始角色数据: {len(raw_data)} 条")

    # 3. 加载tokenizer用于截断过长文本
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 4. 格式化并清洗数据
    processed_data = []
    for idx, item in enumerate(raw_data):
        # 检查必要字段（确保与修改后的elysia_role_data.json匹配）
        required_fields = ["input", "instruction", "output"]
        if not all(field in item for field in required_fields):
            print(f"跳过无效样本 {idx + 1}：缺少字段 {[f for f in required_fields if f not in item]}")
            continue

        # 构建完整输入（整合input和instruction，供模型学习物理逻辑+角色风格）
        full_input = f"物理题场景：{item['input']}\n解析要求：{item['instruction']}"
        full_output = item["output"]

        # 截断过长序列（合并input和output计算总长度，避免训练时溢出）
        combined_text = f"{full_input}\n{full_output}"
        tokenized_length = len(tokenizer.tokenize(combined_text))

        if tokenized_length > MAX_LENGTH:
            # 优先先截断output，保留核心逻辑和角色风格
            output_tokens = tokenizer.tokenize(full_output)
            input_tokens = tokenizer.tokenize(full_input)
            input_length = len(input_tokens)
            remaining_length = MAX_LENGTH - input_length - 5  # 预留分隔符空间

            if remaining_length > 0:
                truncated_output = tokenizer.convert_tokens_to_string(output_tokenstokens[:remaining_length])
                full_output = truncated_output
                print(f"截断样本 {idx + 1}：原始长度 {tokenized_length} → 截断后 {input_length + remaining_length}")
            else:
                print(f"跳过过长样本 {idx + 1}：输入部分已超过最大长度")
                continue

        # 保存格式化后的数据（与train_elysia.py的preprocess_function兼容）
        processed_data.append({
            "input": full_input,
            "instruction": item["instruction"],  # 保留原始逻辑提示，用于评估
            "output": full_output
        })

    print(f"数据清洗完成：有效样本 {len(processed_data)} 条（过滤无效/过长样本 {len(raw_data) - len(processed_data)} 条）")

    # 5. 划分训练集和验证集
    train_data, val_data = train_test_split(
        processed_data,
        test_size=1 - TRAIN_VAL_SPLIT,
        random_state=42  # 固定随机种子，确保划分一致
    )
    print(f"数据集划分：训练集 {len(train_data)} 条，验证集 {len(val_data)} 条")

    # 6. 保存处理后的数据（供train_elysia.py加载）
    train_path = os.path.join(OUTPUT_DIR, "train.json")
    val_path = os.path.join(OUTPUT_DIR, "val.json")

    with open(train_path, "w", encoding="utf-8") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(val_path, "w", encoding="utf-8") as f:
        json.dump(val_data, f, ensure_ascii=False, indent=2)

    print(f"处理完成！数据已保存至：")
    print(f"  训练集: {train_path}")
    print(f"  验证集: {val_path}")


if __name__ == "__main__":
    # 支持命令行参数指定配置文件路径
    parser = argparse.ArgumentParser(description="爱莉希雅角色数据预处理脚本")
    parser.add_argument("--config", type=str, default="../config/elysia_config.json",
                        help="配置文件路径（默认：../config/elysia_config.json）")
    args = parser.parse_args()

    process_elysia_data(args.config)
