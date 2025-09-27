import os
import json
import torch
import argparse
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
# 复用已有的评估函数
from code.eval.eval_physics import evaluate_formula_accuracy, evaluate_step_accuracy
from code.eval.eval_elysia import evaluate_role_consistency, extract_physics_content


def load_test_data(data_path):
    """加载综合测试数据（包含物理题和角色对话）"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"测试数据路径不存在: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # 分离物理测试数据和角色测试数据
    physics_tests = [item for item in test_data if item["type"] == "physics"]
    role_tests = [item for item in test_data if item["type"] == "role"]

    print(f"加载测试数据完成：物理题{len(physics_tests)}道，角色对话{len(role_tests)}条")
    return physics_tests, role_tests


def evaluate_physical_accuracy(model, tokenizer, physics_tests, device):
    """评估模型的物理准确性"""
    metrics = {
        "公式准确率": [],
        "步骤完整性": [],
        "整体准确率": []
    }

    model.eval()
    with torch.no_grad():
        for item in physics_tests:
            question = item["question"]
            reference = item["reference_answer"]

            # 生成模型回答
            input_text = f"解析物理题：{question}"
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )
            model_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 评估公式准确性
            formula_score, _ = evaluate_formula_accuracy(model_answer, reference)
            # 评估步骤完整性
            step_metrics = evaluate_step_accuracy(model_answer, reference)
            step_score = step_metrics["步骤匹配率"]

            # 计算整体准确率（公式60% + 步骤40%）
            overall_score = 0.6 * (formula_score / 100) + 0.4 * step_score

            # 记录指标
            metrics["公式准确率"].append(formula_score)
            metrics["步骤完整性"].append(step_score * 100)
            metrics["整体准确率"].append(overall_score * 100)

    # 计算平均值
    return {
        "公式准确率": np.mean(metrics["公式准确率"]),
        "步骤完整性": np.mean(metrics["步骤完整性"]),
        "整体准确率": np.mean(metrics["整体准确率"])
    }


def evaluate_role_consistency(model, tokenizer, role_tests, device, style_features):
    """评估模型的角色风格一致性"""
    metrics = {
        "风格特征覆盖率": [],
        "物理内容准确率": [],
        "整体一致性": []
    }

    model.eval()
    with torch.no_grad():
        for item in role_tests:
            input_text = item["input"]  # 角色对话输入
            reference_physics = item["reference_physics"]  # 参考物理内容

            # 生成模型回答
            inputs = tokenizer(
                input_text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(device)

            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=False
            )
            model_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 评估风格一致性
            style_metrics = evaluate_role_consistency(model_answer, style_features)
            style_score = style_metrics["风格特征覆盖率"]

            # 提取并评估物理内容准确性
            physics_content = extract_physics_content(model_answer)
            formula_score, _ = evaluate_formula_accuracy(physics_content, reference_physics)
            physics_score = formula_score / 100  # 转为0-1范围

            # 计算整体一致性（风格50% + 物理50%）
            overall_score = 0.5 * style_score + 0.5 * physics_score

            # 记录指标
            metrics["风格特征覆盖率"].append(style_score * 100)
            metrics["物理内容准确率"].append(physics_score * 100)
            metrics["整体一致性"].append(overall_score * 100)

    # 计算平均值
    return {
        "风格特征覆盖率": np.mean(metrics["风格特征覆盖率"]),
        "物理内容准确率": np.mean(metrics["物理内容准确率"]),
        "整体一致性": np.mean(metrics["整体一致性"])
    }


def calculate_balance_score(physics_score, role_score):
    """
    计算平衡分数：同时考虑物理准确性和角色一致性
    使用调和平均，惩罚偏科严重的模型
    """
    if physics_score == 0 or role_score == 0:
        return 0.0
    # 物理权重稍高（60%），因为角色风格是在正确物理基础上的修饰
    return 2 * (0.6 * physics_score * 0.4 * role_score) / (0.6 * physics_score + 0.4 * role_score)


def eval_combined(model_path, test_data_path, style_features_path):
    """综合评估入口函数"""
    # 1. 加载模型和分词器
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备评估：{device}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    model.eval()

    # 2. 加载角色风格特征
    with open(style_features_path, "r", encoding="utf-8") as f:
        style_config = json.load(f)
    style_features = style_config["role_adversarial"]["style_features"]

    # 3. 加载测试数据
    physics_tests, role_tests = load_test_data(test_data_path)

    # 4. 评估物理准确性
    print("\n===== 开始物理准确性评估 =====")
    physics_metrics = evaluate_physical_accuracy(model, tokenizer, physics_tests, device)
    print(f"物理公式准确率：{physics_metrics['公式准确率']:.2f}%")
    print(f"物理步骤完整性：{physics_metrics['步骤完整性']:.2f}%")
    print(f"物理整体准确率：{physics_metrics['整体准确率']:.2f}%")

    # 5. 评估角色一致性
    print("\n===== 开始角色一致性评估 =====")
    role_metrics = evaluate_role_consistency(model, tokenizer, role_tests, device, style_features)
    print(f"风格特征覆盖率：{role_metrics['风格特征覆盖率']:.2f}%")
    print(f"角色物理准确率：{role_metrics['物理内容准确率']:.2f}%")
    print(f"角色整体一致性：{role_metrics['整体一致性']:.2f}%")

    # 6. 计算平衡分数
    balance_score = calculate_balance_score(
        physics_metrics["整体准确率"],
        role_metrics["整体一致性"]
    )
    print(f"\n===== 综合评估结果 =====")
    print(f"物理整体准确率：{physics_metrics['整体准确率']:.2f}%")
    print(f"角色整体一致性：{role_metrics['整体一致性']:.2f}%")
    print(f"平衡分数（越高越均衡）：{balance_score:.2f}%")

    # 7. 保存评估结果
    result = {
        "物理评估指标": physics_metrics,
        "角色评估指标": role_metrics,
        "平衡分数": balance_score,
        "评估时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    with open("combined_evaluation_result.json", "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print("\n评估结果已保存至：combined_evaluation_result.json")

    return result


if __name__ == "__main__":
    from datetime import datetime

    parser = argparse.ArgumentParser(description="合并模型综合评估脚本")
    parser.add_argument("--model_path", type=str, required=True,
                        help="合并后的模型路径")
    parser.add_argument("--test_data", type=str, required=True,
                        help="综合测试数据路径（JSON文件）")
    parser.add_argument("--style_config", type=str,
                        default="../../config/elysia_config.json",
                        help="角色风格配置文件路径")
    args = parser.parse_args()

    eval_combined(
        model_path=args.model_path,
        test_data_path=args.test_data,
        style_features_path=args.style_config
    )
