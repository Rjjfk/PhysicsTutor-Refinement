import os
import json
import re
import torch
import numpy as np
from typing import Dict, List, Tuple
from datasets import load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from rouge import Rouge  # 用于计算步骤逻辑一致性（需安装：pip install rouge）


def load_physics_val_data(val_data_dir: str) -> List[Dict]:
    """加载物理模型验证集（预处理后的数据，含标准教师思维链）"""
    if not os.path.exists(val_data_dir):
        raise FileNotFoundError(f"物理验证集目录不存在: {val_data_dir}")
    # 加载process_physics.py生成的验证集（含raw_input/raw_output，即题干和标准解析）
    val_dataset = load_from_disk(val_data_dir)
    # 提取关键字段：题干、标准解析、标准结果（若有）
    val_data = []
    for sample in val_dataset:
        # 从raw_input中提取纯题干（去除指令前缀）
        raw_input = sample["raw_input"]
        question = re.sub(r"请严格按照物理规范分步骤解析以下问题.*：", "", raw_input).strip()
        # 标准教师思维链（含步骤、公式、结果）
        standard_answer = sample["raw_output"]
        # 提取标准结果（如速度、距离，需从standard_answer中匹配数值）
        standard_results = extract_physical_results(standard_answer)

        val_data.append({
            "question": question,
            "standard_answer": standard_answer,
            "standard_results": standard_results  # 标准数值结果（如{"v1": "(m-M)√(2gl)/(m+M)", "V": "2m√(2gl)/(m+M)"}）
        })
    print(f"加载物理验证集：共{len(val_data)}条样本")
    return val_data


def extract_physical_results(text: str) -> Dict:
    """从物理解析文本中提取关键数值结果（如速度、距离、碰撞次数）"""
    results = {}
    # 1. 匹配速度结果（如v1=xxx、V=xxx）
    speed_patterns = [
        r"v1\s*=\s*([^，。\n]+)",  # v1=...
        r"V\s*=\s*([^，。\n]+)",  # V=...
        r"速度\s*[:：]\s*([^，。\n]+)"  # 速度：...
    ]
    for pattern in speed_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            if "v1" in pattern.lower():
                results["v1（小球速度）"] = match.strip()
            elif "V" in pattern:
                results["V（圆盘速度）"] = match.strip()
            else:
                results["速度结果"] = match.strip()

    # 2. 匹配距离结果（如s=xxx、最远距离=xxx）
    distance_patterns = [
        r"s\s*=\s*([^，。\n]+)",  # s=...
        r"距离\s*[:：]\s*([^，。\n]+)",  # 距离：...
        r"最远距离\s*[:：]\s*([^，。\n]+)"  # 最远距离：...
    ]
    for pattern in distance_patterns:
        matches = re.findall(pattern, text, re.IGNORECASE)
        for match in matches:
            results["距离结果"] = match.strip()

    # 3. 匹配碰撞次数（如碰撞次数=xxx）
    collision_pattern = r"碰撞次数\s*[:：]\s*(\d+)"
    collision_matches = re.findall(collision_pattern, text)
    if collision_matches:
        results["碰撞次数"] = collision_matches[0].strip()

    return results


def evaluate_formula_accuracy(model_output: str, standard_answer: str) -> Tuple[float, Dict]:
    """
    评估公式准确性：
    - 公式覆盖度：模型输出是否包含所有关键公式
    - 公式正确性：公式参数/符号是否正确（如用l而非L、v0而非v1）
    """
    # 1. 定义当前物理场景的关键公式（可根据题型扩展）
    key_formulas = {
        "动量守恒": r"m\*v0\s*=\s*m\*v1\s*\+\s*M\*V",
        "动能守恒": r"\(1/2\)\*m\*v0²\s*=\s*\(1/2\)\*m\*v1²\s*\+\s*\(1/2\)\*M\*V²",
        "自由落体速度": r"v0²\s*=\s*2\*g\*l",
        "匀变速位移": r"s\s*=\s*v0\*t\s*\+\s*\(1/2\)\*a\*t²"
    }

    formula_metrics = {
        "公式覆盖度": 0.0,
        "公式正确性": 0.0,
        "覆盖详情": {},
        "错误详情": []
    }

    covered_count = 0
    correct_count = 0

    for formula_name, formula_pattern in key_formulas.items():
        # 检查公式是否覆盖
        if re.search(formula_pattern, model_output, re.IGNORECASE):
            covered_count += 1
            formula_metrics["覆盖详情"][formula_name] = "已覆盖"

            # 检查公式参数是否正确（如自由落体公式中是否用"l"而非"L"）
            if formula_name == "自由落体速度":
                # 错误案例：v0²=2*g*L（用了管长L而非下落高度l）
                wrong_param = re.search(r"v0²\s*=\s*2\*g\*L", model_output, re.IGNORECASE)
                if wrong_param:
                    formula_metrics["错误详情"].append(f"{formula_name}：参数错误（用L代替l）")
                else:
                    correct_count += 1
            else:
                # 其他公式暂按“覆盖即正确”（可根据需求扩展参数检查）
                correct_count += 1
        else:
            formula_metrics["覆盖详情"][formula_name] = "未覆盖"
            formula_metrics["错误详情"].append(f"{formula_name}：未包含该公式")

    # 计算公式覆盖度（已覆盖数/总公式数）
    formula_metrics["公式覆盖度"] = round((covered_count / len(key_formulas)) * 100, 2)
    # 计算公式正确性（正确数/已覆盖数，若未覆盖则为0）
    formula_metrics["公式正确性"] = round((correct_count / covered_count) * 100, 2) if covered_count > 0 else 0.0

    return formula_metrics["公式覆盖度"], formula_metrics


def evaluate_step_consistency(model_output: str, standard_answer: str) -> float:
    """评估步骤逻辑一致性：用ROUGE-L分数对比模型输出与标准思维链的步骤匹配度"""
    rouge = Rouge()
    try:
        # 提取模型和标准的“步骤文本”（去除公式和特殊符号，聚焦逻辑）
        model_steps = re.sub(r"【.*?】|\(1/2\)|[*²√=+\-\/()a-zA-Z0-9]+", "", model_output).strip()
        standard_steps = re.sub(r"【.*?】|\(1/2\)|[*²√=+\-\/()a-zA-Z0-9]+", "", standard_answer).strip()

        # 计算ROUGE-L分数（越高表示步骤逻辑越一致）
        score = rouge.get_scores(model_steps, standard_steps)[0]["rouge-l"]["f"]
        return round(score * 100, 2)
    except:
        # 解析失败时返回0
        return 0.0


def evaluate_result_correctness(model_results: Dict, standard_results: Dict) -> Tuple[float, Dict]:
    """评估结果正确性：对比模型输出的数值结果与标准结果（允许符号/格式微小差异）"""
    result_metrics = {
        "结果匹配度": 0.0,
        "匹配详情": {},
        "差异详情": []
    }

    if not standard_results:
        result_metrics["匹配详情"]["提示"] = "标准解析中无明确数值结果，无法评估"
        return 0.0, result_metrics

    matched_count = 0
    total_count = len(standard_results)

    for key, standard_val in standard_results.items():
        # 检查模型结果中是否有对应key
        model_val = model_results.get(key, "")
        if not model_val:
            result_metrics["匹配详情"][key] = f"模型未输出该结果（标准：{standard_val}）"
            result_metrics["差异详情"].append(f"{key}：模型未输出")
            continue

        # 简化对比（去除空格、括号，忽略大小写）
        model_val_simple = re.sub(r"[\s\(\)]", "", model_val).lower()
        standard_val_simple = re.sub(r"[\s\(\)]", "", standard_val).lower()

        # 检查核心表达式是否一致（如"(m-M)√(2gl)/(m+M)" 与 "(m-M)sqrt(2gl)/(m+M)" 视为一致）
        if model_val_simple in standard_val_simple or standard_val_simple in model_val_simple:
            result_metrics["匹配详情"][key] = f"匹配（模型：{model_val} | 标准：{standard_val}）"
            matched_count += 1
        else:
            result_metrics["匹配详情"][key] = f"不匹配（模型：{model_val} | 标准：{standard_val}）"
            result_metrics["差异详情"].append(f"{key}：模型={model_val}，标准={standard_val}")

    # 计算结果匹配度
    result_metrics["结果匹配度"] = round((matched_count / total_count) * 100, 2)
    return result_metrics["结果匹配度"], result_metrics


def evaluate_physics_model(
        model_path: str,
        val_data_dir: str,
        device: str = None
) -> Tuple[Dict, List[Dict]]:
    """
    物理模型完整评估流程：
    1. 公式准确性评估
    2. 步骤逻辑一致性评估
    3. 结果正确性评估
    """
    # 自动选择设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 1. 加载验证数据和模型
    val_data = load_physics_val_data(val_data_dir)
    print(f"加载物理模型：{model_path}（设备：{device}）")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    model.eval()

    # 2. 逐样本评估
    detailed_results = []
    total_formula_coverage = 0.0
    total_step_consistency = 0.0
    total_result_correctness = 0.0

    with torch.no_grad():
        for idx, sample in enumerate(val_data, 1):
            question = sample["question"]
            standard_answer = sample["standard_answer"]
            standard_results = sample["standard_results"]

            print(f"\n评估第{idx}/{len(val_data)}题：{question[:60]}...")

            # 3. 模型生成解析
            input_text = f"请严格按物理规范分步骤解析，包含公式、计算过程和最终结果：{question}"
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1,  # 低温度保证逻辑稳定
                do_sample=False,
                num_return_sequences=1
            )
            model_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 提取纯解析内容（去除输入前缀）
            if "解析：" in model_output:
                model_output = model_output.split("解析：")[-1].strip()

            # 4. 分项评估
            # 4.1 公式准确性
            formula_coverage, formula_metrics = evaluate_formula_accuracy(model_output, standard_answer)
            # 4.2 步骤逻辑一致性
            step_consistency = evaluate_step_consistency(model_output, standard_answer)
            # 4.3 结果正确性
            model_results = extract_physical_results(model_output)
            result_correctness, result_metrics = evaluate_result_correctness(model_results, standard_results)

            # 5. 综合得分（公式30% + 步骤40% + 结果30%）
            comprehensive_score = round(
                formula_coverage * 0.3 + step_consistency * 0.4 + result_correctness * 0.3,
                2
            )

            # 6. 保存单样本结果
            detailed_results.append({
                "题号": idx,
                "题干": question,
                "模型解析": model_output,
                "标准解析（摘要）": standard_answer[:200] + "..." if len(standard_answer) > 200 else standard_answer,
                "公式评估": formula_metrics,
                "步骤逻辑一致性（ROUGE-L）": f"{step_consistency}%",
                "结果评估": result_metrics,
                "综合得分": comprehensive_score
            })

            # 累加总指标
            total_formula_coverage += formula_coverage
            total_step_consistency += step_consistency
            total_result_correctness += result_correctness

            print(
                f"第{idx}题综合得分：{comprehensive_score}（公式：{formula_coverage}% | 步骤：{step_consistency}% | 结果：{result_correctness}%）")

    # 7. 计算总体指标
    total_samples = len(detailed_results)
    overall_metrics = {
        "评估样本总数": total_samples,
        "平均公式覆盖度": round(total_formula_coverage / total_samples, 2) if total_samples > 0 else 0.0,
        "平均步骤逻辑一致性": round(total_step_consistency / total_samples, 2) if total_samples > 0 else 0.0,
        "平均结果正确性": round(total_result_correctness / total_samples, 2) if total_samples > 0 else 0.0,
        "整体综合得分": round(
            (
                        total_formula_coverage * 0.3 + total_step_consistency * 0.4 + total_result_correctness * 0.3) / total_samples,
            2
        ) if total_samples > 0 else 0.0,
        "评估权重说明": "公式覆盖度30% + 步骤逻辑一致性40% + 结果正确性30%"
    }

    return overall_metrics, detailed_results


def generate_physics_eval_report(
        overall_metrics: Dict,
        detailed_results: List[Dict],
        report_path: str = "../../logs/physics_eval_report.json"
) -> str:
    """生成物理模型评估报告（JSON文件+可读文本）"""
    # 构建完整报告
    report = {
        "评估时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "评估模型路径": model_path,
        "总体指标": overall_metrics,
        "样本详细结果": detailed_results
    }

    # 保存JSON报告
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 生成可读文本报告
    text_report = f"""
================================ 物理模型评估报告 ================================
评估时间：{report["评估时间"]}
评估模型：{report["评估模型路径"]}
评估样本数：{overall_metrics["评估样本总数"]}

一、总体指标（权重：公式30% + 步骤40% + 结果30%）
1. 整体综合得分：{overall_metrics["整体综合得分"]}
2. 平均公式覆盖度：{overall_metrics["平均公式覆盖度"]}%
3. 平均步骤逻辑一致性：{overall_metrics["平均步骤逻辑一致性"]}%
4. 平均结果正确性：{overall_metrics["平均结果正确性"]}%

二、关键结论
"""
    # 根据总体得分生成结论
    if overall_metrics["整体综合得分"] >= 80:
        text_report += "✅ 模型物理解题能力优秀：公式覆盖完整、步骤逻辑清晰、结果准确性高，可直接用于后续角色训练。\n"
    elif overall_metrics["整体综合得分"] >= 60:
        text_report += "⚠️ 模型物理解题能力合格：但存在部分不足（如公式缺失/步骤跳跃/结果误差），建议补充同类数据微调。\n"
    else:
        text_report += "❌ 模型物理解题能力不足：需重新检查训练数据（如思维链完整性）或调整训练参数（如增加epoch）。\n"

    # 补充前3个样本的简要结果
    text_report += "\n三、样本示例（前3个）\n"
    for i, res in enumerate(detailed_results[:3], 1):
        text_report += f"""
{i}. 题干：{res["题干"][:80]}...
   综合得分：{res["综合得分"]}
   公式覆盖度：{res["公式评估"]["公式覆盖度"]}%
   步骤逻辑：{res["步骤逻辑一致性（ROUGE-L）"]}
   结果正确性：{res["结果评估"]["结果匹配度"]}%
"""

    text_report += f"\n================================ 报告保存路径 ================================\n{report_path}"
    return text_report


if __name__ == "__main__":
    # 配置参数（根据项目结构调整）
    model_path = "../../model/physics_lora"  # 物理模型路径
    val_data_dir = "../../data/processed/physics_train/val_dataset"  # 物理验证集路径
    report_path = "../../logs/physics_eval_report.json"  # 报告保存路径

    # 检查验证集是否存在（若不存在，提示先运行process_physics.py）
    if not os.path.exists(val_data_dir):
        raise FileNotFoundError(
            f"物理验证集不存在！请先运行process_physics.py生成：\n"
            f"python code/data/process_physics.py --config config/physics_config.json"
        )

    # 执行评估
    print("=== 开始物理模型专项评估 ===")
    overall_metrics, detailed_results = evaluate_physics_model(model_path, val_data_dir)

    # 生成报告
    print("\n=== 生成评估报告 ===")
    text_report = generate_physics_eval_report(overall_metrics, detailed_results, report_path)
    print(text_report)


def eval_physics_adversarial(model_path: str, test_error_data: List[Dict]) -> float:
    """
    对抗训练后专用评估：检查模型是否能拒绝错误物理样本
    返回值：错误样本识别率（越高越好）
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.float16, device_map="auto"
    )
    model.eval()

    correct_reject_count = 0
    with torch.no_grad():
        for item in test_error_data:
            input_text = f"解析物理题：{item['question']}"
            inputs = tokenizer(input_text, return_tensors="pt", max_length=512).to(device)
            outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
            gen_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 判断模型是否拒绝错误逻辑（如生成“该解析存在公式错误”）
            if "错误" in gen_text or "不正确" in gen_text or item["wrong_formula"] not in gen_text:
                correct_reject_count += 1

    return round((correct_reject_count / len(test_error_data)) * 100, 2)


# 调用示例（对抗训练后使用）
if __name__ == "__main__":
    # 原有评估逻辑...
    # 新增对抗评估（需提前准备错误样本数据）
    test_error_data = [
        {
            "question": "竖直圆管小球碰撞题",
            "wrong_formula": "v0²=2*g*L",  # 错误公式
            "expected_reject": True
        }
    ]
    reject_rate = eval_physics_adversarial("../../model/physics_adversarial_lora", test_error_data)
    print(f"错误物理样本识别率：{reject_rate}%")