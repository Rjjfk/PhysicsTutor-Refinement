import os
import json
import torch
from typing import Dict, List, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer
# 修正角色适配器导入路径（根据项目结构）
from code.infer.role_adapter import role_adapter


def load_test_data(test_data_path: str) -> List[Dict]:
    """加载物理测试数据（需包含「物理题描述」和「预期评估指标」）"""
    if not os.path.exists(test_data_path):
        raise FileNotFoundError(f"测试数据文件不存在: {test_data_path}")
    with open(test_data_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    # 检查测试数据格式（确保包含必要字段）
    required_fields = ["物理题描述", "预期阶段", "预期公式"]
    for idx, item in enumerate(test_data, 1):
        if not all(f in item for f in required_fields):
            raise ValueError(f"测试数据{idx}缺失字段：{[f for f in required_fields if f not in item]}")
    return test_data


def evaluate_physics_capability(
        model_path: str,
        test_data: List[Dict],
        device: str = None
) -> Tuple[Dict, List[Dict]]:
    """
    评估物理模型核心能力：
    1. 阶段覆盖率：是否覆盖「审题→建模→计算→迭代」关键阶段
    2. 公式准确率：是否正确包含核心物理公式
    3. 逻辑完整性：是否按步骤拆解多阶段问题
    """
    # 自动选择设备（GPU优先）
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 加载模型和分词器
    print(f"加载物理模型：{model_path}（设备：{device}）")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map="auto"
    )
    model.eval()

    results = []
    total_samples = len(test_data)

    with torch.no_grad():
        for idx, item in enumerate(test_data, 1):
            question = item["物理题描述"]
            expected_stages = item["预期阶段"]  # 如["审题闭环", "建模闭环", "计算闭环", "迭代闭环"]
            expected_formulas = item["预期公式"]  # 如["动量守恒", "v0²=2*g*l", "s=v0*t+(1/2)*a*t²"]

            print(f"\n正在评估第{idx}/{total_samples}题：{question[:50]}...")

            # 1. 模型生成原始解析（无角色风格）
            input_text = f"请严格按物理规范分步骤解析：{question}"
            inputs = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=1024,
                temperature=0.1,  # 低温度保证逻辑稳定
                do_sample=False,
                num_return_sequences=1
            )
            raw_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 提取模型生成的解析部分（去除输入前缀）
            if "解析：" in raw_answer:
                raw_answer = raw_answer.split("解析：")[-1].strip()

            # 2. 评估阶段覆盖率
            stage_coverage = {stage: stage in raw_answer for stage in expected_stages}
            stage_pass = sum(stage_coverage.values()) / len(expected_stages) >= 0.75  # ≥3/4阶段视为通过

            # 3. 评估公式准确率
            formula_coverage = {formula: formula in raw_answer for formula in expected_formulas}
            formula_pass = sum(formula_coverage.values()) / len(expected_formulas) >= 0.5  # ≥1/2公式视为通过

            # 4. 综合判断是否通过（阶段和公式均通过）
            overall_pass = stage_pass and formula_pass

            # 5. 应用角色适配器（生成爱莉希雅风格输出，用于角色一致性评估）
            role_answer = role_adapter.adapt(raw_answer, role="elysia", difficulty=0.8)  # 高难度标记

            # 保存单题结果
            results.append({
                "题号": idx,
                "物理题描述": question,
                "原始解析": raw_answer,
                "角色风格解析": role_answer,
                "阶段覆盖": stage_coverage,
                "公式覆盖": formula_coverage,
                "阶段通过": stage_pass,
                "公式通过": formula_pass,
                "综合通过": overall_pass
            })

            print(f"第{idx}题评估结果：{'通过' if overall_pass else '失败'}（阶段：{stage_pass}，公式：{formula_pass}）")

    # 计算总体指标
    total_pass = sum(1 for res in results if res["综合通过"])
    overall_metrics = {
        "总体通过率": round((total_pass / total_samples) * 100, 2),
        "总测试题数": total_samples,
        "通过题数": total_pass,
        "阶段平均覆盖率": round(
            sum(sum(res["阶段覆盖"].values()) for res in results) / (total_samples * len(expected_stages)) * 100, 2
        ),
        "公式平均覆盖率": round(
            sum(sum(res["公式覆盖"].values()) for res in results) / (total_samples * len(expected_formulas)) * 100, 2
        )
    }

    return overall_metrics, results


def evaluate_role_consistency(results: List[Dict]) -> Dict:
    """评估角色风格一致性（爱莉希雅特征覆盖率）"""
    # 定义爱莉希雅核心特征（与role_adapter一致）
    role_features = {
        "关键词": ["飞花", "水晶", "舞会", "裙摆", "～♪"],
        "行为锚点": ["[轻转裙摆]", "[托腮歪头]", "[指尖绕发丝]"],
        "句式特征": ["呢～", "呀～", "对不对呀？"]
    }

    feature_metrics = {}
    total_samples = len(results)

    # 统计每个特征的覆盖率
    for feature_type, features in role_features.items():
        feature_counts = {feat: 0 for feat in features}
        for res in results:
            role_answer = res["角色风格解析"]
            for feat in features:
                if feat in role_answer:
                    feature_counts[feat] += 1
        # 计算该类型特征的平均覆盖率（百分比）
        avg_coverage = sum(feature_counts.values()) / (len(features) * total_samples) * 100
        feature_metrics[feature_type] = {
            "特征覆盖详情": feature_counts,
            "平均覆盖率": round(avg_coverage, 2)
        }

    # 综合角色一致性得分（各类型特征平均覆盖率的均值）
    overall_role_score = round(
        sum(metrics["平均覆盖率"] for metrics in feature_metrics.values()) / len(feature_metrics), 2
    )
    feature_metrics["综合角色一致性得分"] = overall_role_score

    return feature_metrics


def generate_test_report(
        physics_metrics: Dict,
        role_metrics: Dict,
        detailed_results: List[Dict],
        report_path: str = "../../logs/physics_role_test_report.json"
) -> str:
    """生成综合测试报告（含物理能力和角色一致性）"""
    # 构建报告内容
    report = {
        "测试时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "物理模型能力评估": physics_metrics,
        "角色风格一致性评估": role_metrics,
        "详细答题结果": detailed_results
    }

    # 保存报告到JSON文件
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 生成可读的文本报告
    text_report = f"""
================================ 物理+角色综合测试报告 ================================
1. 测试基本信息
   - 测试时间：{report["测试时间"]}
   - 测试题数：{physics_metrics["总测试题数"]}
   - 通过题数：{physics_metrics["通过题数"]}

2. 物理模型能力
   - 总体通过率：{physics_metrics["总体通过率"]}%
   - 阶段平均覆盖率：{physics_metrics["阶段平均覆盖率"]}%
   - 公式平均覆盖率：{physics_metrics["公式平均覆盖率"]}%

3. 角色风格一致性（爱莉希雅）
   - 综合角色得分：{role_metrics["综合角色一致性得分"]}%
   - 关键词平均覆盖率：{role_metrics["关键词"]["平均覆盖率"]}%
   - 行为锚点平均覆盖率：{role_metrics["行为锚点"]["平均覆盖率"]}%
   - 句式特征平均覆盖率：{role_metrics["句式特征"]["平均覆盖率"]}%

4. 失败案例概览（前5个）
"""
    failed_cases = [res for res in detailed_results if not res["综合通过"]][:5]
    if failed_cases:
        for case in failed_cases:
            text_report += f"""
   题号{case["题号"]}：
   问题：{case["物理题描述"][:80]}...
   失败原因：{'阶段覆盖不足' if not case["阶段通过"] else ''} {'公式缺失' if not case["公式通过"] else ''}
"""
    else:
        text_report += "\n   无失败案例，所有测试题均通过！"

    text_report += f"\n================================ 报告保存路径 ================================\n{report_path}"
    return text_report


if __name__ == "__main__":
    # 配置路径（根据项目结构调整）
    MODEL_PATH = "../../model/physics_lora"  # 物理模型路径（或角色模型路径../../model/elysia_lora）
    TEST_DATA_PATH = "../../data/raw/physics_test_data.json"  # 测试数据路径
    REPORT_PATH = "../../logs/physics_role_test_report.json"  # 报告保存路径

    # 1. 准备测试数据（示例格式，需提前创建physics_test_data.json）
    sample_test_data = [
        {
            "物理题描述": "竖直圆管内小球与圆盘弹性碰撞，求碰撞后速度和后续运动距离",
            "预期阶段": ["审题闭环", "建模闭环", "计算闭环", "迭代闭环"],
            "预期公式": ["动量守恒", "动能守恒", "v0²=2*g*l", "s=v0*t+(1/2)*a*t²"]
        }
    ]
    # 若测试数据文件不存在，创建示例文件
    if not os.path.exists(TEST_DATA_PATH):
        with open(TEST_DATA_PATH, "w", encoding="utf-8") as f:
            json.dump(sample_test_data, f, ensure_ascii=False, indent=2)
        print(f"已创建示例测试数据：{TEST_DATA_PATH}，请根据实际需求补充更多题目")

    # 2. 加载测试数据
    test_data = load_test_data(TEST_DATA_PATH)

    # 3. 评估物理能力
    print("\n=== 开始评估物理模型能力 ===")
    physics_metrics, detailed_results = evaluate_physics_capability(MODEL_PATH, test_data)

    # 4. 评估角色一致性（若使用角色模型，需将MODEL_PATH改为elysia_lora）
    print("\n=== 开始评估角色风格一致性 ===")
    role_metrics = evaluate_role_consistency(detailed_results)

    # 5. 生成并打印报告
    print("\n=== 生成综合测试报告 ===")
    text_report = generate_test_report(physics_metrics, role_metrics, detailed_results, REPORT_PATH)
    print(text_report)