import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Dict, List, Tuple
from role_adapter import elysia_adapter  # 导入角色适配器


def load_test_data(test_data_path: str) -> List[Dict]:
    """加载测试数据（需包含物理题和预期教师逻辑链）"""
    with open(test_data_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def evaluate_physics_capability(
        model_path: str,
        test_data: List[Dict],
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
) -> Tuple[Dict, List[Dict]]:
    """评估物理解题能力（教师逻辑链覆盖率）"""
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map=device
    )

    results = []
    for item in test_data:
        question = item["物理题描述"]
        expected_stages = item["教师逻辑链预期阶段"]  # 需在测试数据中定义

        # 模型生成原始回答
        inputs = tokenizer(f"解析物理题：{question}", return_tensors="pt").to(device)
        outputs = model.generate(
            inputs,
            max_length=1024,
            temperature=0.1,  # 低温度保证稳定性
            num_return_sequences=1
        )
        raw_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 检查是否覆盖所有预期阶段
        stage_coverage = {stage: stage in raw_answer for stage in expected_stages}
        has_all_stages = all(stage_coverage.values())

        results.append({
            "问题": question,
            "原始回答": raw_answer,
            "阶段覆盖率": stage_coverage,
            "是否覆盖所有阶段": has_all_stages
        })

    # 统计指标
    total = len(results)
    passed = sum(1 for r in results if r["是否覆盖所有阶段"])
    metrics = {
        "物理解题通过率": passed / total,
        "总测试题数": total,
        "通过数": passed
    }
    return metrics, results


def evaluate_role_consistency(
        physics_eval_results: List[Dict],
        required_keywords: List[str] = ["飞花", "水晶", "～♪", "英桀", "舞会"]
) -> Dict:
    """评估角色化一致性（爱莉希雅特征覆盖率）"""
    role_metrics = {keyword: 0 for keyword in required_keywords}
    total = len(physics_eval_results)

    for result in physics_eval_results:
        # 应用角色适配器，模拟最终输出
        role_answer = elysia_adapter.adapt(result["原始回答"], result["问题"])

        # 统计特征词出现次数
        for keyword in required_keywords:
            if keyword in role_answer:
                role_metrics[keyword] += 1

    # 计算平均覆盖率
    role_metrics["平均特征覆盖率"] = sum(role_metrics.values()) / (len(required_keywords) * total)
    return role_metrics


def generate_test_report(
        physics_metrics: Dict,
        role_metrics: Dict,
        physics_eval_details: List[Dict]
) -> str:
    """生成可视化测试报告"""
    report = f"""
    ===================== 模型综合能力测试报告 =====================
    1. 物理解题能力：
       - 通过率：{physics_metrics["物理解题通过率"]:.2%} 
       - 总题数：{physics_metrics["总测试题数"]} 
       - 通过数：{physics_metrics["通过数"]} 

    2. 角色化一致性（爱莉希雅特征）：
       {json.dumps(role_metrics, ensure_ascii=False, indent=2)} 

    3. 问题详情（前3个失败案例）：
    """
    failed_cases = [r for r in physics_eval_details if not r["是否覆盖所有阶段"]][:3]
    for i, case in enumerate(failed_cases, 1):
        report += f"""
    案例 {i}：
    问题：{case["问题"]}
    原始回答：{case["原始回答"][:100]}...（完整内容过长，略）
    缺失阶段：{[stage for stage, covered in case["阶段覆盖率"].items() if not covered]}
        """
    return report


if __name__ == "__main__":
    # 配置路径（根据实际项目结构调整）
    MODEL_PATH = "../../model/elysia_lora"  # 训练后的模型路径
    TEST_DATA_PATH = "../../data/raw/test_questions.json"  # 测试数据（需包含预期阶段）

    # 1. 加载测试数据
    test_data = load_test_data(TEST_DATA_PATH)

    # 2. 评估物理解题能力
    physics_metrics, physics_eval_details = evaluate_physics_capability(MODEL_PATH, test_data)

    # 3. 评估角色化一致性
    role_metrics = evaluate_role_consistency(physics_eval_details)

    # 4. 生成并打印报告
    test_report = generate_test_report(physics_metrics, role_metrics, physics_eval_details)
    print(test_report)

    # 5. 保存报告到日志（可选）
    with open("../../logs/comprehensive_test_report.txt", "w", encoding="utf-8") as f:
        f.write(test_report)