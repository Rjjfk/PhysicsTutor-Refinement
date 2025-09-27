import json
import re
from collections import Counter
from rouge import Rouge  # 需安装：pip install rouge


def evaluate_elysia(model_outputs_path: str, val_data_path: str) -> Dict:
    """
    评估爱莉希雅角色模型的两项核心指标：
    1. 物理逻辑准确率：生成内容与物理逻辑要求（instruction）的一致性
    2. 角色风格一致性：生成内容符合爱莉希雅角色特征的程度
    """
    # 1. 加载模型输出和验证集数据（含instruction字段）
    with open(model_outputs_path, 'r', encoding='utf-8') as f:
        model_outputs = json.load(f)  # 格式：[{"question": "...", "answer": "..."}]
    with open(val_data_path, 'r', encoding='utf-8') as f:
        val_data = json.load(f)  # 格式：[{"input": "...", "instruction": "...", "output": "..."}]

    # 构建问题→物理逻辑要求的映射（用于匹配评估）
    question_to_instruction = {
        item["input"]: item["instruction"]
        for item in val_data
    }

    # 2. 定义评估指标
    rouge = Rouge()  # 用于计算物理逻辑一致性
    elysia_features = {  # 角色特征与训练数据严格对齐
        "关键词": ["飞花", "水晶", "舞会", "裙摆", "花瓣", "音符", "旋律", "～♪"],
        "行为锚点": ["[轻转裙摆]", "[托腮歪头]", "[指尖绕发丝]", "[递出虚拟飞花]", "[旋转后比心]"],
        "句式特征": ["呢～", "呀～", "对不对呀？", "哦～", "啦～"]
    }

    results = []
    total_physics_score = 0.0
    total_role_score = 0.0

    # 3. 逐样本评估
    for output in model_outputs:
        question = output["question"]
        answer = output["answer"]

        # 3.1 物理逻辑准确率（与instruction对比）
        physics_instruction = question_to_instruction.get(question, "")
        physics_score = 0.0
        if physics_instruction and answer:
            try:
                # 提取答案中的物理逻辑部分（去除角色风格标记）
                answer_logic = re.sub(r'～♪|\[.*?\]|呀|呢|哟', '', answer).strip()
                # 计算ROUGE-L分数（衡量逻辑一致性）
                score = rouge.get_scores(answer_logic, physics_instruction)[0]["rouge-l"]["f"]
                physics_score = round(score * 100, 2)  # 转换为百分比
            except:
                physics_score = 0.0  # 解析失败时记0分

        # 3.2 角色风格一致性
        # 关键词匹配
        keyword_matches = sum(1 for kw in elysia_features["关键词"] if kw in answer)
        keyword_score = keyword_matches / len(elysia_features["关键词"])

        # 行为锚点匹配
        anchor_matches = sum(1 for anchor in elysia_features["行为锚点"] if anchor in answer)
        anchor_score = anchor_matches / len(elysia_features["行为锚点"])

        # 句式特征匹配
        sentence_matches = sum(len(re.findall(pattern, answer)) for pattern in elysia_features["句式特征"])
        total_sentences = max(1, len(re.findall(r'[。！？]', answer)))  # 避免除零
        sentence_score = sentence_matches / total_sentences

        # 综合角色分数（转换为百分比）
        role_score = round(((keyword_score + anchor_score + sentence_score) / 3) * 100, 2)

        # 3.3 保存单样本结果
        results.append({
            "question": question,
            "answer": answer,
            "physics_instruction": physics_instruction,
            "physics_score": physics_score,  # 物理逻辑准确率（%）
            "role_score": role_score,  # 角色风格一致性（%）
            "details": {
                "keyword_matches": keyword_matches,
                "anchor_matches": anchor_matches,
                "sentence_matches": sentence_matches
            }
        })

        total_physics_score += physics_score
        total_role_score += role_score

    # 4. 计算总体指标
    avg_physics = round(total_physics_score / len(results), 2) if results else 0.0
    avg_role = round(total_role_score / len(results), 2) if results else 0.0
    overall_score = round((avg_physics * 0.6 + avg_role * 0.4), 2)  # 物理权重更高

    # 5. 保存评估结果
    eval_report = {
        "overall_score": overall_score,  # 综合得分（物理60%+角色40%）
        "avg_physics_accuracy": avg_physics,  # 平均物理逻辑准确率
        "avg_role_consistency": avg_role,  # 平均角色风格一致性
        "sample_details": results
    }

    with open("../../logs/elysia_evaluation.json", 'w', encoding='utf-8') as f:
        json.dump(eval_report, f, ensure_ascii=False, indent=2)

    print(f"评估完成 | 综合得分: {overall_score} | 物理准确率: {avg_physics}% | 角色一致性: {avg_role}%")
    return eval_report


if __name__ == "__main__":
    # 评估入口：需传入模型输出路径和验证集数据路径（含instruction）
    evaluate_elysia(
        model_outputs_path="../../logs/elysia_model_outputs.json",  # 模型生成的答案
        val_data_path="../../data/processed/elysia/val.json"  # 含instruction的验证集
    )
