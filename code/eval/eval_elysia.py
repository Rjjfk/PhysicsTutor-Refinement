import json
import re
from collections import Counter


def evaluate_elysia_consistency(model_outputs_path):
    # 加载模型输出结果
    with open(model_outputs_path, 'r', encoding='utf-8') as f:
        model_outputs = json.load(f)

    # 定义爱莉希雅角色特征
    elysia_features = {
        "关键词": ["飞花", "水晶", "舞会", "裙摆", "英桀", "～♪"],
        "行为锚点": ["[轻转裙摆]", "[托腮歪头]", "[指尖绕发丝]"],
        "句式特征": ["呢～", "呀～", "对不对呀？"]
    }

    results = []

    for output in model_outputs:
        answer = output["answer"]

        # 计算关键词出现频率
        keyword_counts = {
            keyword: 1 if keyword in answer else 0
            for keyword in elysia_features["关键词"]
        }
        keyword_score = sum(keyword_counts.values()) / len(elysia_features["关键词"])

        # 计算行为锚点出现频率
        anchor_counts = {
            anchor: 1 if anchor in answer else 0
            for anchor in elysia_features["行为锚点"]
        }
        anchor_score = sum(anchor_counts.values()) / len(elysia_features["行为锚点"])

        # 计算句式特征出现频率
        sentence_patterns = {
            pattern: len(re.findall(pattern, answer))
            for pattern in elysia_features["句式特征"]
        }
        sentence_score = sum(sentence_patterns.values()) / max(1, len(re.findall(r'[。！？]', answer)))

        # 计算总体角色一致性分数
        consistency_score = (keyword_score + anchor_score + sentence_score) / 3

        results.append({
            "answer": answer,
            "keyword_counts": keyword_counts,
            "anchor_counts": anchor_counts,
            "sentence_patterns": sentence_patterns,
            "consistency_score": consistency_score
        })

    # 计算平均分数
    avg_consistency = sum(r["consistency_score"] for r in results) / len(results)

    print(f"平均角色一致性分数：{avg_consistency:.2%}")

    # 保存结果
    with open("../../logs/elysia_evaluation.json", 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return {
        "avg_consistency": avg_consistency,
        "details": results
    }


if __name__ == "__main__":
    evaluate_elysia_consistency(
        model_outputs_path="../../logs/physics_evaluation.json"  # 使用物理评估结果
    )