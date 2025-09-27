import random
import re
from typing import Dict, List, Optional, Tuple


class RoleAdapter:
    """角色适配器，将模型输出转换为不同角色的引导风格"""

    def __init__(self):
        """初始化角色配置"""
        # 爱莉希雅专属配置（基于游戏人设+物理教学场景）
        self.elysia_config = {
            "suffixes": ["～♪", "呀", "呢", "哟"],  # 标志性后缀
            "actions": [
                "[轻转裙摆]", "[托腮歪头]", "[指尖绕发丝]",
                "[递出虚拟飞花]", "[旋转后比心]", "[单膝跪地]"
            ],  # 行为锚点
            "metaphors": {  # 物理概念→爱莉希雅比喻
                "动量守恒": "双人舞的默契步伐～",
                "动能守恒": "水晶绽放的活力～",
                "摩擦力": "不想放手的小妖精～",
                "加速度": "突然加快的舞步节奏～",
                "弹性势能": "被握紧的花瓣能量～",
                "牛顿定律": "物理世界的铁律誓言～"
            },
            "stage_mapping": {  # 物理步骤→爱莉希雅"舞会流程"
                "【审题闭环】": "【舞会邀请·审题】💐 先看看舞伴是谁呀～",
                "【建模闭环】": "【舞步设计·建模】💎 给它们设计专属符号吧～",
                "【计算闭环】": "【共舞计算·计算】💃 让数字跳支圆舞曲～",
                "【迭代闭环】": "【谢幕迭代·迭代】🌸 检查下一支舞的节奏哦～"
            },
            "negative_replace": {  # 负面词→治愈系表达
                "错误": "小偏差",
                "失败": "暂时迷路",
                "遗漏": "没注意到的小花瓣",
                "忽略": "暂时忘记了",
                "困难": "有趣的挑战"
            },
            "canonical_phrases": [  # 经典台词
                "爱的少女心，可是无所不能的哦～♪",
                "要心怀感激地收下这束飞花呀！",
                "无论何时何地，爱莉希雅都会回应你的期待～",
                "猜猜我在想什么？是与你共舞的邀请哟♪",
                "前行的道路有群星闪耀，你即是上帝的馈赠",
                "藏着太多秘密...但别担心，我始终在你身边"
            ]
        }

        # 鼓励者角色配置
        self.encourager_config = {
            "prefix": "太棒了！我们一起来分析这道题：\n\n",
            "suffix": "\n\n你已经掌握了关键思路，继续加油！如果有疑问随时问我～",
            "stage_mapping": {
                "【审题闭环】": "【审题闭环】💡 先明确题目类型和已知条件：",
                "【建模闭环】": "【建模闭环】🔧 选择合适的物理规律：",
                "【计算闭环】": "【计算闭环】✖️➗ 联立方程求解：",
                "【迭代闭环】": "【迭代闭环】🔄 检查是否有后续物理过程："
            }
        }

        # 详细解释者角色配置
        self.detailed_config = {
            "prefix": "让我们一步步拆解这道题，确保每个细节都理解：\n\n",
            "suffix": "\n\n需要我解释哪个步骤的细节吗？",
            "stage_mapping": {
                "【审题闭环】": "【审题闭环】📝 详细提取已知条件：",
                "【建模闭环】": "【建模闭环】📌 严格定义物理量：",
                "【计算闭环】": "【计算闭环】🔢 逐步推导公式：",
                "【迭代闭环】": "【迭代闭环】🔍 验证每一步逻辑："
            }
        }

        # 默认角色配置
        self.default_config = {
            "prefix": "解题思路如下：\n\n",
            "suffix": ""
        }

    def adapt(self, response: str, role: str = "default", difficulty: float = 0.5) -> str:
        """
        将模型输出适配为指定角色的引导风格

        Args:
            response: 原始模型输出
            role: 角色类型，可选值："elysia", "encourager", "detailed", "default"
            difficulty: 问题难度(0-1)，仅对爱莉希雅角色有效
        """
        if role == "elysia":
            return self._adapt_elysia(response, difficulty)
        elif role == "encourager":
            return self._adapt_encourager(response)
        elif role == "detailed":
            return self._adapt_detailed(response)
        else:  # default
            return self._adapt_default(response)

    def _adapt_elysia(self, response: str, difficulty: float) -> str:
        """爱莉希雅角色适配"""
        # 1. 添加行为前缀（根据难度调整）
        if difficulty > 0.7:  # 高难度问题
            action = random.choice([
                "[水晶蔷薇绽放]", "[指尖凝聚星光]",
                "[单膝跪地]", "[旋转裙摆扬起飞花]"
            ])
            prefix = f"{action} 这道题可是很有挑战性的呢～就像和崩坏战斗一样刺激！让我们一起攻克它吧～♪\n\n"
        else:  # 普通难度
            action = random.choice(self.elysia_config["actions"])
            prefix = f"{action} 物理题呀～就像一场华丽的舞会呢～让我们一起解开它吧～♪\n\n"

        # 2. 添加标志性后缀
        suffix = f"\n\n怎么样？是不是和飞花绽放一样有趣呀～{random.choice(self.elysia_config['suffixes'])}"

        # 3. 替换阶段标签
        for original_stage, elysia_stage in self.elysia_config["stage_mapping"].items():
            response = response.replace(original_stage, elysia_stage)

        # 4. 注入比喻化解释
        for concept, metaphor in self.elysia_config["metaphors"].items():
            if concept in response:
                response = response.replace(
                    concept, f"{concept}（就是{metaphor}）"
                )

        # 5. 替换负面词汇
        for negative, positive in self.elysia_config["negative_replace"].items():
            response = response.replace(negative, positive)

        # 6. 随机插入经典台词（每3次调用插入一次）
        if random.random() < 0.33:
            phrases = self.elysia_config["canonical_phrases"]
            insert_position = random.choice([
                int(len(response) * 0.25),  # 前1/4处
                int(len(response) * 0.5),  # 中间
                int(len(response) * 0.75)  # 后1/4处
            ])
            response = (
                    response[:insert_position] +
                    f"\n\n{random.choice(phrases)}\n\n" +
                    response[insert_position:]
            )

        # 7. 增强疑问句比例
        sentences = re.split(r'[。！？]', response)
        new_sentences = []
        for s in sentences:
            if s and random.random() < 0.4:  # 40%概率添加疑问
                s = s.rstrip('。！？') + "，对不对呀？"
            new_sentences.append(s)
        response = ''.join(new_sentences)

        # 8. 确保句尾有标志性后缀
        if not any(suffix in response[-5:] for suffix in self.elysia_config["suffixes"]):
            response += random.choice(self.elysia_config["suffixes"])

        return prefix + response + suffix

    def _adapt_encourager(self, response: str) -> str:
        """鼓励者角色适配"""
        config = self.encourager_config
        for original, replacement in config["stage_mapping"].items():
            response = response.replace(original, replacement)
        return config["prefix"] + response + config["suffix"]

    def _adapt_detailed(self, response: str) -> str:
        """详细解释者角色适配"""
        config = self.detailed_config
        for original, replacement in config["stage_mapping"].items():
            response = response.replace(original, replacement)
        return config["prefix"] + response + config["suffix"]

    def _adapt_default(self, response: str) -> str:
        """默认角色适配"""
        config = self.default_config
        return config["prefix"] + response + config["suffix"]


# 实例化适配器（单例模式）
role_adapter = RoleAdapter()