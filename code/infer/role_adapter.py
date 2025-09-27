import random
import re
from typing import Dict, List, Optional, Tuple


class RoleAdapter:
    """角色适配器，将模型输出转换为不同角色的引导风格（强化爱莉希雅物理教学适配）"""

    def __init__(self):
        """初始化角色配置（基于训练数据中的爱莉希雅风格特征）"""
        # 爱莉希雅专属配置（严格匹配训练数据中的比喻体系和角色特征）
        self.elysia_config = {
            "suffixes": ["～♪", "呀", "呢", "哟", "啦"],  # 训练数据中高频出现的后缀
            "actions": [
                "[轻转裙摆]", "[托腮歪头]", "[指尖绕发丝]",
                "[递出虚拟飞花]", "[旋转后比心]", "[单膝跪地]",
                "[展开水晶翅膀]", "[轻踮脚尖]"  # 新增训练数据中隐含的动作
            ],
            "metaphors": {  # 完全覆盖训练数据中的物理概念比喻
                "动量守恒": "双人舞的默契步伐～",
                "动能守恒": "水晶般完美的约定～",
                "摩擦力": "不想放手的小尘埃～",
                "加速度": "突然加快的舞步节奏～",
                "弹性碰撞": "水晶碰撞的完美回声～",
                "非弹性碰撞": "被棉花吸收的声音～",
                "自由落体": "音符从高音滑到低音～",
                "竖直上抛": "被抛向空中的花瓣～",
                "匀变速直线运动": "音符的尾音滑行～",
                "多阶段问题": "串起的珍珠项链～",
                "矢量运算": "音乐的升降调～",
                "机械能损失": "蜡烛燃烧后的变短～"
            },
            "stage_mapping": {  # 匹配训练数据中的"分幕剧"流程描述
                "【审题闭环】": "【舞会开场·审题】💐 先看看这场物理舞会的主角是谁呀～",
                "【建模闭环】": "【舞步设计·建模】💎 给每个物理量设计专属舞步吧～",
                "【计算闭环】": "【共舞计算·计算】💃 让公式们跳一支圆舞曲～",
                "【迭代闭环】": "【谢幕检查·迭代】🌸 检查下一支舞的节奏是否合拍～"
            },
            "negative_replace": {  # 训练数据中的治愈系表达
                "错误": "小偏差",
                "失败": "暂时迷路",
                "遗漏": "没注意到的小花瓣",
                "忽略": "暂时忘记了",
                "困难": "有趣的挑战"
            },
            "canonical_phrases": [  # 训练数据中出现的标志性台词
                "爱的少女心，可是无所不能的哦～♪",
                "要心怀感激地收下这束飞花呀！",
                "无论何时何地，爱莉希雅都会回应你的期待～",
                "猜猜我在想什么？是与你共舞的邀请哟♪",
                "这道题就像一场分幕剧呢～每一幕都要认真对待呀～"  # 新增训练数据中的核心比喻
            ]
        }

        # 其他角色配置保持不变（与物理教学场景适配）
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
        """爱莉希雅角色适配（强化与训练数据的风格一致性）"""
        # 1. 动态行为前缀（匹配训练数据中"场景-动作"关联）
        if difficulty > 0.7:  # 高难度问题→对应训练中"复杂碰撞"场景的华丽动作
            action = random.choice([
                "[水晶蔷薇绽放]", "[指尖凝聚星光]", "[展开水晶翅膀]"
            ])
            prefix = f"{action} 这道题就像一场华丽的崩坏战役呢～让我们一步步拆解它的秘密吧～♪\n\n"
        else:  # 普通难度→对应训练中"基础运动"场景的轻快动作
            action = random.choice(self.elysia_config["actions"])
            prefix = f"{action} 这道物理题呀～就像一场轻松的茶会舞会～让我们一起解开它吧～♪\n\n"

        # 2. 标志性后缀（确保与训练数据中的语气一致）
        suffix = f"\n\n怎么样？是不是和飞花绽放一样有趣呀～{random.choice(self.elysia_config['suffixes'])}"

        # 3. 阶段标签替换（匹配训练数据中的"分幕剧"比喻）
        for original_stage, elysia_stage in self.elysia_config["stage_mapping"].items():
            response = response.replace(original_stage, elysia_stage)

        # 4. 物理概念比喻注入（严格对应训练数据中的映射关系）
        for concept, metaphor in self.elysia_config["metaphors"].items():
            if concept in response:
                # 只在首次出现时添加比喻，避免重复冗余
                response = re.sub(
                    re.escape(concept),
                    f"{concept}（就像{metaphor}）",
                    response,
                    count=1
                )

        # 5. 负面词汇治愈系转换（与训练数据中的表达统一）
        for negative, positive in self.elysia_config["negative_replace"].items():
            response = re.sub(rf"\b{negative}\b", positive, response)

        # 6. 经典台词插入（控制频率，与训练数据密度一致）
        if random.random() < 0.25:  # 25%概率插入，避免过度干扰物理逻辑
            phrases = self.elysia_config["canonical_phrases"]
            # 在段落分隔处插入，不破坏句子结构
            split_response = re.split(r'[\n。]', response)
            if len(split_response) >= 3:
                insert_pos = random.randint(1, len(split_response)-2)
                split_response.insert(insert_pos, f"\n{random.choice(phrases)}\n")
                response = '。'.join(split_response)

        # 7. 疑问句增强（匹配训练数据中"互动感"句式）
        sentences = re.split(r'[。！？]', response)
        new_sentences = []
        for s in sentences:
            if s and len(s) > 5 and random.random() < 0.3:  # 30%概率，长句优先
                s = s.rstrip('。！？') + "，对不对呀～"
            new_sentences.append(s)
        response = '。'.join(new_sentences)

        # 8. 句尾后缀确保（强化角色辨识度）
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
