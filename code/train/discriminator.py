# 修改后：discriminator.py（双分支判别器，通过type参数切换任务）
import torch
import torch.nn as nn
from transformers import AutoTokenizer


class DualDiscriminator(nn.Module):
    def __init__(self, tokenizer_path, task_type="physics", hidden_size=768):
        """
        task_type: "physics"（物理逻辑判别） / "role"（角色风格判别）
        """
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        self.task_type = task_type
        self.embedding = nn.Embedding(self.tokenizer.vocab_size, hidden_size)

        # 1. 共享特征提取层（LSTM/CNN，复用参数减少冗余）
        if self.task_type == "physics":
            # 物理任务：聚焦公式/步骤逻辑，用CNN捕捉局部公式特征
            self.feature_extractor = nn.Conv1d(hidden_size, 256, kernel_size=3, padding=1)
        else:
            # 角色任务：聚焦全局风格特征，用双向LSTM捕捉句式/关键词
            self.feature_extractor = nn.LSTM(hidden_size, 256, bidirectional=True, batch_first=True)

        # 2. 任务专属输出层（二分类：正确/错误）
        self.pool = nn.AdaptiveMaxPool1d(1) if task_type == "physics" else nn.AdaptiveAvgPool1d(1)
        output_dim = 256 if task_type == "physics" else 512  # LSTM双向输出=256*2
        self.fc = nn.Linear(output_dim, 1)
        self.sigmoid = nn.Sigmoid()

        # 角色任务专属：爱莉希雅风格特征库（从config读取，避免硬编码）
        self.style_features = []
        if task_type == "role":
            with open("../../config/elysia_config.json", "r") as f:
                self.style_features = json.load(f)["role_adversarial"]["style_features"]

    def forward(self, input_ids):
        """
        input_ids: [batch_size, seq_len]（模型生成的解析文本）
        返回：[batch_size, 1]（0=错误/偏离，1=正确/符合）
        """
        # 文本嵌入
        emb = self.embedding(input_ids)  # [batch_size, seq_len, hidden_size]

        # 任务专属特征提取
        if self.task_type == "physics":
            emb = emb.permute(0, 2, 1)  # [batch_size, hidden_size, seq_len]
            feat = self.feature_extractor(emb)  # [batch_size, 256, seq_len]
            feat = self.pool(feat).squeeze(-1)  # [batch_size, 256]
        else:
            feat, _ = self.feature_extractor(emb)  # [batch_size, seq_len, 512]
            feat = feat.permute(0, 2, 1)  # [batch_size, 512, seq_len]
            feat = self.pool(feat).squeeze(-1)  # [batch_size, 512]

        # 二分类输出
        logits = self.fc(feat)
        return self.sigmoid(logits)

    # --------------------------
    # 任务专属辅助函数（增强判别准确性）
    # --------------------------
    def check_physics_logic(self, text):
        """物理任务：检查文本是否包含正确公式（辅助判别）"""
        key_formulas = ["动量守恒", "v0²=2*g*l", "s=v0*t+(1/2)*a*t²"]
        return sum(1 for f in key_formulas if f in text) / len(key_formulas)

    def check_role_style(self, text):
        """角色任务：检查文本是否包含爱莉希雅风格特征（辅助判别）"""
        return sum(1 for f in self.style_features if f in text) / len(self.style_features)


# 使用示例（对抗训练中调用）
if __name__ == "__main__":
    # 物理判别器
    physics_disc = DualDiscriminator(
        tokenizer_path="../../model/base",
        task_type="physics"
    )
    # 角色判别器
    role_disc = DualDiscriminator(
        tokenizer_path="../../model/base",
        task_type="role"
    )