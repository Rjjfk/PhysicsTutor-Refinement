import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class PhysicsDiscriminator(nn.Module):
    def __init__(self, base_model_path="bert-base-chinese"):
        super().__init__()
        self.bert = AutoModel.from_pretrained(base_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_path)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)  # 二分类：真实=1，生成=0
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_feat = outputs.last_hidden_state[:, 0, :]  # [CLS] token特征
        logits = self.classifier(cls_feat)
        return self.sigmoid(logits)  # 输出0-1概率

    def train_discriminator(self, real_data, fake_data, epochs=3, lr=2e-5):
        """训练判别器区分真实数据和生成数据"""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.BCELoss()

        # 构建训练数据（真实=1，生成=0）
        real_texts = [f"物理解析：{item['output']}" for item in real_data]
        fake_texts = [f"物理解析：{item['output']}" for item in fake_data]
        all_texts = real_texts + fake_texts
        labels = torch.cat([torch.ones(len(real_texts)), torch.zeros(len(fake_texts))]).unsqueeze(1)

        # 分词
        inputs = self.tokenizer(all_texts, padding=True, truncation=True, return_tensors="pt")
        input_ids = inputs["input_ids"].to(next(self.parameters()).device)
        attention_mask = inputs["attention_mask"].to(next(self.parameters()).device)
        labels = labels.to(next(self.parameters()).device)

        # 训练循环
        self.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            print(f"判别器Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")

        return self