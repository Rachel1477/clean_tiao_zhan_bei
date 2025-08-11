import torch
import torch.nn as nn
import math

"""
雷达数据（RD 图是图像模态、辅助参数是表格模态 ），
按时间周期分段后，用 Transformer 融合多模态特征，
LSTM 捕捉时间依赖，最终输出分类结果。核心是 “多模态融合 + 时序建模” 双流程。


原本的transformer的一个样本是一条航迹中的一个时间步，即一个点，
此“多模态融合 + 时序建模” 改了一下时序多模态雷达数据集类，按航迹组织数据，一个样本是包含了多个时间步的每条航迹
由于每个航迹的点数不同，分布在15-30之间，因此在这里加入了支持可变长序列

主办方说测试集只能看当前点与之前的，
那训练模型的时候按航迹组织数据算作弊吗？

我也试过一个样本是一条航迹中的一个时间步（即一个点）的雷达数据集类，但是效果不好，准确率在89.88
最原先的transformer是89.19
现在上传的这个是按航迹组织数据，每条航迹包含多个时间步，准确率在92.88
鸟类由最远先的0.82点多到了0.8992，类别2、1也略有上涨，只有类别4掉了0.002

"""
class PatchEmbedding(nn.Module):
    """将图像分割成Patch并进行线性投射"""

    def __init__(self, img_size=224, patch_size=16, in_channels=1, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # 卷积层可以巧妙地实现Patch切分和线性投射
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        # x: (B, C, H, W) -> (B, D, n_patches_h, n_patches_w)
        x = self.proj(x)
        # -> (B, D, N) N = n_patches
        x = x.flatten(2)
        # -> (B, N, D)
        x = x.transpose(1, 2)
        return x


class MultimodalTransformerWithLSTM(nn.Module):
    """
    融合RD图和表格数据的多模态Transformer模型，并添加LSTM处理时间序列。
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_channels=1,
                 num_tabular_features=15,
                 num_classes=6,
                 embed_dim=128,
                 depth=3,
                 heads=4,
                 mlp_dim=256,
                 lstm_hidden_dim=128,  # LSTM隐藏层维度
                 lstm_num_layers=2,  # LSTM层数
                 dropout=0.1):
        super().__init__()

        # --- 1. 图像分支 ---
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches

        # --- 2. 表格数据分支 ---
        self.tabular_projector = nn.Sequential(
            nn.Linear(num_tabular_features + 1, embed_dim // 2),  # +1 用于时间间隔特征
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )

        # --- 3. 融合与序列构建 ---
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        # --- 4. Transformer编码器 ---
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # --- 5. LSTM层处理时间序列 ---
        self.lstm = nn.LSTM(
            input_size=embed_dim,  # 输入维度 = Transformer 输出维度
            hidden_size=lstm_hidden_dim,  # LSTM 隐藏层维度
            num_layers=lstm_num_layers,  # 层数
            batch_first=True,  # 输入格式 (B, T, Dim)
            dropout=dropout,
            bidirectional=False,  # 单向 LSTM，保留时间顺序
        )

        # --- 6. 分类头 ---
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(lstm_hidden_dim),
            nn.Linear(lstm_hidden_dim, num_classes)
        )

    def forward(self, rd_map_sequence, tabular_features_sequence, lengths=None):
        """
        前向传播。

        参数:
        rd_map_sequence: (B, T, C, H, W) 批次×时间步×通道×高×宽
        tabular_features_sequence: (B, T, F) 批次×时间步×特征数
        lengths: (B,) 每个样本的实际长度(用于处理变长序列)
        """
        B, T, C, H, W = rd_map_sequence.shape  # 解析批次大小和时间步

        # 存储每个时间步的CLS输出
        time_step_outputs = []

        for t in range(T):
            # 1. 处理当前时间步的图像数据
            rd_map = rd_map_sequence[:, t]  # (B, C, H, W)
            img_patches = self.patch_embed(rd_map)  # (B, N_patches, D)

            # 2. 添加CLS token
            cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, D)
            img_x = torch.cat((cls_tokens, img_patches), dim=1)  # (B, N_patches + 1, D)

            # 3. 添加位置编码
            img_x = img_x + self.pos_embedding
            img_x = self.dropout(img_x)

            # 4. 处理当前时间步的表格数据和时间间隔
            tabular_features = tabular_features_sequence[:, t]  # (B, F)
            tab_x = self.tabular_projector(tabular_features).unsqueeze(1)  # (B, 1, D)

            # 5. 融合两个模态
            full_sequence = torch.cat((img_x, tab_x), dim=1)  # (B, N_patches + 2, D)

            # 6. Transformer编码 多模态特征交互
            encoded_sequence = self.transformer_encoder(full_sequence)

            # 7. 提取CLS token输出
            cls_output = encoded_sequence[:, 0]  # (B, D)
            time_step_outputs.append(cls_output)# 保存当前时间步结果

        # 构建时间序列 (B, T, D)把每个时间步的 CLS 输出堆叠
        time_series = torch.stack(time_step_outputs, dim=1)

        # 处理变长序列
        if lengths is not None:
            # 按长度降序排序，以便使用pack_padded_sequence
            # 先将 lengths 转换为 CPU 上的 int64 类型
            lengths = lengths.cpu().to(torch.int64)
            lengths, sorted_idx = lengths.sort(descending=True)
            time_series = time_series[sorted_idx]

            # 打包序列
            packed_sequences = nn.utils.rnn.pack_padded_sequence(
                time_series, lengths, batch_first=True
            )

            # LSTM处理
            packed_output, _ = self.lstm(packed_sequences)

            # 解包
            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True
            )

            # 恢复原始顺序
            _, unsorted_idx = sorted_idx.sort()
            lstm_output = lstm_output[unsorted_idx]

            # 获取每个序列的最后一个有效输出
            final_outputs = []
            for i in range(B):
                final_outputs.append(lstm_output[i, lengths[unsorted_idx[i]] - 1])
            final_output = torch.stack(final_outputs, dim=0)
        else:
            # 如果没有长度信息，直接使用最后一个时间步的输出
            lstm_output, _ = self.lstm(time_series)
            final_output = lstm_output[:, -1, :]

        # 8. 分类
        logits = self.mlp_head(final_output)
        return logits
    def forward_online(self, rd_map_t, tabular_features_t, hidden_state=None):
        """
        处理单个时间步的数据，并更新隐藏状态（用于在线预测）。

        参数:
        rd_map_t: 单个时间步的RD图 (B=1, 1, H, W)
        tabular_features_t: 单个时间步的表格特征 (B=1, F)
        hidden_state: 上一时刻的LSTM隐藏状态 (h_{t-1}, c_{t-1})

        返回:
        logits: 当前时间步的航迹级预测
        hidden_state: 更新后的LSTM隐藏状态 (h_t, c_t)
        """
        # 1. 使用Transformer部分作为特征提取器
        # (这部分代码与forward方法中循环内的逻辑几乎一样)
        B = rd_map_t.shape[0] # 在线预测时B通常为1
        img_patches = self.patch_embed(rd_map_t)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        img_x = torch.cat((cls_tokens, img_patches), dim=1)
        img_x = img_x + self.pos_embedding
        img_x = self.dropout(img_x)
        tab_x = self.tabular_projector(tabular_features_t).unsqueeze(1)
        full_sequence = torch.cat((img_x, tab_x), dim=1)
        encoded_sequence = self.transformer_encoder(full_sequence)
        
        # 提取当前时间步的特征向量
        feature_vector_t = encoded_sequence[:, 0].unsqueeze(1) # (B=1, 1, D)

        # 2. 将特征向量送入LSTM，并传入/传出隐藏状态
        # lstm_output 的形状是 (B, 1, lstm_hidden_dim)
        lstm_output, new_hidden_state = self.lstm(feature_vector_t, hidden_state)
        
        # 3. 使用LSTM的输出进行分类
        # 我们取序列的最后一个（也是唯一一个）时间步的输出
        final_output = lstm_output[:, -1, :] # (B, lstm_hidden_dim)
        
        # 4. 分类
        logits = self.mlp_head(final_output)
        
        return logits, new_hidden_state