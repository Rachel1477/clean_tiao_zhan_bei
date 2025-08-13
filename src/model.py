import torch
import torch.nn as nn
import math

class PatchEmbedding(nn.Module):
    """将图像分割成Patch并进行线性投射"""
    def __init__(self, img_size=224, patch_size=16, in_channels=1, embed_dim=128):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2)
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
                 lstm_hidden_dim=128, 
                 lstm_num_layers=2, 
                 dropout=0.1):
        super().__init__()


        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        num_patches = self.patch_embed.n_patches


        self.tabular_projector = nn.Sequential(
            nn.Linear(num_tabular_features + 1, embed_dim // 2),  # +1 用于时间间隔特征
            nn.ReLU(),
            nn.Linear(embed_dim // 2, embed_dim),
        )


        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)


        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=mlp_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)


        self.lstm = nn.LSTM(
            input_size=embed_dim,  
            hidden_size=lstm_hidden_dim,  
            num_layers=lstm_num_layers,  
            batch_first=True,  
            dropout=dropout,
            bidirectional=False,  
        )

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(lstm_hidden_dim),
            nn.Linear(lstm_hidden_dim, num_classes)
        )

    def forward(self, rd_map_sequence, tabular_features_sequence, lengths=None):
        """
        rd_map_sequence: (B, T, C, H, W) 批次×时间步×通道×高×宽
        tabular_features_sequence: (B, T, F) 批次×时间步×特征数
        lengths: (B,) 每表示个样本的实际长度(用于处理变长序列)
        """
        B, T, C, H, W = rd_map_sequence.shape 


        time_step_outputs = []

        for t in range(T):
            rd_map = rd_map_sequence[:, t]  
            img_patches = self.patch_embed(rd_map) 

            cls_tokens = self.cls_token.expand(B, -1, -1)  
            img_x = torch.cat((cls_tokens, img_patches), dim=1) 

            img_x = img_x + self.pos_embedding
            img_x = self.dropout(img_x)
            tabular_features = tabular_features_sequence[:, t]  
            tab_x = self.tabular_projector(tabular_features).unsqueeze(1)  

            full_sequence = torch.cat((img_x, tab_x), dim=1)  

            encoded_sequence = self.transformer_encoder(full_sequence)


            cls_output = encoded_sequence[:, 0] 
            time_step_outputs.append(cls_output)

        time_series = torch.stack(time_step_outputs, dim=1)

        # 处理变长序列
        if lengths is not None:
            lengths = lengths.cpu().to(torch.int64)
            lengths, sorted_idx = lengths.sort(descending=True)
            time_series = time_series[sorted_idx]

            packed_sequences = nn.utils.rnn.pack_padded_sequence(
                time_series, lengths, batch_first=True
            )


            packed_output, _ = self.lstm(packed_sequences)


            lstm_output, _ = nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True
            )


            _, unsorted_idx = sorted_idx.sort()
            lstm_output = lstm_output[unsorted_idx]


            final_outputs = []
            for i in range(B):
                final_outputs.append(lstm_output[i, lengths[unsorted_idx[i]] - 1])
            final_output = torch.stack(final_outputs, dim=0)
        else:
            lstm_output, _ = self.lstm(time_series)
            final_output = lstm_output[:, -1, :]

        logits = self.mlp_head(final_output)
        return logits
    def forward_online(self, rd_map_t, tabular_features_t, hidden_state=None):
        """
        处理单个时间步的数据，并更新隐藏状态，用于避免误解为作弊。
        """
        B = rd_map_t.shape[0] 
        img_patches = self.patch_embed(rd_map_t)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        img_x = torch.cat((cls_tokens, img_patches), dim=1)
        img_x = img_x + self.pos_embedding
        img_x = self.dropout(img_x)
        tab_x = self.tabular_projector(tabular_features_t).unsqueeze(1)
        full_sequence = torch.cat((img_x, tab_x), dim=1)
        encoded_sequence = self.transformer_encoder(full_sequence)
        
        feature_vector_t = encoded_sequence[:, 0].unsqueeze(1)

        lstm_output, new_hidden_state = self.lstm(feature_vector_t, hidden_state)

        final_output = lstm_output[:, -1, :] 
        
        logits = self.mlp_head(final_output)
        
        return logits, new_hidden_state