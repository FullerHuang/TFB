import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from ..layers.Layer import Transformer_Layer
from ..utils.Other import SparseDispatcher, FourierLayer, series_decomp_multi, MLP
import torch.nn.init as init


class AMS(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        num_experts,
        device,
        num_nodes=1,    # 输入数据的特征维度，默认为1，表示单变量时间序列；如果是多变量时间序列，则设置为变量数量
        d_model=32,
        d_ff=64,
        dynamic=False,
        patch_size=[8, 6, 4, 2],
        noisy_gating=True,
        k=4,
        layer_number=1,
        residual_connection=1,
        batch_norm=False,
    ):
        super(AMS, self).__init__()
        self.num_experts = num_experts  # 专家数量 = patchSize数量
        self.output_size = output_size
        self.input_size = input_size
        self.k = k

        self.start_linear = nn.Linear(in_features=num_nodes, out_features=1)
        self.seasonality_model = FourierLayer(pred_len=0, k=3)
        self.trend_model = series_decomp_multi(kernel_size=[4, 8, 12])

        self.experts = nn.ModuleList()
        self.MLPs = nn.ModuleList()
        for patch in patch_size:
            patch_nums = int(input_size / patch)
            self.experts.append(
                Transformer_Layer(
                    device=device,
                    d_model=d_model,
                    d_ff=d_ff,
                    dynamic=dynamic,
                    num_nodes=num_nodes,
                    patch_nums=patch_nums,
                    patch_size=patch,
                    factorized=True,
                    layer_number=layer_number,
                    batch_norm=batch_norm,
                )
            )

        self.w_gate = torch.empty(input_size, num_experts)      # 可学习的路由权重矩阵
        init.kaiming_uniform_(self.w_gate, mode="fan_in", nonlinearity="relu")  # 使用Kaiming均匀分布策略初始化权重矩阵，适用于ReLU激活函数
        self.w_gate = nn.Parameter(self.w_gate, requires_grad=True) # 将权重矩阵注册为模型参数，使其在训练过程中可更新

        self.w_noise = torch.empty(input_size, num_experts)
        init.kaiming_uniform_(self.w_noise, mode="fan_in", nonlinearity="relu")
        self.w_noise = nn.Parameter(self.w_noise, requires_grad=True)

        # #self.w_gate = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        # self.w_noise = nn.Parameter(torch.zeros(input_size, num_experts), requires_grad=True)
        # init.kaiming_uniform_(self.w_gate, mode='fan_in', nonlinearity='relu')
        # init.kaiming_uniform_(self.w_noise, mode='fan_in', nonlinearity='relu')

        self.residual_connection = residual_connection
        self.end_MLP = nn.Linear(input_size, output_size)

        self.noisy_gating = noisy_gating
        self.softplus = nn.Softplus()   # Softplus 是ReLU的光滑近似，避免了ReLU在0处不可导的问题
        self.softmax = nn.Softmax(1)
        self.register_buffer("mean", torch.tensor([0.0]))
        self.register_buffer("std", torch.tensor([1.0]))
        assert self.k <= self.num_experts

    def cv_squared(self, x):
        '''
        方差系数：它衡量分布的均匀程度——值越小表示分布越均匀。
        '''
        eps = 1e-10
        if x.shape[0] == 1:
            return torch.tensor([0], device=x.device, dtype=x.dtype)
        return x.float().var() / (x.float().mean() ** 2 + eps)

    def _gates_to_load(self, gates):
        return (gates > 0).sum(0)

    def _prob_in_top_k(
        self, clean_values, noisy_values, noise_stddev, noisy_top_values
    ):
        batch = clean_values.size(0)
        m = noisy_top_values.size(1)
        top_values_flat = noisy_top_values.flatten()

        threshold_positions_if_in = (
            torch.arange(batch, device=clean_values.device) * m + self.k
        )
        threshold_if_in = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_in), 1
        )
        is_in = torch.gt(noisy_values, threshold_if_in)
        threshold_positions_if_out = threshold_positions_if_in - 1
        threshold_if_out = torch.unsqueeze(
            torch.gather(top_values_flat, 0, threshold_positions_if_out), 1
        )
        normal = Normal(self.mean, self.std)
        prob_if_in = normal.cdf((clean_values - threshold_if_in) / noise_stddev)
        prob_if_out = normal.cdf((clean_values - threshold_if_out) / noise_stddev)
        prob = torch.where(is_in, prob_if_in, prob_if_out)
        return prob

    def seasonality_and_trend_decompose(self, x):
        x = x[:, :, :, 0]
        _, trend = self.trend_model(x)
        seasonality, _ = self.seasonality_model(x)
        return x + seasonality + trend      # 季节成分（FFT) + 趋势成分（平均卷积） + 原始输入

    def noisy_top_k_gating(self, x, train, noise_epsilon=1e-2):
        '''
        获取输入x对应的每个专家的门控权重，最终返回稀疏门控向量和每个专家的负载。
        '''
        x = self.start_linear(x).squeeze(-1)    # 压缩特征维度，(batch, seq_len, num_nodes) -> (batch, seq_len)

        clean_logits = x @ self.w_gate  # @为矩阵乘法运算符，(batch, seq_len) @ (seq_len, num_experts) -> (batch, num_experts)
                                        # 得到每个输入对应每个专家(patchSize)的原始得分(无噪声)，形状为(batch, num_experts)
                                        # w_gate 是可学习的路由权重矩阵，模型通过训练不断调整这个矩阵，使得不同输入能够更好地路由到合适的专家(patchSize)，从而提升模型性能

        if self.noisy_gating and train: # 在训练阶段，如果启用噪声门控，则为logits添加噪声
            raw_noise_stddev = x @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + noise_epsilon
            noisy_logits = clean_logits + (
                torch.randn_like(clean_logits) * noise_stddev
            )
            logits = noisy_logits
        else:
            logits = clean_logits   # (batch, num_experts)
        # calculate topk + 1 that will be needed for the noisy gates
        top_logits, top_indices = logits.topk(min(self.k + 1, self.num_experts), dim=1)

        top_k_logits = top_logits[:, : self.k]      # 取前k个专家得分，形状为(batch, k)
        top_k_indices = top_indices[:, : self.k]
        top_k_gates = self.softmax(top_k_logits)

        zeros = torch.zeros_like(logits, requires_grad=True)
        gates = zeros.scatter(1, top_k_indices, top_k_gates)    # 将值(softmax后)填充到 (batch, num_experts) 的对应位置，其余为 0。
                                                                # 此即稀疏门控向量，对应论文中经过 Top‑K 选择后的路由权重

        if self.noisy_gating and self.k < self.num_experts and train:
            load = (
                self._prob_in_top_k(
                    clean_logits, noisy_logits, noise_stddev, top_logits
                )
            ).sum(0)
        else:                                   # gates: 每个样本在每个专家上的最终权重，用于加权聚合专家(开启的 PatchSize 对应的 Multi-Scale Transformer Block)输出
            load = self._gates_to_load(gates)   # load: 每个专家被选择的概率/次数，即每个专家的负载，形状为(num_experts,)
        return gates, load

    def forward(self, x, loss_coef=1e-2):
        new_x = self.seasonality_and_trend_decompose(x) # 季节成分（FFT) + 趋势成分（平均卷积） + 原始输入

        # multi-scale router
        gates, load = self.noisy_top_k_gating(new_x, self.training)
        # calculate balance loss
        importance = gates.sum(0)   # 每个专家的总权重（跨 batch），即每个专家的重要性，形状为(num_experts,)
        balance_loss = self.cv_squared(importance) + self.cv_squared(load)  # 计算负载平衡损失，鼓励模型在不同专家之间均匀分配输入，避免过度依赖某些专家，从而提升模型的泛化能力和鲁棒性
        balance_loss *= loss_coef
        dispatcher = SparseDispatcher(self.num_experts, gates)
        expert_inputs = dispatcher.dispatch(x)
        expert_outputs = [
            self.experts[i](expert_inputs[i])[0] for i in range(self.num_experts)
        ]
        output = dispatcher.combine(expert_outputs)
        if self.residual_connection:
            output = output + self.end_MLP(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        return output, balance_loss
