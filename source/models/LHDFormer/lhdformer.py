import torch
import torch.nn as nn
from torch.nn import TransformerEncoderLayer
from .ptdec import DEC
from typing import List
from .components import InterpretableTransformerEncoder
from omegaconf import DictConfig
from ..base import BaseModel
import pickle
from ...utils.device import device_set

device = device_set()

import torch
import torch.nn.functional as F


def neuro_walk_embedding(data, walk_length=8, beta=0.1):
    """
    NeuroWalk-based biased random walk embedding.
    """
    device = data.device
    num_nodes = data.shape[0]

    deg = data.sum(dim=1)
    deg_inv = 1.0 / deg
    deg_inv[deg_inv == float('inf')] = 0

    P = data * deg_inv.view(-1, 1)

    fro_norm = torch.norm(data, p='fro')
    scale = torch.sigmoid(fro_norm / torch.sqrt(torch.tensor(num_nodes, dtype=torch.float32, device=device)))

    pe_list = [torch.eye(num_nodes, device=device)]
    current = torch.eye(num_nodes, device=device)

    for k in range(1, walk_length):
        exp_term = torch.tensor(beta * (1 - k), dtype=torch.float32, device=device)
        factor = torch.exp(exp_term) * scale

        R_k = (factor * data) * deg_inv.view(-1, 1)

        current = current @ R_k
        pe_list.append(current)

    pe = torch.stack(pe_list, dim=-1)
    abs_pe = pe.diagonal().transpose(0, 1)
    return abs_pe



def add_full_rrwp(data, walk_length, thresholded=False, thresh=0.3, beta=0.1, add_identity=True):
    """
    Batch wrapper. `data` expected shape: [batch, N, N] (same as original usage in your forward).
    Returns: tensor of shape [batch, N, walk_length]
    """
    pes = []
    batch = data.shape[0]
    for idx in range(batch):
        dt = data[idx].squeeze()
        dt = dt.to(torch.float32)
        pe = add_every_rrwp(dt, walk_length=walk_length, thresholded=thresholded, thresh=thresh,
                             beta=beta, add_identity=add_identity)
        pes.append(pe)
    return torch.stack(pes, dim=0)


def add_every_rrwp(data,
                   walk_length=8,
                   add_identity=True,
                   spd=False,
                   thresholded=False,
                   thresh=0.3,
                   beta=0.1,
                   **kwargs):

    device = data.device
    data = data.to(torch.float32).to(device)

    if thresholded:
        mask = (data > thresh).to(data.dtype)
        data = data * mask

    data = torch.abs(data)
    data = data.fill_diagonal_(0)

    pe = neuro_walk_embedding(data, walk_length=walk_length, beta=beta)

    if not add_identity:
        if pe.shape[1] > 0:
            pe = pe[:, 1:]
        else:
            pass

    return pe


class TransPoolingEncoder(nn.Module):

    def __init__(self, input_feature_size, input_node_num, hidden_size, output_node_num, pooling=True, orthogonal=True,
                 freeze_center=False, project_assignment=True, nHead=4, local_transformer=False):
        super().__init__()
        self.transformer = InterpretableTransformerEncoder(d_model=input_feature_size, nhead=nHead,
                                                           dim_feedforward=hidden_size,
                                                           batch_first=True)

        self.local_transformer = local_transformer
        if local_transformer:
            self.pooling = False
        else:
            self.pooling = pooling
        if self.pooling:
            encoder_hidden_size = 32
            self.encoder = nn.Sequential(
                nn.Linear(input_feature_size *
                          input_node_num, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size, encoder_hidden_size),
                nn.LeakyReLU(),
                nn.Linear(encoder_hidden_size,
                          input_feature_size * input_node_num),
            )
            self.dec = DEC(cluster_number=output_node_num, hidden_dimension=input_feature_size, encoder=self.encoder,
                           orthogonal=orthogonal, freeze_center=freeze_center, project_assignment=project_assignment)

        if local_transformer:
            self.class_token = nn.ParameterList()
            self.class_token.append(nn.Parameter(torch.Tensor(1, input_feature_size), requires_grad=True).to(device))
            self.class_token.append(nn.Parameter(torch.Tensor(1, input_feature_size), requires_grad=True).to(device))
            self.class_token.append(nn.Parameter(torch.Tensor(1, input_feature_size), requires_grad=True).to(device))
            self.class_token.append(nn.Parameter(torch.Tensor(1, input_feature_size), requires_grad=True).to(device))
            self.class_token.append(nn.Parameter(torch.Tensor(1, input_feature_size), requires_grad=True).to(device))

        self.reset_parameters(local_transformer)

    def reset_parameters(self, local_transformer=False):
        if local_transformer:
            for i in range(len(self.class_token)):
                self.class_token[i] = nn.init.xavier_normal_(self.class_token[i])

    def is_pooling_enabled(self):
        return self.pooling

    def forward(self,
                x: torch.tensor, cluster_num=-1):
        bz, node_num, dim = x.shape
        if self.local_transformer:
            class_token = self.class_token[cluster_num]
            class_token = class_token.repeat(bz, 1, 1)
            x = torch.cat((class_token, x), dim=1)
        x = self.transformer(x)
        if self.local_transformer:
            cls_token = x[:, 0, :]
            x = x[:, 1:, :]
            return x, None, cls_token.reshape(x.shape[0], 1, -1)
        else:
            cls_token = x[:, 0, :]
            x = x[:, 1:, :]
            if self.pooling:
                x, assignment = self.dec(x)
                return x, assignment, cls_token.reshape(x.shape[0], 1, -1)
            else:
                return x, None, cls_token.reshape(x.shape[0], 1, -1)

    def get_attention_weights(self):
        return self.transformer.get_attention_weights()

    def loss(self, assignment):
        return self.dec.loss(assignment)


class LHDFormer(BaseModel):

    def __init__(self, config: DictConfig):

        super().__init__()

        self.attention_list = nn.ModuleList()
        forward_dim = config.dataset.node_sz

        self.pos_encoding = config.model.pos_encoding
        self.pos_embed_dim = config.model.pos_embed_dim

        if self.pos_encoding == 'identity':
            self.node_identity = nn.Parameter(torch.zeros(
                config.dataset.node_sz, config.model.pos_embed_dim), requires_grad=True)
            forward_dim = config.dataset.node_sz + config.model.pos_embed_dim
            nn.init.kaiming_normal_(self.node_identity)
        if self.pos_encoding == 'rrwp':
            forward_dim = config.dataset.node_sz + config.model.pos_embed_dim

        self.num_MHSA = config.model.num_MHSA
        sizes = config.model.sizes
        sizes[0] = config.dataset.node_sz
        in_sizes = [config.dataset.node_sz] + sizes[:-1]
        do_pooling = config.model.pooling
        self.do_pooling = do_pooling

        self.local_transformer = TransPoolingEncoder(input_feature_size=forward_dim,
                                                     input_node_num=200,
                                                     hidden_size=1024,
                                                     output_node_num=sizes[1],
                                                     pooling=False,
                                                     orthogonal=config.model.orthogonal,
                                                     freeze_center=config.model.freeze_center,
                                                     project_assignment=config.model.project_assignment,
                                                     nHead=config.model.nhead,
                                                     local_transformer=True)

        if config.model.num_MHSA == 1:
            self.attention_list.append(
                TransPoolingEncoder(input_feature_size=forward_dim,
                                    input_node_num=200,
                                    hidden_size=1024,
                                    output_node_num=sizes[1],
                                    pooling=do_pooling[1],
                                    orthogonal=config.model.orthogonal,
                                    freeze_center=config.model.freeze_center,
                                    project_assignment=config.model.project_assignment,
                                    nHead=config.model.nhead,
                                    local_transformer=False))
        else:
            for index, size in enumerate(sizes):
                self.attention_list.append(
                    TransPoolingEncoder(input_feature_size=forward_dim,
                                        input_node_num=in_sizes[index],
                                        hidden_size=1024,
                                        output_node_num=size,
                                        pooling=do_pooling[index],
                                        orthogonal=config.model.orthogonal,
                                        freeze_center=config.model.freeze_center,
                                        project_assignment=config.model.project_assignment,
                                        nHead=config.model.nhead,
                                        local_transformer=False))

        self.dim_reduction = nn.Sequential(
            nn.Linear(forward_dim, 8),
            nn.LeakyReLU()
        )

        self.fc = nn.Sequential(
            nn.Linear(8 * sizes[-1], 256),
            nn.LeakyReLU(),
            nn.Linear(256, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 2)
        )

        self.assignMat = None
        self.mlp = nn.Sequential(
            nn.Linear(5 * forward_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, forward_dim),
            nn.LeakyReLU()
        )

        self.mlp_feature = nn.Sequential(
            nn.Linear(1000 * forward_dim, 512),
            nn.LeakyReLU(),
            nn.Linear(512, forward_dim*200),
            nn.LeakyReLU()
        )

        self.sliding_size = [200, 400, 600, 800, 1000]
        self.sliding = [20, 40, 60, 80, 100]

    def sliding_windows_pearson(self, tensor):
        batch_size, num_nodes, time_series_length = tensor.shape
        correlation_matrices = []
        for i in range(batch_size):
            sample = tensor[i]
            corr_matrix = torch.corrcoef(sample)
            nan_mask = torch.isnan(corr_matrix)
            if nan_mask.any():
                corr_matrix[nan_mask] = 0.000001
            corr_matrix = torch.abs(corr_matrix)
            corr_matrix = corr_matrix.fill_diagonal_(0)
            corr_matrix = torch.round(corr_matrix * 1000) / 1000
            correlation_matrices.append(corr_matrix)
        return torch.stack(correlation_matrices)

    def forward(self,
                time_seires: torch.tensor,
                node_feature: torch.tensor):

        bz, _, _, = node_feature.shape
        sliding_feature = torch.zeros(bz, 1000, 200).to(device)
        sliding_feature[:, :self.sliding_size[0], :] = self.sliding_windows_pearson(time_seires[:, :, :self.sliding[0]])
        sliding_feature[:, self.sliding_size[0]: self.sliding_size[1], :] = self.sliding_windows_pearson(time_seires[:, :, self.sliding[0]: self.sliding[1]])
        sliding_feature[:, self.sliding_size[1]: self.sliding_size[2], :] = self.sliding_windows_pearson(time_seires[:, :, self.sliding[1]: self.sliding[2]])
        sliding_feature[:, self.sliding_size[2]: self.sliding_size[3], :] = self.sliding_windows_pearson(time_seires[:, :, self.sliding[2]: self.sliding[3]])
        sliding_feature[:, self.sliding_size[3]: self.sliding_size[4], :] = self.sliding_windows_pearson(time_seires[:, :, self.sliding[3]: self.sliding[4]])

        input_feature = torch.zeros(bz, 1000, 232).to(device)
        if self.pos_encoding == 'identity':
            pos_emb = self.node_identity.expand(bz, *self.node_identity.shape)
            node_feature = torch.cat([node_feature, pos_emb], dim=-1)

        if self.pos_encoding == 'rrwp':
            pos_emb = add_full_rrwp(node_feature, self.pos_embed_dim)
            node_feature = torch.cat([node_feature, pos_emb], dim=-1)

            pos_emb0 = add_full_rrwp(sliding_feature[:, :self.sliding_size[0], :], self.pos_embed_dim)
            pos_emb1 = add_full_rrwp(sliding_feature[:, self.sliding_size[0]: self.sliding_size[1], :], self.pos_embed_dim)
            pos_emb2 = add_full_rrwp(sliding_feature[:, self.sliding_size[1]: self.sliding_size[2], :], self.pos_embed_dim)
            pos_emb3 = add_full_rrwp(sliding_feature[:, self.sliding_size[2]: self.sliding_size[3], :], self.pos_embed_dim)
            pos_emb4 = add_full_rrwp(sliding_feature[:, self.sliding_size[3]: self.sliding_size[4], :], self.pos_embed_dim)

            input_feature[:, : self.sliding_size[0], :] = torch.cat([sliding_feature[:, :self.sliding_size[0], :], pos_emb0], dim=-1)
            input_feature[:, self.sliding_size[0]: self.sliding_size[1], :] = torch.cat([sliding_feature[:, self.sliding_size[0]: self.sliding_size[1], :], pos_emb1], dim=-1)
            input_feature[:, self.sliding_size[1]: self.sliding_size[2], :] = torch.cat([sliding_feature[:, self.sliding_size[1]: self.sliding_size[2], :], pos_emb2], dim=-1)
            input_feature[:, self.sliding_size[2]: self.sliding_size[3], :] = torch.cat([sliding_feature[:, self.sliding_size[2]: self.sliding_size[3], :], pos_emb3], dim=-1)
            input_feature[:, self.sliding_size[3]: self.sliding_size[4], :] = torch.cat([sliding_feature[:, self.sliding_size[3]: self.sliding_size[4], :], pos_emb4], dim=-1)

        assignments = []
        attn_weights = []

        input_feature[:, :self.sliding_size[0], :], _, local_class_tokens0 = self.local_transformer(
            input_feature[:, :self.sliding_size[0], :], cluster_num=0)
        input_feature[:, self.sliding_size[0]:self.sliding_size[1], :], _, local_class_tokens1 = self.local_transformer(
            input_feature[:, self.sliding_size[0]:self.sliding_size[1], :], cluster_num=1)
        input_feature[:, self.sliding_size[1]:self.sliding_size[2], :], _, local_class_tokens2 = self.local_transformer(
            input_feature[:, self.sliding_size[1]:self.sliding_size[2], :], cluster_num=2)
        input_feature[:, self.sliding_size[2]:self.sliding_size[3], :], _, local_class_tokens3 = self.local_transformer(
            input_feature[:, self.sliding_size[2]:self.sliding_size[3], :], cluster_num=3)
        input_feature[:, self.sliding_size[3]:self.sliding_size[4], :], _, local_class_tokens4 = self.local_transformer(
            input_feature[:, self.sliding_size[3]:self.sliding_size[4], :], cluster_num=4)

        # node_feature = node_feature_rearranged
        # input_feature = input_feature.reshape((bz, -1))
        # input_feature = self.mlp_feature(input_feature)
        # input_feature = input_feature.reshape((bz, 200, -1))

        class_token = torch.cat((local_class_tokens0, local_class_tokens1, local_class_tokens2, local_class_tokens3,
                                 local_class_tokens4), dim=1)
        class_token = class_token.reshape((bz, -1))
        class_token = self.mlp(class_token)
        class_token = class_token.reshape((bz, 1, -1))
        node_feature = torch.cat((class_token, node_feature), dim=1)

        if self.num_MHSA == 1:
            node_feature, assign_mat, cls_token = self.attention_list[0](node_feature)
            assignments.append(assign_mat)
            attn_weights.append(self.attention_list[0].get_attention_weights())
        else:
            for atten in self.attention_list:
                node_feature, _, cls_token = atten(node_feature)
                attn_weights.append(atten.get_attention_weights())

        self.assignMat = assignments[0]

        node_feature = self.dim_reduction(node_feature)

        node_feature = node_feature.reshape((bz, -1))

        return self.fc(node_feature), None

    def get_assign_mat(self):
        return self.assignMat

    def get_attention_weights(self):
        return [atten.get_attention_weights() for atten in self.attention_list]

    def get_local_attention_weights(self):
        return self.local_transformer.get_attention_weights()

    def get_cluster_centers(self) -> torch.Tensor:
        """
        Get the cluster centers, as computed by the encoder.

        :return: [number of clusters, hidden dimension] Tensor of dtype float
        """
        return self.dec.get_cluster_centers()

    def loss(self, assignments):
        """
        Compute KL loss for the given assignments. Note that not all encoders contain a pooling layer.
        Inputs: assignments: [batch size, number of clusters]
        Output: KL loss
        """
        decs = list(
            filter(lambda x: x.is_pooling_enabled(), self.attention_list))
        assignments = list(filter(lambda x: x is not None, assignments))
        loss_all = None

        for index, assignment in enumerate(assignments):
            if loss_all is None:
                loss_all = decs[index].loss(assignment)
            else:
                loss_all += decs[index].loss(assignment)
        return loss_all



