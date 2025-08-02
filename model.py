

import torch
import torch.nn as nn
from torch_geometric.nn import  global_mean_pool as gmeanp, global_max_pool as gmaxp
import torch.nn.functional as F
from GCN import DrugGraph, ProteinGraph
import os



class MMF(nn.Module):
    def __init__(self, hp,):
        super(MMF, self).__init__()

        self.attention_dim = hp.out_dim
        self.mix_attention_head = 5
        self.compound_structure_dim = hp.compound_structure_dim
        self.sequence_structure_dim = hp.sequence_structure_dim
        self.out_dim = hp.out_dim
        self.compound_llm_dim = hp.compound_llm_dim
        self.sequence_llm_dim = hp.sequence_llm_dim


        self.gcn_drug = DrugGraph(
            in_channels=self.compound_structure_dim,
            hidden_channels=self.compound_structure_dim,
            out_channels=self.compound_structure_dim,
            num_layers=3,  # 这里可以灵活调参
            num_edge_types=4,  # 单键/双键/三键/芳香键
            dropout_p=0.1
        )
        self.gcn_protein = ProteinGraph(
            in_channels=self.sequence_structure_dim,
            hidden_channels=self.sequence_structure_dim,
            out_channels=self.sequence_structure_dim,
            num_layers=3,
            dropout_p=0.1
        )

        self.Drugllm_max_pool = nn.MaxPool1d(100)
        self.Proteinllm_max_pool = nn.MaxPool1d(1000)
        self.fusion = Multi_Fusion()

        self.drugllm_linear = nn.Linear(self.compound_llm_dim, self.out_dim)
        self.proteinllm_linear = nn.Linear(self.sequence_llm_dim, self.out_dim)
        self.drugstr_linear = nn.Linear(self.compound_structure_dim, self.out_dim)
        self.proteinstr_linear = nn.Linear(self.sequence_structure_dim, self.out_dim)
        self.drug_norm = nn.LayerNorm(self.out_dim)
        self.protein_norm = nn.LayerNorm(self.out_dim)

        self.mix_attention_layer = nn.MultiheadAttention(
            self.attention_dim, self.mix_attention_head, batch_first=True)
        self.drop_drugllm = nn.Dropout(0.1)
        self.drop_prollm = nn.Dropout(0.1)

        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        self.dropout3 = nn.Dropout(0.1)
        self.leaky_relu = nn.LeakyReLU()
        self.fc1 = nn.Linear(self.out_dim * 4, 1024)

        self.fc2 = nn.Linear(1024, 1024)
        self.fc3 = nn.Linear(1024, 512)
        self.out = nn.Linear(512, 2)



    def forward(self, drug_graph, protein_graph, drug_llm, protein_llm):

        drug_x, drug_edge_index, drug_edge_weight, drug_edge_type, drug_batch = drug_graph.x, drug_graph.edge_index, drug_graph.edge_attr, drug_graph.edge_type, drug_graph.batch

        protein_x, protein_edge_index, protein_batch = protein_graph.x, protein_graph.edge_index, protein_graph.batch

        drug_gcn = self.gcn_drug(drug_x, drug_edge_index, drug_edge_weight, drug_edge_type)
        protein_gcn = self.gcn_protein(protein_x, protein_edge_index)
        drug_struc = gmaxp(drug_gcn, drug_batch)+gmeanp(drug_gcn, drug_batch)
        protein_struc = gmaxp(protein_gcn, protein_batch)+gmeanp(protein_gcn, protein_batch)

        drug_struc = self.drug_norm(self.drugstr_linear(drug_struc))
        protein_struc = self.protein_norm(self.proteinstr_linear(protein_struc))


        drug_llm = self.drugllm_linear(drug_llm)
        protein_llm = self.proteinllm_linear(protein_llm)


        drug_llmatt, protein_llmatt, drug2protein_attn, protein2drug_attn= self.cross_attention(drug_llm, protein_llm)
        drug_llm = self.Drugllm_max_pool(drug_llmatt.permute(0, 2, 1)).squeeze(2)
        protein_llm = self.Proteinllm_max_pool(protein_llmatt.permute(0, 2, 1)).squeeze(2)

        # '''特征融合'''
        drug_fusion = self.fusion(drug_llm, drug_struc)
        protein_fusion = self.fusion(protein_llm, protein_struc)
        pair = torch.cat([drug_fusion, protein_fusion], dim=1)
        pair = self.dropout1(pair)
        fully1 = self.leaky_relu(self.fc1(pair))
        fully1 = self.dropout2(fully1)
        fully2 = self.leaky_relu(self.fc2(fully1))
        fully2 = self.dropout3(fully2)
        fully3 = self.leaky_relu(self.fc3(fully2))
        predict = self.out(fully3)

        return predict


    def cross_attention(self, drug, protein):

        drug_att, drug2protein_attn = self.mix_attention_layer(drug, protein, protein)
        protein_att, protein2drug_attn = self.mix_attention_layer(protein, drug, drug)
        drug = drug * 0.5 + drug_att * 0.5
        protein = protein * 0.5 + protein_att * 0.5

        return drug, protein, drug2protein_attn, protein2drug_attn


class Multi_Fusion(nn.Module):
    def __init__(self):
        super(Multi_Fusion, self).__init__()
        self.so_f = nn.Sigmoid()


    def forward(self, LM_fea, Sty_fea):
        Sty_fea_norm = Sty_fea * (torch.mean(LM_fea.abs()) / torch.mean(Sty_fea.abs()))
        f_att = torch.mean(torch.stack([LM_fea, Sty_fea_norm]), dim=0)
        f_att = self.so_f(f_att)


        # 拼接
        fus_fea = torch.cat([
            LM_fea,
            Sty_fea_norm * f_att
        ], dim=1)

        return fus_fea



