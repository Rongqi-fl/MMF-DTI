import torch
from torch.utils.data import Dataset
from utils.protein_structure import *
import numpy as np
from torch_geometric.data import Batch, Data



# 加载数据
class FeatureCache:
    def __init__(self, drug_str_path, protein_str_path, drug_llm_path, protein_llm_path):
        # 预加载小文件，直接存入内存
        self.drug_str = torch.load(drug_str_path)  # 假设这些特征比较小，直接加载
        self.protein_str = torch.load(protein_str_path)
        self.drug_llm = torch.load(drug_llm_path, weights_only=True)  # 也可以预加载

        self.protein_llm_path = protein_llm_path
        self.protein_llm = None  # 不直接加载大文件

    def get_drug_str(self, id):
        return self.drug_str.get(id)

    def get_protein_str(self, id):
        return self.protein_str.get(id)

    def get_drug_llm(self, id):
        return self.drug_llm.get(id)

    def load_protein_llm(self, id):
        # 只加载所需的部分
        if self.protein_llm is None:
            self.protein_llm = torch.load(self.protein_llm_path, weights_only=True)  # 初次加载整个文件
        return self.protein_llm.get(id)





class CustomDataSet(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __getitem__(self, item):
        return self.pairs[item]

    def __len__(self):
        return len(self.pairs)


def collate_fn(batch_data, cache, DATASET):

    N = len(batch_data)
    # 设定统一长度 L
    drug_llmL = 100
    protein_llmL = 1000
    drug_llm_dim = 384
    protein_llm_dim = 640
    labels_new = torch.zeros(N, dtype=torch.long)
    drug_graphs = [None] * N
    protein_graphs = [None] * N

    # 预分配 NumPy 数组，提高速度
    drug_llm_features = torch.zeros((N, drug_llmL, drug_llm_dim), dtype=torch.float32)
    protein_llm_features = torch.zeros((N, protein_llmL, protein_llm_dim), dtype=torch.float32)

    # compound_new = torch.zeros((N, drug_llmL), dtype=torch.long)
    # protein_new = torch.zeros((N, protein_llmL), dtype=torch.long)

    # 批量填充特征
    for i, pair in enumerate(batch_data):
        pair = pair.strip().split()  # 拆分每行数据
        drug_id, protein_id, smiles, sequence, label = pair[-5], pair[-4], pair[-3], pair[-2], pair[-1]
        label = float(label)
        labels_new[i] = np.int32(label)

        drug_llm_feature = cache.get_drug_llm(drug_id)  # [drug_llmL, drug_llm_dim]
        protein_llm_feature = cache.load_protein_llm(protein_id)  # [protein_llmL, protein_llm_dim]
        l = min(drug_llm_feature.shape[0], drug_llmL)  # 取 min 以防超出 L
        drug_llm_features[i, :l, :] = drug_llm_feature[:l, :]  # 直接填充到
        lp = min(protein_llm_feature.shape[0], protein_llmL)  # 取 min 以防超出 L
        protein_llm_features[i, :lp, :] = protein_llm_feature[:lp, :]  # 直接填充到 Tensor
        #-----消融实验########
        # compoundint = torch.from_numpy(label_smiles(
        #     smiles, CHARISOSMISET, drug_llmL))
        # compound_new[i] = compoundint
        # proteinint = torch.from_numpy(label_sequence(
        #     sequence, CHARPROTSET, protein_llmL))
        # protein_new[i] = proteinint
        ####################

        drug_graphs[i] = cache.get_drug_str(drug_id)
        protein_graphs[i] = cache.get_protein_str(protein_id)


    # 处理 PyG 图数据
    drug_graph_batch = Batch.from_data_list(drug_graphs)
    protein_graph_batch = Batch.from_data_list(protein_graphs)


    return (
            drug_graph_batch, protein_graph_batch,
            drug_llm_features, protein_llm_features,
            # compound_new, protein_new,
            labels_new
        )