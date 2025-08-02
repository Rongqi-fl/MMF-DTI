import os
import numpy as np
import torch.nn as nn
import pandas as pd
from rdkit import Chem
from drug_structure import *
from protein_structure import *
from torch_geometric.data import Batch, Data

# 读取文件
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 自动选择设备（GPU 或 CPU）



def process_file(file_path, contact_dir, aln_dir):
    drug_graphs = {}
    protein_graphs = {}

    # 读取数据，明确指定 'drug_id' 和 'protein_id' 列为字符串类型
    df = pd.read_csv(file_path, sep=' ', header=None, names=['drug_id', 'protein_id', 'smiles', 'sequence', 'label'],
                     dtype={'drug_id': str, 'protein_id': str})

    for _, row in df.iterrows():
        drug_id, protein_id, smiles, sequence = row['drug_id'], row['protein_id'], row['smiles'], row['sequence']

       # 处理药物
        if drug_id not in drug_graphs:
            drug_graph = smile_to_graph(smiles)

            if drug_graph:
                drug_size, drug_node_features, drug_edge_index, drug_edge_weight, drug_edge_type = drug_graph

                drug_graphs[drug_id] = (Data(
                    x=drug_node_features,
                    edge_index=drug_edge_index,
                    edge_attr=drug_edge_weight,
                    edge_type=drug_edge_type,
                    num_nodes=drug_size

                ))


        # 处理蛋白质
        if protein_id not in protein_graphs:

            protein_graph = target_to_graph(protein_id, sequence, contact_dir, aln_dir)
            if protein_graph:
                protein_size, protein_node_features, protein_edge_index = protein_graph

                protein_graphs[protein_id] = (Data(
                    x=protein_node_features,
                    edge_index=protein_edge_index,
                    num_nodes=protein_size
                ))

    return drug_graphs, protein_graphs


# 示例调用
file_path = "../DataSets/KIBA.txt"  # 替换为你的txt文件路径
contact_dir = r"D:\FB_Data\Data\KIBA\npy"
aln_dir = r"D:\FB_Data\Data\KIBA\aln"
drug_graph, protein_graph = process_file(file_path, contact_dir, aln_dir)

# 保存结果

torch.save(drug_graph, "../Input/KIBA/drug_graph.pt")
torch.save(protein_graph, "../Input/KIBA/protein_graph.pt")

print("add protein_graph!")
print("add drug_graph!")


