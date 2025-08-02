import numpy as np
from rdkit import Chem
import networkx as nx
import torch


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])




def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


def get_bond_weight(bond):
    bond_type = bond.GetBondType()
    if bond_type == Chem.rdchem.BondType.SINGLE:
        return 0, 1.0
    elif bond_type == Chem.rdchem.BondType.DOUBLE:
        return 1, 1.5
    elif bond_type == Chem.rdchem.BondType.TRIPLE:
        return 2, 2.0
    elif bond_type == Chem.rdchem.BondType.AROMATIC:
        return 3, 1.2
    else:
        return 0, 1.0  # 默认当作单键处理

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)
    c_size = mol.GetNumAtoms()

    # 处理原子特征
    features = []
    for atom in mol.GetAtoms():
        features.append(atom_features(atom))

    features = torch.tensor(np.stack(features),dtype=torch.float32)  # 转换为 (num_atoms, feature_dim)

    # 处理边
    edges = []
    edge_weights = []
    edge_type = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()

        bond_type_id, weight = get_bond_weight(bond)
        for u, v in [(i, j), (j, i)]:
            edges.append([u, v])
            edge_weights.append(weight)
            edge_type.append(bond_type_id)

        # 添加自环
    for i in range(c_size):
        # edges.append((i, i))
        edges.append([i, i])
        edge_weights.append(1.0)
        edge_type.append(0)

    # #### g = nx.Graph(edges).to_directed()  # 无向图转为有向图
    # edges = np.array(edges, dtype=np.int64).T
    edges = torch.tensor(edges, dtype=torch.long).t().contiguous()  # shape: [2, num_edges]
    edge_weight = torch.tensor(edge_weights, dtype=torch.float32)
    edge_type = torch.tensor(edge_type, dtype=torch.long)
    return c_size, features, edges, edge_weight, edge_type





