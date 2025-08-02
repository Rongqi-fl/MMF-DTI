
import torch
import esm
import numpy as np
import pickle
from transformers import AutoTokenizer, AutoModel


# 加载预训练模型和分词器
print("load done")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device', device)


chem_tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
# print(chem_tokenizer.model_max_length)
chem_model = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM").to(device)

# 加载 ESM2 预训练模型
print("Loading ESM2 Model...")
esm_model, alphabet = esm.pretrained.esm2_t30_150M_UR50D()
batch_converter = alphabet.get_batch_converter()
esm_model = esm_model.to(device)
esm_model.eval()

# print(prot_tokenizer.model_max_length)
print("Model loaded successfully!")

# except Exception as e:
    # print(f"Error loading model: {e}")

# 读取数据集
def load_data(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
        print("read done")
        # print(lines)

    drug_list, protein_list, smiles_list, sequences_list, labels = [], [], [], [], []
    for line in lines:
        drug_id, protein_id, smiles, sequence, label = line.strip().split()
        drug_list.append(drug_id)
        protein_list.append(protein_id)
        smiles_list.append(smiles)
        sequences_list.append(sequence)
        labels.append(label)
    return drug_list, protein_list, smiles_list, sequences_list, labels




# 获取药物的嵌入向量
def get_drug_embedding(smiles, tokenizer, model):
    inputs = tokenizer(smiles, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state.squeeze(0).cpu()

def get_protein_embedding(sequence, model):
    data = [("protein", sequence)]  # ESM2 需要 tuple 形式的输入
    batch_labels, batch_strs, batch_tokens = batch_converter(data)
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[30], return_contacts=False)

    # 获取最后一层表示，并去掉 <cls> 和 <eos>
    embedding = results["representations"][30][:, 1:-1, :]  # (序列长度, 640)
    return embedding.squeeze(0).cpu() # 返回 NumPy 数组 (序列长度, 640)
# 主流程：生成字典并保存
def main(file_path):
    drug_list, protein_list, smiles_list, sequences_list, labels = load_data(file_path)
    print('smiles_list done')
    # print(smiles_list)
    print('sequences_list done')
    # print(sequences_list)

    drug_dict = {}
    protein_dict = {}

    for drug_id, protein_id, smiles, sequence in zip(drug_list, protein_list, smiles_list, sequences_list):
        # 如果药物（SMILES）序列已在字典中，跳过
        if smiles not in drug_dict:

            drug_embedding = get_drug_embedding(smiles, chem_tokenizer, chem_model)
            drug_dict[drug_id] = drug_embedding  # 将smiles与嵌入向量存入字典
            # print("add drug_embedding!")

        # 如果蛋白质（氨基酸序列）已在字典中，跳过
        if sequence not in protein_dict:
            protein_embedding = get_protein_embedding(sequence, esm_model)
            protein_dict[protein_id] = protein_embedding  # 存入字典
            # print("add protein_embedding!")

    # 保存字典

    # 保存成 Torch 格式
    torch.save(protein_dict, '../Input/EGFR/protein_llm.pt')
    torch.save(drug_dict, '../Input/EGFR/drug_llm.pt')
    print("Saved protein and drug embeddings as .pt")



if __name__ == '__main__':
    file_path = '../DataSets/Davis.txt'  # 替换为你的数据集路径
    main(file_path)









