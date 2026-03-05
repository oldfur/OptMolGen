import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from rdkit import Chem
from pathlib import Path


class FineTuneDataset(Dataset):
    def __init__(self, folder_path, n_dims, max_nodes=None):
        """
        Args:
            folder_path: .mol文件路径
            max_nodes: 最大节点数（用于填充）
            n_dims: 坐标维度（默认为3）
        """
        self.folder_path = Path(folder_path)
        self.mol_files = list(self.folder_path.glob('*.mol'))
        self.n_dims = n_dims
        
        # 预先计算每个分子的原子数，确定max_nodes
        self.atom_counts = []
        self.mol_objects = []
        self.unique_atomic_numbers = set()
        
        for mol_file in self.mol_files:
            mol = Chem.MolFromMolFile(str(mol_file), removeHs=False) # include H !!!
            self.mol_objects.append(mol)
            self.atom_counts.append(mol.GetNumAtoms())
            for atom in mol.GetAtoms():
                self.unique_atomic_numbers.add(atom.GetAtomicNum())
        
        # 设置最大节点数（用于padding）
        if max_nodes is None:
            self.max_nodes = max(self.atom_counts)
            print(f"Max nodes not provided. Using max nodes from dataset: {self.max_nodes}")
        else:
            self.max_nodes = max_nodes
        
        # 严格对应 GEOM_with_H 的映射
        # 索引 0:H, 1:B, 2:C, 3:N, 4:O, 5:F, 6:Al, 7:Si...
        self.atom_mapping = {
            1: 0, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5, 13: 6, 14: 7,
            15: 8, 16: 9, 17: 10, 33: 11, 35: 12, 53: 13, 80: 14, 83: 15
        }
        self.num_classes = len(self.atom_mapping)
        print(f"对应的原子序数: {sorted(list(self.unique_atomic_numbers))}")

    def __getitem__(self, idx):
        mol = self.mol_objects[idx]
        n_atoms = mol.GetNumAtoms()
        
        # 1. 提取坐标并进行“零均值化” (Zero-mean centering)
        # 这是为满足 VAE 中 assert_mean_zero_with_mask 的要求
        conformer = mol.GetConformer()
        pos = conformer.GetPositions()  # [n_atoms, 3]
        
        # 只对有效原子计算重心
        centroid = pos.mean(axis=0)
        pos_centered = pos - centroid
        
        x_padded = np.zeros((self.max_nodes, self.n_dims))
        x_padded[:n_atoms] = pos_centered
        
        # 2. 映射原子类型为 GEOM num_classes 维 One-hot
        h_cat_padded = np.zeros((self.max_nodes, self.num_classes))
        h_int_padded = np.zeros((self.max_nodes, 1))
        
        for i, atom in enumerate(mol.GetAtoms()):
            atomic_num = atom.GetAtomicNum()
            if atomic_num in self.atom_mapping:
                idx_cat = self.atom_mapping[atomic_num]
                h_cat_padded[i, idx_cat] = 1
            else:
                # 如果出现不在 GEOM 列表里的原子，通常映射到最后一个或抛出异常
                print(f"Warning: Atomic number {atomic_num} not in GEOM mapping.")
            
            # 填充形式电荷
            h_int_padded[i, 0] = atom.GetFormalCharge()
        
        # 3. 创建节点掩码 (注意维度要匹配 [max_nodes, 1])
        node_mask = np.zeros((self.max_nodes, 1))
        node_mask[:n_atoms] = 1

        # 4. 创建简单的全连接边掩码 (可选，根据你的模型需要)
        # edge_mask 通常是 [max_nodes, max_nodes, 1]
        
        return {
            'x': torch.FloatTensor(x_padded),               # [max_nodes, 3]
            'h_categorical': torch.FloatTensor(h_cat_padded), # [max_nodes, num_classes]
            'h_integer': torch.FloatTensor(h_int_padded),     # [max_nodes, 1]
            'node_mask': torch.FloatTensor(node_mask),       # [max_nodes, 1]
            'n_atoms': n_atoms
        }
            
    def __len__(self):
        return len(self.mol_files)

def collate_for_geoldm(batch):
    """
    将 batch 数据整理成适配 GeoLDM (VAE + LDM) 的格式
    """
    batch_size = len(batch)
    max_nodes = batch[0]['x'].shape[0]
    
    # 1. 初始化 Batch 张量
    # 坐标 x: [B, N, 3]
    x_batch = torch.zeros(batch_size, max_nodes, 3)
    # 分类特征 h_cat: [B, N, num_classes] (对应 GEOM num_classes维)
    h_cat_batch = torch.zeros(batch_size, max_nodes, batch[0]['h_categorical'].shape[-1])
    # 整数电荷 h_int: [B, N, 1]
    h_int_batch = torch.zeros(batch_size, max_nodes, 1)
    # 节点掩码 node_mask: [B, N, 1] (注意：VAE 内部通常需要最后一个维度为 1)
    node_mask_batch = torch.zeros(batch_size, max_nodes, 1)
    
    # 2. 填充数据
    file_names = []
    for i, item in enumerate(batch):
        x_batch[i] = item['x']
        h_cat_batch[i] = item['h_categorical']
        h_int_batch[i] = item['h_integer']
        node_mask_batch[i] = item['node_mask']
        file_names.append(item.get('file_name', f'mol_{i}'))
    
    # 3. 创建边掩码 (Edge Mask) [B, N, N, 1]
    # EGNN 需要知道哪些节点对之间存在有效的“消息传递”
    edge_mask = torch.zeros(batch_size, max_nodes, max_nodes, 1)
    for i in range(batch_size):
        # 只有 non-padding 的节点之间才有边
        n_atoms = int(torch.sum(node_mask_batch[i]).item())
        edge_mask[i, :n_atoms, :n_atoms, 0] = 1
        
        # 移除自环：i号节点到i号节点不传递消息 (EGNN 常见设定)
        diag_indices = torch.arange(n_atoms)
        edge_mask[i, diag_indices, diag_indices, 0] = 0

    # 4. 预先拼接一个 xh 用于某些 Dynamics 模块 (可选)
    # xh = [x, h_cat, h_int] -> [B, N, 3 + num_classes + 1]
    xh = torch.cat([x_batch, h_cat_batch, h_int_batch], dim=-1)

    return {
        'x': x_batch,                # [B, N, 3]
        'h_categorical': h_cat_batch, # [B, N, num_classes]
        'h_integer': h_int_batch,     # [B, N, 1]
        'xh': xh,                    # [B, N, 3 + num_classes + 1]
        'node_mask': node_mask_batch, # [B, N, 1]
        'edge_mask': edge_mask,       # [B, N, N, 1]
        'batch_size': batch_size,
        'file_names': file_names
    }

if __name__ == "__main__":
    batch_size = 16

    # create dataset and dataloader for fine-tuning
    dataset =FineTuneDataset(folder_path='dataset/BPN_structures', n_dims=3)
    # in fact, max_nodes for 2-MBA is 22(without H), 39(with H)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_for_geoldm,
        num_workers=0
    )

    # 使用示例
    for batch in dataloader:
        x = batch['x']                    
        h_cat = batch['h_categorical']    
        h_int = batch['h_integer']        
        node_mask = batch['node_mask']       
        edge_mask = batch['edge_mask']       
        print(f"Batch x shape: {x.shape}, h_cat shape: {h_cat.shape}, h_int shape: {h_int.shape}, \
              node_mask shape: {node_mask.shape}, edge_mask shape: {edge_mask.shape}")
        
        break