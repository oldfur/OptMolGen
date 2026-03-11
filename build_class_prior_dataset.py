import os
import glob
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from rdkit import Chem  # 确保已安装 rdkit
from configs.datasets_config import get_dataset_info
from build_geom_dataset import GeomDrugsDataset, GeomDrugsDataLoader, GeomDrugsTransform

def build_data_list_from_pkl(pkl_dir):
    """
    遍历 pkl_dir 中的所有 .pkl 文件，提取原子序数和坐标，
    构建 data_list (每个元素为 (n_atoms, 4) 的 numpy 数组) 和
    atomic_nb 列表（所有出现过的原子序数，排序后）。
    """
    pkl_files = glob.glob(os.path.join(pkl_dir, "*.pkl"))
    print(f"找到 {len(pkl_files)} 个 .pkl 文件")

    data_list = []
    all_atomic_nums = set()

    for i, fpath in enumerate(pkl_files):
        with open(fpath, "rb") as f:
            mol_data = pickle.load(f)

        # 1. 提取坐标（转换为 numpy，确保 float32）
        pos = mol_data['pos']
        if isinstance(pos, torch.Tensor):
            pos = pos.numpy()
        elif not isinstance(pos, np.ndarray):
            pos = np.array(pos)
        pos = pos.astype(np.float32)  # (n_atoms, 3)

        # 2. 提取原子序数（从 'mol' 对象）
        mol = mol_data['mol']
        if mol is None:
            raise ValueError(f"文件 {fpath} 中的 'mol' 为 None，无法获取原子类型。")
        atom_nums = np.array([atom.GetAtomicNum() for atom in mol.GetAtoms()], dtype=int)

        # 检查原子数是否一致
        assert len(atom_nums) == pos.shape[0], f"原子序数数量 ({len(atom_nums)}) 与坐标数量 ({pos.shape[0]}) 不匹配"

        # 3. 合并为 (n_atoms, 4) 数组
        mol_array = np.column_stack([atom_nums, pos])  # 形状 (n_atoms, 4)
        data_list.append(mol_array)

        # 4. 记录出现的原子序数
        all_atomic_nums.update(np.unique(atom_nums))

    # 5. 生成 atomic_nb（排序列表）
    atomic_nb = sorted(all_atomic_nums)
    atomic_nb_int = [int(x) for x in atomic_nb]
    print(f"所有原子序数: {atomic_nb_int}")
    # [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]

    return data_list

# ========== 使用示例 ==========
if __name__ == "__main__":
    # dataset = "geom_drugs_test_1000"
    dataset = "geom_drugs_class_prior_2000"
    dataset_info = get_dataset_info(dataset, remove_h=False)
    data_dir_path = f"dataset/geom/{dataset}" 
    data_list = build_data_list_from_pkl(data_dir_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 4

    # 创建 transform
    transform = GeomDrugsTransform(
        dataset_info=dataset_info,
        include_charges=False,   # 如果需要电荷信息可设为 True，但这里没有电荷数据
        device=device,
        sequential=False         # 如果使用 CustomBatchSampler 则 True，否则 False
    )

    # 构建 Dataset
    class_prior_dataset = GeomDrugsDataset(data_list, transform=transform)

    # 创建 DataLoader（两种模式任选一种）

    # 模式1：sequential=True（使用 CustomBatchSampler，要求 shuffle=False）
    # loader = GeomDrugsDataLoader(
    #     sequential=True,
    #     dataset=dataset,
    #     batch_size=32,
    #     shuffle=False,
    #     drop_last=False
    # )

    # 模式2：sequential=False（使用随机采样和 collate_fn，自动 padding）
    class_prior_loader = GeomDrugsDataLoader(
        sequential=False,
        dataset=class_prior_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False
    )

    # 测试一个 batch
    for batch in class_prior_loader:
        print("Batch keys:", batch.keys())
        print("positions shape:", batch['positions'].shape)   # (batch_size, max_n_atoms, 3)
        print("one_hot shape:", batch['one_hot'].shape)       # (batch_size, max_n_atoms, len(atomic_nb))
        print("atom_mask shape:", batch['atom_mask'].shape)     # (batch_size, max_n_atoms)
        break