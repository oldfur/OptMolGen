import os
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import numpy as np


def save_rdkit_svg(mol, path, label=""):
    """保存 RDKit 能识别出的分子图"""
    try:
        Draw.MolToFile(mol, path, size=(400, 400), legend=label)
    except Exception as e:
        print(f"Failed to draw RDKit image: {e}")

def save_collapsed_plot(x, one_hot, node_mask, atom_mapping, path):
    """保存 RDKit 崩溃时的原始原子点云图"""
    idx_to_num = {v: k for k, v in atom_mapping.items()}
    mask = node_mask.bool().squeeze()
    coords = x[mask].cpu().numpy()
    types = one_hot[mask].cpu().numpy()
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    for i, pos in enumerate(coords):
        atomic_num = idx_to_num[np.argmax(types[i])]
        color = 'red' if atomic_num == 8 else 'blue' if atomic_num == 7 else 'gray'
        ax.scatter(pos[0], pos[1], pos[2], s=80, color=color)
        ax.text(pos[0], pos[1], pos[2], f"{atomic_num}")
    
    plt.title("Atomic Cloud (RDKit Failed)")
    plt.savefig(path)
    plt.close()


def save_molecule_images(mol_list, output_dir, prefix="hit"):
    from rdkit.Chem import Draw
    import os
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for i, mol in enumerate(mol_list):
        if mol is None: continue
        
        # 使用 MolToFile 保存为 SVG
        svg_path = os.path.join(output_dir, f"{prefix}_{i}.svg")
        Draw.MolToFile(mol, svg_path, size=(400, 400))


def visualize_failures(contains_dict, x, one_hot, node_mask, atom_mapping, out_path):
    """
    专门可视化失败样本：
    1. 转换成功但未命中 Motif 的分子。
    2. 转换失败（rdDetermineBonds 崩溃）的原始原子堆叠。
    """
    os.makedirs(out_path, exist_ok=True)
    index_to_atomic_num = {v: k for k, v in atom_mapping.items()}
    
    # 遍历 batch 中的所有失败情况
    for mol_obj, idx in contains_dict['failed_mols']:
        if mol_obj is not None:
            # 情况 A: 分子图生成了，但没匹配上（通常是结构太乱）
            img = Draw.MolToImage(mol_obj, size=(400, 400), legend=f"Failed Match Index {idx}")
            img.save(os.path.join(out_path, f"failed_match_{idx}.svg"))
        else:
            # 情况 B: rdDetermineBonds 崩溃，可视化原始 3D 坐标
            visualize_atomic_cloud(
                x[idx], one_hot[idx], node_mask[idx], 
                index_to_atomic_num, 
                save_path=os.path.join(out_path, f"collapsed_atoms_{idx}.svg")
            )

def visualize_atomic_cloud(x, one_hot, node_mask, idx_to_num, save_path):
    """使用 Matplotlib 画出那些无法成键的‘原子云’，诊断塌陷问题"""
    mask = node_mask.bool().squeeze()
    coords = x[mask].cpu().numpy()
    types = one_hot[mask].cpu().numpy()
    
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    for i, pos in enumerate(coords):
        atomic_num = idx_to_num[np.argmax(types[i])]
        # 根据原子序数简单区分颜色
        color = 'red' if atomic_num == 8 else 'blue' if atomic_num == 7 else 'gray'
        ax.scatter(pos[0], pos[1], pos[2], s=100, color=color, alpha=0.6)
        ax.text(pos[0], pos[1], pos[2], f"{atomic_num}", fontsize=8)
    
    ax.set_title("Collapsed Atomic Structure (RDKit Failed)")
    plt.savefig(save_path)
    plt.close()
