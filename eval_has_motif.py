# 判断生成 sample 的 3D 构象中是否含有指定官能团
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
import torch
import numpy as np
from rdkit.Chem import rdDetermineBonds
from scipy.spatial.distance import pdist


def check_contains_motif(x, one_hot, node_mask, atom_mapping, motif_smarts, fuzzy=False):
    index_to_atomic_num = {v: k for k, v in atom_mapping.items()}
    mask = node_mask.bool().squeeze()
    if mask.sum() == 0: return False, None  # 如果没有有效节点，直接返回 False

    scale = 1.01 if fuzzy else 1.0
    valid_coords = x[mask].detach().cpu().numpy() * scale
    valid_one_hot = one_hot[mask].detach().cpu().numpy()
    
    mol = Chem.RWMol()
    atomic_nums = [index_to_atomic_num[np.argmax(h)] for h in valid_one_hot]
    for atomic_num in atomic_nums:
        mol.AddAtom(Chem.Atom(int(atomic_num)))
    
    conf = Chem.Conformer(len(atomic_nums))
    for i, pos in enumerate(valid_coords):
        conf.SetAtomPosition(i, pos.tolist())
    mol.AddConformer(conf)
    
    try:
        new_mol = mol.GetMol()
        # 确定连接
        rdDetermineBonds.DetermineConnectivity(new_mol)

        if not fuzzy:
            # 确定键级 (解决 [C][N] 变成单键的问题)
            rdDetermineBonds.DetermineBondOrders(new_mol, charge=0)
            # 芳香化
            Chem.SanitizeMol(new_mol)
            query_mol = Chem.MolFromSmarts(motif_smarts)
        else: 
            # 模糊模式：将 SMARTS 中的特定键级符号 (#, =, :) 替换为任意键 ~
            # 例如 BPN: 'N#CCC(=O)c1ccccc1' -> '[N]~[C]~[C]~[C](=[O])~[c,C]1[c,C][c,C][c,C][c,C][c,C]1'
            fuzzy_smarts = motif_smarts.replace('#', '~').replace('=', '~').replace(':', '~').replace('c', 'C')
            query_mol = Chem.MolFromSmarts(fuzzy_smarts)
        
        if query_mol is None:
            query_mol = Chem.MolFromSmarts(motif_smarts) # 回退到原始 SMARTS，确保至少能进行子结构匹配

        # 执行匹配
        if new_mol.HasSubstructMatch(query_mol):
            return True, new_mol
        return False, new_mol
    except:
        return False, None

# 测试调用 (使用 BPN SMARTS)
# bpn_smarts = 'N#CCC(=O)c1ccccc1'
# success, _ = check_contains_motif_v2(..., bpn_smarts)


def batch_check_contains_motif(x, one_hot, node_mask, atom_mapping, motif_smarts, fuzzy=False):
    """
    批量检查 Batch 中生成的分子是否包含特定官能团 (Motif)
    
    Args:
        x: [batch_size, max_nodes, 3] 坐标张量
        one_hot: [batch_size, max_nodes, num_classes] 原子类型张量
        node_mask: [batch_size, max_nodes, 1] 掩码张量
        atom_mapping: 原子映射字典
        motif_smarts: 目标官能团的 SMARTS 字符串
        
    Returns:
        results: 包含命中数、命中率和匹配成功的分子对象的字典
    """
    batch_size = x.size(0)
    hit_count = 0
    matched_mols = []
    failed_mols = []

    for i in range(batch_size):
        # 提取第 i 个样本的数据
        # 注意：check_contains_motif 内部已经处理了 node_mask 的 bool 转换
        is_match, mol_obj = check_contains_motif(
            x[i], 
            one_hot[i], 
            node_mask[i], 
            atom_mapping,
            motif_smarts,
            fuzzy=fuzzy
        )
        
        if is_match:
            hit_count += 1
            matched_mols.append(mol_obj)
        else:
            # 即使没有命中，只要 mol_obj 不是 None (即成功转换成了分子图)，就记录
            # 如果 mol_obj 是 None，说明 rdDetermineBonds 崩溃了
            failed_mols.append((mol_obj, i))
            
    hit_rate = hit_count / batch_size if batch_size > 0 else 0.0
    
    return {
        "hit_count": hit_count,
        "hit_rate": hit_rate,
        "matched_mols": matched_mols,
        "failed_mols": failed_mols,
        "total_count": batch_size
    }