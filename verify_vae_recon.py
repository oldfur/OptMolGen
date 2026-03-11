# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import utils
import argparse
from qm9 import dataset
from models.models import get_latent_diffusion
import os
from models.equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask, sample_gaussian_with_mask
import torch
import time
import pickle
from configs.datasets_config import get_dataset_info
from os.path import join
from qm9.utils import prepare_context, compute_mean_mad

try:
    from qm9 import rdkit_functions
except ModuleNotFoundError:
    print('Not importing rdkit functions.')

from build_finetune_dataset import FineTuneDataset, collate_for_geoldm
from torch.utils.data import DataLoader
from torch.nn import functional as F
from models.equivariant_diffusion.en_diffusion import EnHierarchicalVAE
from tqdm import tqdm
from eval_has_motif import check_contains_motif

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds



def check_mask_correct(variables, node_mask):
    for variable in variables:
        assert_correctly_masked(variable, node_mask)
    
def sample_combined_position_feature_noise(self, n_samples, n_nodes, node_mask):
    """
    Samples mean-centered normal noise for z_x, and standard normal noise for z_h.
    """
    z_x = sample_center_gravity_zero_gaussian_with_mask(
        size=(n_samples, n_nodes, self.n_dims), device=node_mask.device,
        node_mask=node_mask)
    z_h = sample_gaussian_with_mask(
        size=(n_samples, n_nodes, self.in_node_nf), device=node_mask.device,
        node_mask=node_mask)
    z = torch.cat([z_x, z_h], dim=2)
    return z

def sample_normal(self, mu, sigma, node_mask, fix_noise=False):
    """Samples from a Normal distribution."""
    bs = 1 if fix_noise else mu.size(0)
    eps = self.sample_combined_position_feature_noise(bs, mu.size(1), node_mask)
    return mu + sigma * eps

def verify_vae_reconstruction(vae: EnHierarchicalVAE, dataloader, device, motif_smarts=None):
    vae.to(device)
    vae.eval()
    
    total_rmsd = 0
    total_h_acc = 0
    count = 0
    total_samples = len(dataloader.dataset)

    hit_count_before = 0
    hit_count_after = 0
    atom_mapping = {1: 0, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5, 13: 6, 14: 7, 15: 8, 
                16: 9, 17: 10, 33: 11, 35: 12, 53: 13, 80: 14, 83: 15}
    # atom_mapping = {nb: i for i, nb in enumerate(dataset_info['atomic_nb'])} # if the dataset_info available

    # 1. 初始化 tqdm 进度条
    pbar = tqdm(dataloader, desc=">>> VAE 重构验证", unit="batch")
    
    print("\n开始处理隐空间编码与解码...")
    
    with torch.no_grad():
        for i, batch in enumerate(pbar):
            # 2. 准备数据
            x = batch['x'].to(device)
            batch_size = x.size(0)
            h_cat = batch['h_categorical'].to(device)
            # h_int = batch['h_integer'].to(device)
            h_int = torch.zeros(h_cat.shape[0], h_cat.shape[1], 0).to(device)
            node_mask = batch['node_mask'].to(device)
            edge_mask = batch['edge_mask'].to(device)
            
            h_input = {'categorical': h_cat, 
                       'integer': h_int} # no h_integer
            
            # 3. VAE 流程
            # 编码
            z_x_mu, _, z_h_mu, _ = vae.encode(x, h_input, node_mask, edge_mask)
            z_xh = torch.cat([z_x_mu, z_h_mu], dim=2)
            # 解码
            x_recon, h_recon = vae.decode(z_xh, node_mask, edge_mask)

            # 4. 指标计算
            # RMSD
            sq_dist = torch.sum((x_recon - x)**2, dim=-1) * node_mask.squeeze(-1)
            rmsd = torch.sqrt(torch.sum(sq_dist) / torch.sum(node_mask)).item()
            # contains motif
            if motif_smarts is not None:
                # --- 失败样本诊断模块 ---
                for j in range(batch_size):
                    # 检查 VAE 之前 (原始数据)
                    is_hit_before, mol_before = check_contains_motif(
                        x[j], h_cat[j], 
                        node_mask[j], atom_mapping, motif_smarts, fuzzy=True
                    )
                    # 检查 VAE 之后 (重构数据)
                    is_hit_after, mol_after = check_contains_motif(
                        x_recon[j], h_recon['categorical'][j], 
                        node_mask[j], atom_mapping, motif_smarts, fuzzy=True
                    )

                    if is_hit_before: hit_count_before += 1
                    if is_hit_after: hit_count_after += 1

                    # 打印诊断信息
                    # 情况 A: 原始数据集就没匹配上 (你提到的那 4%)
                    if not is_hit_before:
                        smiles = Chem.MolToSmiles(mol_before) if mol_before else "Structure Break"
                        print(f"\n[Dataset Fail] Batch {i}, Sample {j}: {smiles}")

                    # 情况 B: 原始有，但 VAE 重构后丢了 (VAE 造成的损失)
                    elif not is_hit_after:
                        smiles_recon = Chem.MolToSmiles(mol_after) if mol_after else "Structure Break"
                        print(f"\n[Recon Fail] Batch {i}, Sample {j}: {smiles_recon}")
                # -----------------------
                
                # --- batch 内的统计信息 ---
                # contains_motif_info_before = batch_check_contains_motif(x, h_cat, node_mask, atom_mapping, motif_smarts)
                # contains_motif_info_after = batch_check_contains_motif(x_recon, h_recon['categorical'], node_mask, atom_mapping, motif_smarts)
                # print(f"Batch {i+1}: 包含目标官能团: {contains_motif_info_before['hit_count']} / {x.size(0)}")
                # print(f"Batch {i+1}: 包含目标官能团: {contains_motif_info_after['hit_count']} / {x_recon.size(0)}")
                # hit_count_before += contains_motif_info_before['hit_count']
                # hit_count_after += contains_motif_info_after['hit_count']
            
            # 原子类型准确率
            true_types = torch.argmax(h_cat, dim=-1)
            pred_types = torch.argmax(h_recon['categorical'], dim=-1)
            correct_types = ((true_types == pred_types) * node_mask.squeeze(-1)).sum().item()
            acc = correct_types / node_mask.sum().item()

            # 5. 更新累计数据
            total_rmsd += rmsd
            total_h_acc += acc
            count += 1

            # 6. 动态更新进度条右侧的显示信息 (Postfix)
            pbar.set_postfix({
                'RMSD': f"{rmsd:.4f}", 
                'Acc': f"{acc*100:.1f}%",
                'Avg_R': f"{total_rmsd/count:.4f}"
            })

            # 在验证循环中调用一次即可
            if i == 0:
                probe_atom_mapping(vae, batch, device)

    # 最终总结
    print("\n" + "="*40)
    print(f"验证完成！平均 RMSD: {total_rmsd/count:.4f} 埃")
    print(f"平均原子类型准确率: {(total_h_acc/count)*100:.2f}%")
    print(f"before vae, 包含目标官能团的比例: {hit_count_before/total_samples:.2%}")
    print(f"after vae, 包含目标官能团的比例: {hit_count_after/total_samples:.2%}")
    print("="*40)

    return total_rmsd/count

def probe_atom_mapping(vae: EnHierarchicalVAE, batch, device):
    vae.eval()
    with torch.no_grad():
        # 1. 正常执行编码解码
        x, h_cat = batch['x'].to(device), batch['h_categorical'].to(device)
        h_int = torch.zeros(h_cat.shape[0], h_cat.shape[1], 0).to(device)
        node_mask, edge_mask = batch['node_mask'].to(device), batch['edge_mask'].to(device)
        
        h_input = {'categorical': h_cat, 'integer': torch.zeros_like(h_int)}
        z_x, _, z_h, _ = vae.encode(x, h_input, node_mask, edge_mask)
        z_xh = torch.cat([z_x, z_h], dim=2)
        _, h_recon = vae.decode(z_xh, node_mask, edge_mask)

        # 2. 提取原始索引和重构后的索引
        true_indices = torch.argmax(h_cat, dim=-1)[node_mask.squeeze(-1) > 0]
        pred_indices = torch.argmax(h_recon['categorical'], dim=-1)[node_mask.squeeze(-1) > 0]
        
        # 3. 打印对比（只看前几个原子）
        print("\n[原子映射探测结果]:")
        for t, p in zip(true_indices[:15], pred_indices[:15]):
            print(f"输入索引: {t.item()} ---> 解码索引: {p.item()}")


def debug_dataset_sample(dataloader, motif_smarts, normalization_factor=1.0):
    # 1. 你的原子映射表
    atom_mapping = {
        1: 0, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5, 13: 6, 14: 7,
        15: 8, 16: 9, 17: 10, 33: 11, 35: 12, 53: 13, 80: 14, 83: 15
    }


    """
    从 DataLoader 中取一个样本进行深度诊断
    """
    index_to_atomic_num = {v: k for k, v in atom_mapping.items()}
    
    # 获取第一个 Batch
    data = next(iter(dataloader))
    
    # 提取第一个分子
    i = 0 
    x = data['x'][i]
    one_hot = data['h_categorical'][i]
    node_mask = data['node_mask'][i].bool().squeeze()
    
    # 应用缩放因子（如果数据是被标准化过的）
    valid_coords = x[node_mask].cpu().numpy() * normalization_factor
    valid_one_hot = one_hot[node_mask].cpu().numpy()
    
    print(f"--- 样本诊断报告 ---")
    print(f"有效原子数: {len(valid_coords)}")
    
    # 还原原子序数
    atomic_nums = [index_to_atomic_num[np.argmax(h)] for h in valid_one_hot]
    print(f"原子序数列表: {atomic_nums}")
    
    # 构建 RDKit 分子
    mol = Chem.RWMol()
    for atomic_num in atomic_nums:
        mol.AddAtom(Chem.Atom(int(atomic_num)))
    
    conf = Chem.Conformer(len(atomic_nums))
    for j, pos in enumerate(valid_coords):
        conf.SetAtomPosition(j, pos.tolist())
    mol.AddConformer(conf)
    
    try:
        new_mol = mol.GetMol()
        # 第一步：推断连接性
        rdDetermineBonds.DetermineConnectivity(new_mol)
        print(f"推断出的化学键数量: {new_mol.GetNumBonds()}")
        
        # 第二步：Sanitize (处理芳香性)
        Chem.SanitizeMol(new_mol)
        
        # 第三步：打印还原出的 SMILES
        print(f"还原出的 SMILES: {Chem.MolToSmiles(new_mol)}")
        
        # 第四步：匹配测试
        query = Chem.MolFromSmarts(motif_smarts)
        if new_mol.HasSubstructMatch(query):
            print("✅ 匹配成功！官能团已识别。")
        else:
            print("❌ 匹配失败。SMARTS 无法在还原结构中找到目标。")
            
    except Exception as e:
        print(f"🚨 运行错误: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="outputs/drugs_latent2",
                        help='Specify model path')
    parser.add_argument('--dataset_path', type=str, default="dataset/",
                        help='Path to the dataset for fine-tuning')
    parser.add_argument('--motif_name', type=str, default="BPN", 
                        help='Name of the motif to check for (e.g., BPN or 2-MBA)')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for verification')

    finetune_args, unparsed_args = parser.parse_known_args()

    assert finetune_args.model_path is not None

    with open(join(finetune_args.model_path, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    # CAREFUL with this -->
    if not hasattr(args, 'normalization_factor'):
        args.normalization_factor = 1
    if not hasattr(args, 'aggregation_method'):
        args.aggregation_method = 'sum'

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")
    args.device = device
    dtype = torch.float32
    utils.create_folders(args)
    print(args)

    # Retrieve dataloaders
    if len(args.conditioning) == 0 :
        dataloaders, charge_scale = None, None
        print("Not retrieving dataloaders since no conditioning is used.")
    else:
        dataloaders, charge_scale = dataset.retrieve_dataloaders(args)

    dataset_info = get_dataset_info(args.dataset, args.remove_h)

    # Load model
    if len(args.conditioning) == 0:
        generative_model, nodes_dist, prop_dist = get_latent_diffusion(args, device, dataset_info)
    else:
        generative_model, nodes_dist, prop_dist = get_latent_diffusion(args, device, dataset_info, dataloaders['train'])
    if prop_dist is not None:
        property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
        prop_dist.set_normalizer(property_norms)
    generative_model.to(device)

    fn = 'generative_model_ema.npy' if args.ema_decay > 0 else 'generative_model.npy'
    flow_state_dict = torch.load(join(finetune_args.model_path, fn), map_location=device)
    generative_model.load_state_dict(flow_state_dict)

    # Extract the VAE parameters and frozen it. 
    vae = generative_model.vae
    vae.eval() # 设置为推断模式，关闭 Dropout 和 BatchNorm 的更新
    for param in vae.parameters():
        param.requires_grad = False
    print("VAE components frozen successfully.")

    # Create dataloader for fine-tuning
    finetune_dataset_path = os.path.join(finetune_args.dataset_path, f'{finetune_args.motif_name}_structures')
    finetune_dataset = FineTuneDataset(folder_path=finetune_dataset_path, n_dims=3)
    finetune_dataloader = DataLoader(
        finetune_dataset, batch_size=args.batch_size // 8, shuffle=True,
        collate_fn=collate_for_geoldm, num_workers=0
    )
    
    # breakpoint()

    if finetune_args.motif_name == '2-MBA':
        motif_smarts = 'Cc1ccccc1C=O'
    elif finetune_args.motif_name == 'BPN':
        motif_smarts = 'N#CCC(=O)c1ccccc1'
    
    # Verify VAE reconstruction quality on the fine-tuning dataset
    verify_vae_reconstruction(vae, finetune_dataloader, device, motif_smarts=motif_smarts)

    # --- 运行测试 ---
    # debug_dataset_sample(finetune_dataloader, motif_smarts, normalization_factor=1.0)


if __name__ == "__main__":
    main()
