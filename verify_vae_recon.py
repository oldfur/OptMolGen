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

def verify_vae_reconstruction(vae: EnHierarchicalVAE, dataloader, device):
    vae.to(device)
    vae.eval()
    
    total_rmsd = 0
    total_h_acc = 0
    count = 0

    # 1. 初始化 tqdm 进度条
    pbar = tqdm(dataloader, desc=">>> VAE 重构验证", unit="batch")
    
    print("\n开始处理隐空间编码与解码...")
    
    with torch.no_grad():
        for i, batch in enumerate(pbar):
            # 2. 准备数据
            x = batch['x'].to(device)
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

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="outputs/drugs_latent2",
                        help='Specify model path')
    parser.add_argument('--dataset_path', type=str, default="dataset/BPN_structures",
                        help='Path to the dataset for fine-tuning')

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
    finetune_dataset = FineTuneDataset(folder_path=finetune_args.dataset_path, n_dims=3)
    finetune_dataloader = DataLoader(
        finetune_dataset, batch_size=args.batch_size // 8, shuffle=True,
        collate_fn=collate_for_geoldm, num_workers=0
    )
    
    breakpoint()
    
    # Verify VAE reconstruction quality on the fine-tuning dataset
    verify_vae_reconstruction(vae, finetune_dataloader, device)


if __name__ == "__main__":
    main()
