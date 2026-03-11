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
import copy
from models.equivariant_diffusion import utils as diffusion_utils
from train_test import train_epoch_finetune
import wandb
from qm9.sampling import sample
from qm9.analyze import analyze_stability_for_molecules
from qm9 import visualizer as qm9_visualizer
from eval_has_motif import batch_check_contains_motif
from build_geom_dataset import GeomDrugsDataset, GeomDrugsDataLoader, GeomDrugsTransform
from build_class_prior_dataset import build_data_list_from_pkl
from visualize_utils import save_rdkit_png, save_collapsed_plot, save_molecule_images
from models.egnn.lora import inject_lora_to_last_layers


def analyze_and_save_finetune(args, eval_args, device, generative_model,
                     nodes_dist, prop_dist, dataset_info, n_samples=10,
                     batch_size=10, save_to_xyz=False, epoch=None,
                     virtual_token=None, motif_smarts=None, atom_mapping=None):
    batch_size = min(batch_size, n_samples)
    assert n_samples % batch_size == 0
    molecules = {'one_hot': [], 'x': [], 'node_mask': []}
    start_time = time.time()

    # acquire base seed from args, default to 2026 if not specified
    base_seed = getattr(args, 'seed', 2026)
    for i in range(int(n_samples/batch_size)):
        # set random seed dynamically for each batch in torch
        current_seed = base_seed + i
        torch.manual_seed(current_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(current_seed)
        
        # start sample...
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, node_mask = sample(
            args, device, generative_model, dataset_info, 
            prop_dist=prop_dist, nodesxsample=nodesxsample, virtual_token=virtual_token)

        molecules['one_hot'].append(one_hot.detach().cpu())
        molecules['x'].append(x.detach().cpu())
        molecules['node_mask'].append(node_mask.detach().cpu())

        current_num_samples = (i+1) * batch_size
        secs_per_sample = (time.time() - start_time) / current_num_samples
        print('\t %d/%d Molecules generated at %.2f secs/sample' % (
            current_num_samples, n_samples, secs_per_sample))

        if save_to_xyz:
            id_from = i * batch_size
            qm9_visualizer.save_xyz_file(
                join(eval_args.model_path, 'eval/analyzed_molecules/'),
                one_hot, charges, x, dataset_info, id_from, name='molecule',
                node_mask=node_mask) # .txt document for visualization in Py3DMol

    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    stability_dict, rdkit_metrics = analyze_stability_for_molecules(
        molecules, dataset_info)
    
    # check motif presence...
    contains_dict = batch_check_contains_motif( x=molecules['x'], 
                                                one_hot=molecules['one_hot'],
                                                node_mask=molecules['node_mask'],
                                                atom_mapping=atom_mapping,
                                                motif_smarts=motif_smarts,
                                                fuzzy=True)
    
    # visualize valid molecules that contain the motif...
    viz_dir = join(eval_args.model_path, f'eval/viz_epoch_{epoch}')
    os.makedirs(viz_dir, exist_ok=True)
    print(f"Visualizing samples to {viz_dir}...")
    
    # visualize hits (those that contain the motif)
    if len(contains_dict['matched_mols']) > 0:
        print(f"Found {len(contains_dict['matched_mols'])} molecules matching the motif!")
        save_molecule_images(contains_dict['matched_mols'], viz_dir, prefix="hit")

    # visualize failed samples
    for mol_obj, batch_idx in contains_dict['failed_mols']:
        file_id = f"fail_{batch_idx}"
        
        if mol_obj is not None:
            # case A: transform succeeded but didn't match the motif 
            # 可能是结构太乱或官能团学歪了
            save_rdkit_png(mol_obj, join(viz_dir, f"{file_id}_mismatch.png"), label="No Motif Match")
        else:
            # case B: RDKit failed to determine bond orders 
            # 通常是原子距离太近导致 NaN 或价键爆炸
            # point cloud plotting tool
            save_collapsed_plot(
                molecules['x'][batch_idx], 
                molecules['one_hot'][batch_idx], 
                molecules['node_mask'][batch_idx], 
                atom_mapping, 
                join(viz_dir, f"{file_id}_collapsed.png")
            )

    return stability_dict, rdkit_metrics, contains_dict


def main():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="outputs/drugs_latent2",
                        help='Specify model path')
    parser.add_argument('--dataset_path', type=str, default="dataset/",
                        help='Path to the dataset for fine-tuning')
    parser.add_argument('--prior_dataset', type=str, default="geom_drugs_class_prior_2000",)
    parser.add_argument('--motif_name', type=str, default="BPN", 
                        help='Name of the motif to check for (e.g., BPN or 2-MBA)')
    parser.add_argument('--virtual_token_dim', type=int, default=256,
                        help='Dimension of the virtual token to be injected into EGNN')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate for fine-tuning')
    parser.add_argument('--ema_decay', type=float, default=0.999,           # TODO
                        help='Amount of EMA decay, 0 means off. A reasonable value is 0.999.')
    parser.add_argument('--finetune_batch_size', type=int, default=4, help='Batch size for fine-tuning')
    parser.add_argument('--sample_batch_size', type=int, default=4, help='Batch size for sampling during evaluation')
    parser.add_argument('--n_samples', type=int, default=8, help='Number of samples to generate for evaluation')
    parser.add_argument('--save_samples', type=bool, default=False, 
                        help='Whether to save sampled molecules as XYZ files for visualization')
    parser.add_argument('--save_epoch', type=int, default=10, help='Save checkpoint every N epochs')
    parser.add_argument('--report_epoch', type=int, default=2, help='Report evaluation results every N epochs')
    parser.add_argument('--n_epochs', type=int, default=400, help='Number of fine-tuning epochs')
    parser.add_argument('--wandb-mode', type=str, default='online',
                       choices=['online', 'offline', 'disabled', 'dryrun'],
                       help='Wandb mode: online/offline/disabled/dryrun')
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
    args.trainable_ae = False # 只微调 EGNN 相关的参数，保持 VAE 冻结
    utils.create_folders(args)
    print(args)

    # Retrieve dataloaders
    if len(args.conditioning) == 0 :
        dataloaders, charge_scale = None, None
        print("Not retrieving dataloaders since no conditioning is used.")
    else:
        dataloaders, charge_scale = dataset.retrieve_dataloaders(args)

    dataset_info = get_dataset_info(args.dataset, args.remove_h)
    atom_mapping = {nb: i for i, nb in enumerate(dataset_info['atomic_nb'])}

    # Create dataloader for fine-tuning
    if finetune_args.motif_name == '2-MBA':
        motif_smarts = 'Cc1ccccc1C=O'
    elif finetune_args.motif_name == 'BPN':
        motif_smarts = 'N#CCC(=O)c1ccccc1'
    finetune_dataset_path = os.path.join(finetune_args.dataset_path, f'{finetune_args.motif_name}_structures')
    finetune_dataset = FineTuneDataset(folder_path=finetune_dataset_path, n_dims=3)
    finetune_dataloader = DataLoader(
        finetune_dataset, batch_size=finetune_args.finetune_batch_size, shuffle=True,
        collate_fn=collate_for_geoldm, num_workers=0
    )
    print(f"Fine-tuning dataset loaded from {finetune_dataset_path} with {len(finetune_dataset)} samples.")

    # Create dataloader for class prior 
    # (optional, can be used for monitoring distribution shift during fine-tuning)
    class_prior_dataset_path = f"dataset/geom/{finetune_args.prior_dataset}"
    class_prior_data_list = build_data_list_from_pkl(class_prior_dataset_path)
    transform = GeomDrugsTransform(
        dataset_info=dataset_info,
        include_charges=False,   # 没有电荷数据
        device=device, 
        sequential=False         # no CustomBatchSampler
    )
    class_prior_dataset = GeomDrugsDataset(class_prior_data_list, transform=transform)
    class_prior_dataloader = GeomDrugsDataLoader(
        sequential=False,
        dataset=class_prior_dataset,
        batch_size=finetune_args.finetune_batch_size,
        shuffle=True,
        drop_last=False
    )

    # Load model
    if len(args.conditioning) == 0:
        generative_model, nodes_dist, prop_dist = get_latent_diffusion(args, device, dataset_info, finetune_args=finetune_args)
    else:
        generative_model, nodes_dist, prop_dist = get_latent_diffusion(args, device, dataset_info, 
                                                                       dataloaders['train'], finetune_args=finetune_args)
    # LoRA
    generative_model.dynamics.egnn = inject_lora_to_last_layers(generative_model.dynamics.egnn)
    if prop_dist is not None:
        property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
        prop_dist.set_normalizer(property_norms)
    generative_model.to(device)

    fn = 'generative_model_ema.npy' if args.ema_decay > 0 else 'generative_model.npy'
    flow_state_dict = torch.load(join(finetune_args.model_path, fn), map_location=device)
    # generative_model.load_state_dict(flow_state_dict)
    model_dict = generative_model.state_dict()
    pretrained_dict = {k: v for k, v in flow_state_dict.items() if k in model_dict} # 过滤权重中不存在的键
    model_dict.update(pretrained_dict) # 更新现有的 model_dict
    generative_model.load_state_dict(model_dict) # load model state dict
    # 打印没加载上的层(新加的层)
    missing_keys = [k for k in model_dict.keys() if k not in flow_state_dict]
    print(f"以下层是新定义的，将保持随机初始化: {missing_keys}")

    # Extract parameters and frozen
    generative_model.eval()
    for param in generative_model.parameters():
        param.requires_grad = False

    # 初始化 virtual_token for EGNN 
    virtual_dim = finetune_args.virtual_token_dim
    virtual_token = torch.nn.Parameter(torch.randn(1, virtual_dim).to(device) * 0.02)
    virtual_token.requires_grad = True # 确保 virtual_token 也被优化器更新
    
    # 局部解冻
    for name, param in generative_model.named_parameters():
        if 'lora_' in name or 'token_proj' in name or 'film' in name:
            param.requires_grad = True
    
    # 提取 LoRA 参数
    lora_params = [p for n, p in generative_model.named_parameters() if 'lora_' in n]

    # 构建分层学习率表
    trainable_params = [
        # Token 本身：需要最大的步长来在隐空间寻找 BPN 的位置
        {'params': [virtual_token], 'lr': finetune_args.lr},  
        
        # 投影层与 FiLM 层：负责翻译 Token 信号，建议保持在 1e-4
        {'params': generative_model.dynamics.egnn.token_proj.parameters(), 'lr': finetune_args.lr * 0.1},
        {'params': generative_model.dynamics.egnn.film.parameters(), 'lr': finetune_args.lr * 0.1},
        
        # LoRA 参数：直接微调 EGNN 的 node/coord 逻辑
        # 建议初始值设为 1e-4 (finetune_args.lr * 0.1)
        # 如果苯环依然变形，可以尝试进一步降低到 5e-5
        {'params': lora_params, 'lr': finetune_args.lr * 0.1}
    ]

    # AdamW optimizer
    optimizer = torch.optim.AdamW(trainable_params, weight_decay=1e-5)

    # 辅助参量设置
    # 初始化 gradnorm_queue 用于梯度裁剪监控
    gradnorm_queue = utils.Queue()
    gradnorm_queue.add(3000) # 初始一个较大的值

    # EMA
    if finetune_args.ema_decay > 0:
        ema = diffusion_utils.EMA(finetune_args.ema_decay)
        model_ema = copy.deepcopy(generative_model) # 复制一个 EMA 模型
    else:
        ema = None
        model_ema = generative_model

    # 属性正态化参数
    property_norms = None 
    if len(args.conditioning) > 0:
        property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)

    # wandb
    wandb.init(
        mode=finetune_args.wandb_mode,
        project="OptMolGen-Finetune",  # 你的项目名称
        config=finetune_args,          # 记录你的超参数
        name="virtual_token_experiment_{}".format(finetune_args.motif_name) # 本次实验的名称
    )

    # breakpoint()

    best_instance_nll = float('inf')

    # fine-tuning loop
    for epoch in range(finetune_args.n_epochs):
        total_loss, nll_instance, nll_prior = train_epoch_finetune(
            args=args,                                 # 原始配置参数
            loader=finetune_dataloader,                # FineTuneDataset 加载器
            class_prior_loader=class_prior_dataloader, # 用于监控分布的 class prior 加载器
            epoch=epoch,                               # 当前轮次
            model=generative_model,                    # 原始模型对象 (用于 grad clipping 等)
            model_dp=generative_model,                 # 实际运行的模型 (没 DataParallel 就传同一个)
            model_ema=model_ema,                       # EMA 模型对象
            ema=ema,                                   # EMA 更新器
            device=device,                             # 显卡设备
            dtype=dtype,                               # torch.float32
            property_norms=property_norms,             # 属性归一化系数
            optim=optimizer,                           # 仅包含 Token 和 Proj 的优化器
            nodes_dist=nodes_dist,                     # 节点数分布 (来自预训练模型)
            gradnorm_queue=gradnorm_queue,             # 梯度裁剪队列
            dataset_info=dataset_info,                 # 数据集元数据 (原子序数映射等)
            prop_dist=prop_dist,                       # 属性分布
            virtual_token=virtual_token,               # 声明的可学习 token
        )

        if epoch % finetune_args.report_epoch == 0:
            print("Epoch %d - Total Loss: %.4f, NLL Instance: %.4f, NLL Prior: %.4f" \
                  % (epoch, total_loss, nll_instance, nll_prior))
    
        # save checkpoint
        if epoch % finetune_args.save_epoch == 0 and epoch > 0 and \
            nll_instance < best_instance_nll:
            best_instance_nll = nll_instance 
            best_save_path = join(finetune_args.model_path, f'{finetune_args.motif_name}_virtual_token_epoch_{epoch}.pt')
            
            torch.save({
            'epoch': epoch,
            'instance_nll': best_instance_nll,
            'prior_nll': nll_prior,
            'virtual_token': virtual_token.data,
            'token_proj': generative_model.dynamics.egnn.token_proj.state_dict(),
            'film': generative_model.dynamics.egnn.film.state_dict()
            }, best_save_path)
            print(f"*** New Best Instance NLL: {best_instance_nll:.4f}! Saved to {best_save_path} ***")
            # 记录到 wandb 以便在图表中查看最优值
            wandb.log({"Best/Instance NLL": best_instance_nll}, commit=False)

        # # 原有的定期保存逻辑可以保留作为备份
        # if epoch % finetune_args.save_epoch == 0 and epoch > 0:
        #     save_path = join(finetune_args.model_path, f'{finetune_args.motif_name}_virtual_token_epoch_{epoch}.pt')
        #     torch.save({
        #         'virtual_token': virtual_token.data,
        #         'token_proj': generative_model.dynamics.egnn.token_proj.state_dict(),
        #         'film': generative_model.dynamics.egnn.film.state_dict()
        #     }, save_path)
        #     print(f"Periodic save: {save_path}")

        # sample and check motif presence in samples
        if epoch % (finetune_args.save_epoch * 2) == 0 and epoch > 0:
            # sample...
            stability_dict, rdkit_metrics, contains_dict = analyze_and_save_finetune(
                args, finetune_args, device, model_ema,
                nodes_dist, prop_dist, dataset_info,
                n_samples=finetune_args.n_samples,
                batch_size=finetune_args.sample_batch_size,
                save_to_xyz=finetune_args.save_samples,
                epoch=epoch,
                virtual_token=virtual_token,
                motif_smarts=motif_smarts,
                atom_mapping=atom_mapping
            )
            print(f"Epoch {epoch} evaluation results:")
            print("Stability:", stability_dict)
            print("RDKit Metrics:", rdkit_metrics)
            print("Motif Presence rate:", contains_dict['hit_rate'])
            print(f"Motif Presence count: {contains_dict['hit_count']} / {finetune_args.n_samples}")


if __name__ == "__main__":
    main()
