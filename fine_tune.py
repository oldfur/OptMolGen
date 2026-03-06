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
from train_test import train_epoch_virtual_token
import wandb

def main():
    # args
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default="outputs/drugs_latent2",
                        help='Specify model path')
    parser.add_argument('--dataset_path', type=str, default="dataset/BPN_structures",
                        help='Path to the dataset for fine-tuning')
    parser.add_argument('--virtual_token_dim', type=int, default=256,
                        help='Dimension of the virtual token to be injected into EGNN')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate for fine-tuning')
    parser.add_argument('--ema_decay', type=float, default=0.999,           # TODO
                        help='Amount of EMA decay, 0 means off. A reasonable value is 0.999.')

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

    # Create dataloader for fine-tuning
    finetune_dataset = FineTuneDataset(folder_path=finetune_args.dataset_path, n_dims=3)
    finetune_dataloader = DataLoader(
        finetune_dataset, batch_size=args.batch_size // 8, shuffle=True,
        collate_fn=collate_for_geoldm, num_workers=0
    )

    # Load model
    if len(args.conditioning) == 0:
        generative_model, nodes_dist, prop_dist = get_latent_diffusion(args, device, dataset_info, finetune_args=finetune_args)
    else:
        generative_model, nodes_dist, prop_dist = get_latent_diffusion(args, device, dataset_info, 
                                                                       dataloaders['train'], finetune_args=finetune_args)
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
    generative_model.load_state_dict(model_dict) # 加载
    # 打印没加载上的层，确保只有新加的层
    missing_keys = [k for k in model_dict.keys() if k not in flow_state_dict]
    print(f"以下层是新定义的，将保持随机初始化: {missing_keys}")

    # Extract parameters and frozen
    generative_model.eval()
    for param in generative_model.parameters():
        param.requires_grad = False

    # 初始化 virtual_token for EGNN 
    virtual_dim = finetune_args.virtual_token_dim
    virtual_token = torch.nn.Parameter(torch.randn(1, virtual_dim).to(device) * 0.02)

    # 局部解冻：针对 EGNN 内部新增的层
    if hasattr(generative_model.dynamics.egnn, 'token_proj'):
        for param in generative_model.dynamics.egnn.token_proj.parameters():
            param.requires_grad = True
    if hasattr(generative_model.dynamics.egnn, 'film'):
        for param in generative_model.dynamics.egnn.film.parameters():
            param.requires_grad = True

    # 筛选出所有需要更新的参数
    trainable_params = [
        {'params': [virtual_token], 'lr': finetune_args.lr},  
        # Token 本身通常需要稍大的学习率
        {'params': generative_model.dynamics.egnn.token_proj.parameters(), 'lr': finetune_args.lr * 0.1}, 
        # 投影层建议较小，保持稳定
        {'params': generative_model.dynamics.egnn.film.parameters(), 'lr': finetune_args.lr * 0.1} 
        # film 层也建议较小学习率
    ]

    # AdamW optimizer
    optimizer = torch.optim.AdamW(trainable_params, weight_decay=1e-2)

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
        project="OptMolGen-Finetune",  # 你的项目名称
        config=finetune_args,          # 记录你的超参数
        name="virtual_token_experiment" # 本次实验的名称
    )

    # breakpoint()

    # fine-tuning loop
    for epoch in range(args.n_epochs):
    # 调用你定义的训练函数
        train_epoch_virtual_token(
            args=args,                    # 原始配置参数
            loader=finetune_dataloader,   # 你的 FineTuneDataset 加载器
            epoch=epoch,                  # 当前轮次
            model=generative_model,       # 原始模型对象 (用于 grad clipping 等)
            model_dp=generative_model,    # 实际运行的模型 (如果你没用 DataParallel，就传同一个)
            model_ema=model_ema,          # EMA 模型对象
            ema=ema,                      # EMA 更新器
            device=device,                # 显卡设备
            dtype=dtype,                  # torch.float32
            property_norms=property_norms,# 属性归一化系数
            optim=optimizer,              # 仅包含 Token 和 Proj 的优化器
            nodes_dist=nodes_dist,        # 节点数分布 (来自预训练模型)
            gradnorm_queue=gradnorm_queue,# 梯度裁剪队列
            dataset_info=dataset_info,    # 数据集元数据 (原子序数映射等)
            prop_dist=prop_dist,          # 属性分布
            virtual_token=virtual_token   # 你声明的可学习变量
        )
    
        # save checkpoint every 10 epochs
        if epoch % 10 == 0 and epoch > 0: 
            save_path = join(finetune_args.model_path, f'virtual_token_epoch_{epoch}.pt')
            torch.save({
                'virtual_token': virtual_token.data,
                'token_proj': generative_model.dynamics.egnn.token_proj.state_dict(),
                'film': generative_model.dynamics.egnn.film.state_dict()
            }, save_path)
            print(f"Saved {finetune_args.model_path} adapter to {save_path}")

if __name__ == "__main__":
    main()
