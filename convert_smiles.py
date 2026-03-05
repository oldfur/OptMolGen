from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import PyMol
import pandas as pd

def smiles_to_3d(smiles, output_file=None, add_hs=True, optimize=True):
    """
    将SMILES转换为3D结构
    
    参数：
    smiles: SMILES字符串
    output_file: 输出文件路径（如 .mol, .sdf, .pdb）
    add_hs: 是否添加氢原子
    optimize: 是否进行力场优化
    """
    # 1. 从SMILES创建分子对象
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("无效的SMILES字符串")
    
    # 2. 添加氢原子（可选，但推荐）
    if add_hs:
        mol = Chem.AddHs(mol)
    
    # 3. 生成3D坐标（核心步骤）
    # 使用ETKDG方法，这是目前RDKit最好的构象生成算法
    params = AllChem.ETKDGv3()  # 最新版本算法
    params.randomSeed = 42      # 设置随机种子保证可重复性
    success = AllChem.EmbedMolecule(mol, params)
    
    if success == -1:
        print("警告：构象生成失败，尝试替代方法")
        # 备用方法：更简单的距离几何
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())
    
    # 4. 力场优化（可选，但推荐）
    if optimize:
        try:
            # UFF力场优化
            AllChem.UFFOptimizeMolecule(mol, maxIters=1000)
        except:
            print("力场优化失败，使用原始坐标")
    
    # 5. 保存文件（如果指定了输出路径）
    if output_file:
        if output_file.endswith('.mol') or output_file.endswith('.sdf'):
            writer = Chem.SDWriter(output_file)
            writer.write(mol)
            writer.close()
        elif output_file.endswith('.pdb'):
            Chem.MolToPDBFile(mol, output_file)
        print(f"3D结构已保存至: {output_file}")
    
    return mol


def batch_smiles_to_3d(smiles_list, names=None, output_dir='3d_structures'):
    """
    批量转换SMILES为3D结构
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    results = []
    for i, smiles in enumerate(smiles_list):
        name = names[i] if names else f"mol_{i+1}"
        try:
            output_file = os.path.join(output_dir, f"{name}.mol")
            mol = smiles_to_3d(smiles, output_file=output_file)
            results.append((name, mol, True))
            print(f"✓ {name}: 转换成功")
        except Exception as e:
            results.append((name, None, False))
            print(f"✗ {name}: 转换失败 - {e}")
    
    return results

if __name__ == "__main__":
    # 1. 试验单个SMILES转换
    # print("try smiles to 3d structure...")
    # # 使用示例：邻甲基苯醛 (o-tolualdehyde)
    # smiles = "CC1=CC=CC=C1C=O"  # 邻甲基苯醛的SMILES
    # mol_3d = smiles_to_3d(smiles, output_file=None)
    # # 查看分子信息
    # print(f"原子数: {mol_3d.GetNumAtoms()}")
    # print(f"键数: {mol_3d.GetNumBonds()}")
    

    # 2. 试验批量SMILES转换
    # print("try smiles to 3d structure in batch...")
    # # 示例：批量转换
    # smiles_list = [
    #     "CC1=CC=CC=C1C=O",  # 邻甲基苯醛
    #     "C1=CC=C(C=C1)C=O", # 苯甲醛
    #     "CC(C)C",           # 异丁烷
    #     "CCO",              # 乙醇
    # ]
    # batch_results = batch_smiles_to_3d(smiles_list)


    # 3. 试验从Excel读取SMILES并批量转换, 保存3D结构文件
    # print("try smiles to 3d structure from excel...")

    # 3.1 邻甲基苯醛(o-tolualdehyde, 2-MBA)
    # file_path = './邻甲基苯醛.xlsx'   # 读取Excel文件
    # df = pd.read_excel(file_path)
    # # 打印数据
    # print(df.head())
    # print(df.columns)
    # print(df['Smiles'].iloc[0])  # 打印第一行的SMILES值
    # print(df['Name'].iloc[0])  # 打印第一行的Name值
    # smiles_list = df['Smiles'].tolist()
    # names_list = df['Name'].tolist()
    # mol_save_dir = '2-MBA_structures'
    # batch_results = batch_smiles_to_3d(smiles_list, names=names_list, output_dir=mol_save_dir)

    # 3.2 3-氧代-3-苯基丙腈(3-Oxo-3-phenylpropanenitrile, BPN)
    # file_path = './3-氧代-3-苯基丙腈.xlsx'
    # df = pd.read_excel(file_path)
    # smiles_list = df['Smiles'].tolist()
    # names_list = df['Name'].tolist()
    # mol_save_dir = 'BPN_structures'
    # batch_results = batch_smiles_to_3d(smiles_list, names=names_list, output_dir=mol_save_dir)
    
    print("3d structure are saves in respective folders: ./2-MBA_structures and ./BPN_structures")
