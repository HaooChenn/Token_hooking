#!/usr/bin/env python3
"""
VAR Token ID 提取器
功能：将图像压缩为前两个尺度的离散token ID序列（5个数字）并保存为.pt文件

重要说明：
- 第一尺度：1×1 = 1个token ID
- 第二尺度：2×2 = 4个token IDs  
- 总计：5个token IDs，比如 [1247, 892, 3456, 127, 2890]
- 每个ID都是0-4095之间的整数（codebook索引）
"""

import os
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from torchvision import transforms
from typing import List, Union, Optional
import argparse
from pathlib import Path

# 添加模型路径到系统路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入VAR模型组件
from models.vqvae import VQVAE
from models.quant import VectorQuantizer2


class VARTokenExtractor:
    """VAR Token ID 提取器
    
    从图像提取前两个尺度的离散token ID序列，完全复现VAR训练时的GT生成逻辑
    """
    
    def __init__(self, vae_ckpt_path: str, device: str = 'cuda'):
        """
        初始化Token提取器
        
        Args:
            vae_ckpt_path: VAE模型权重路径
            device: 计算设备 ('cuda' 或 'cpu')
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 初始化VQVAE模型（与训练配置完全一致）
        self.vae = VQVAE(
            vocab_size=4096,        # 词汇表大小：token ID范围 [0, 4095]
            z_channels=32,          # 特征通道数
            ch=160,                 # 基础通道数
            dropout=0.0,
            beta=0.25,
            using_znorm=False,
            quant_conv_ks=3,
            quant_resi=0.5,
            share_quant_resi=4,
            default_qresi_counts=0,
            v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),  # 完整尺度序列
            test_mode=True          # 测试模式
        ).to(self.device)
        
        # 加载预训练权重
        print(f"正在加载VAE权重: {vae_ckpt_path}")
        if not os.path.exists(vae_ckpt_path):
            raise FileNotFoundError(f"VAE权重文件不存在: {vae_ckpt_path}")
        
        checkpoint = torch.load(vae_ckpt_path, map_location=self.device)
        self.vae.load_state_dict(checkpoint, strict=True)
        self.vae.eval()
        print("VAE模型加载完成!")
        
        # 图像预处理管道（与VAR训练时完全一致）
        self.transform = transforms.Compose([
            transforms.Resize(288, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(256),      
            transforms.ToTensor(),           
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
        ])
    
    def load_and_preprocess_image(self, image_path: str) -> torch.Tensor:
        """
        加载并预处理单张图像
        
        Args:
            image_path: 图像文件路径
            
        Returns:
            处理后的图像tensor，形状为 [1, 3, 256, 256]
        """
        try:
            with Image.open(image_path) as img:
                img = img.convert('RGB')
                
            img_tensor = self.transform(img)
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            
            return img_tensor
            
        except Exception as e:
            raise ValueError(f"无法加载图像 {image_path}: {str(e)}")
    
    def extract_first_two_scales_token_ids(self, img_tensor: torch.Tensor) -> List[int]:
        """
        提取前两个尺度的token ID序列
        
        这个方法直接调用VAR训练时使用的相同方法，确保完全一致的GT生成
        
        Args:
            img_tensor: 预处理后的图像tensor [1, 3, 256, 256]
            
        Returns:
            5个token ID的列表，比如 [1247, 892, 3456, 127, 2890]
        """
        with torch.no_grad():
            # 方法一：直接使用VAR训练时的方法（推荐）
            # 这确保了与训练时完全相同的GT生成逻辑
            gt_idx_Bl = self.vae.img_to_idxBl(
                img_tensor, 
                v_patch_nums=[(1, 1), (2, 2)]  # 只要前两个尺度
            )
            
            # gt_idx_Bl是一个列表，包含两个tensor：
            # gt_idx_Bl[0]: 第一尺度的token IDs，shape [1, 1]   (1个token)
            # gt_idx_Bl[1]: 第二尺度的token IDs，shape [1, 4]   (4个token)
            
            # 将所有token ID拼接成一个扁平列表
            all_token_ids = []
            
            for scale_idx, token_tensor in enumerate(gt_idx_Bl):
                # token_tensor shape: [1, num_tokens_in_this_scale]
                token_list = token_tensor.squeeze(0).cpu().numpy().tolist()
                all_token_ids.extend(token_list)
                
                print(f"第{scale_idx+1}尺度 ({['1×1', '2×2'][scale_idx]}) 的token IDs: {token_list}")
            
            return all_token_ids
    
    def process_single_image(self, image_path: str, output_dir: str) -> str:
        """
        处理单张图像并保存token ID序列
        
        Args:
            image_path: 输入图像路径
            output_dir: 输出目录
            
        Returns:
            输出文件路径
        """
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"正在处理图像: {image_path}")
        
        # 1. 加载并预处理图像
        img_tensor = self.load_and_preprocess_image(image_path)
        
        # 2. 提取前两个尺度的token IDs
        token_ids = self.extract_first_two_scales_token_ids(img_tensor)
        
        # 3. 验证结果
        assert len(token_ids) == 5, f"应该有5个token，但得到了{len(token_ids)}个"
        assert all(0 <= tid < 4096 for tid in token_ids), "所有token ID应该在[0, 4095]范围内"
        
        print(f"提取到的5个token IDs: {token_ids}")
        print(f"Token ID范围检查: 最小值={min(token_ids)}, 最大值={max(token_ids)}")
        
        # 4. 生成输出文件名并保存
        image_name = Path(image_path).stem
        output_path = os.path.join(output_dir, f"{image_name}_tokens.pt")
        
        # 保存为tensor格式，方便后续使用
        token_tensor = torch.tensor(token_ids, dtype=torch.long)
        torch.save(token_tensor, output_path)
        
        print(f"Token序列已保存至: {output_path}")
        print(f"保存的数据类型: {token_tensor.dtype}, 形状: {token_tensor.shape}")
        
        return output_path
    
    def process_image_list(self, image_paths: List[str], output_dir: str) -> List[str]:
        """
        批量处理多张图像
        
        Args:
            image_paths: 图像路径列表
            output_dir: 输出目录
            
        Returns:
            输出文件路径列表
        """
        output_paths = []
        
        print(f"开始批量处理 {len(image_paths)} 张图像...")
        
        for i, image_path in enumerate(image_paths):
            try:
                print(f"\n{'='*60}")
                print(f"正在处理 [{i+1}/{len(image_paths)}]: {os.path.basename(image_path)}")
                print(f"{'='*60}")
                
                output_path = self.process_single_image(image_path, output_dir)
                output_paths.append(output_path)
                
            except Exception as e:
                print(f"❌ 处理图像失败 {image_path}: {str(e)}")
                continue
        
        print(f"\n{'='*60}")
        print(f"✅ 批量处理完成! 成功处理 {len(output_paths)}/{len(image_paths)} 张图像")
        print(f"{'='*60}")
        
        return output_paths
    
    def demo_load_and_verify(self, pt_file_path: str):
        """
        演示如何加载和验证保存的token序列
        
        Args:
            pt_file_path: 保存的.pt文件路径
        """
        print(f"\n演示：加载和验证 {pt_file_path}")
        
        # 加载token序列
        loaded_tokens = torch.load(pt_file_path, map_location='cpu')
        
        print(f"加载的token序列: {loaded_tokens.tolist()}")
        print(f"数据类型: {loaded_tokens.dtype}")
        print(f"形状: {loaded_tokens.shape}")
        print(f"值范围: [{loaded_tokens.min().item()}, {loaded_tokens.max().item()}]")
        
        # 验证这确实是5个token ID
        assert loaded_tokens.shape == (5,), f"期望形状(5,)，实际{loaded_tokens.shape}"
        assert loaded_tokens.dtype == torch.long, f"期望int64，实际{loaded_tokens.dtype}"
        
        print("✅ 验证通过：这是一个包含5个token ID的序列！")


def main():
    """主函数：命令行接口"""
    parser = argparse.ArgumentParser(
        description="VAR Token ID 提取器 - 将图像转换为前两个尺度的token ID序列",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  # 处理单张图像
  python var_token_extractor.py --vae_ckpt vae_ch160v4096z32.pth --image_path ./test.jpg --output_dir ./tokens
  
  # 批量处理目录
  python var_token_extractor.py --vae_ckpt vae_ch160v4096z32.pth --input_dir ./images --output_dir ./tokens
  
输出说明:
  每个图像会生成一个 *_tokens.pt 文件，包含5个token ID
  比如: [1247, 892, 3456, 127, 2890]
  其中第1个是第一尺度(1×1)，后4个是第二尺度(2×2)
        """
    )
    
    parser.add_argument('--vae_ckpt', required=True, 
                        help='VAE模型权重文件路径')
    parser.add_argument('--image_path', help='单个图像文件路径')
    parser.add_argument('--input_dir', help='输入图像目录路径')
    parser.add_argument('--output_dir', required=True, help='输出目录路径')
    parser.add_argument('--device', default='cuda', choices=['cuda', 'cpu'],
                        help='计算设备')
    parser.add_argument('--extensions', default='jpg,jpeg,png,bmp,tiff',
                        help='支持的图像文件扩展名')
    parser.add_argument('--demo_verify', action='store_true',
                        help='处理完成后演示如何加载和验证结果')
    
    args = parser.parse_args()
    
    # 参数验证
    if not args.image_path and not args.input_dir:
        parser.error("必须指定 --image_path 或 --input_dir 之一")
    
    if args.image_path and args.input_dir:
        parser.error("不能同时指定 --image_path 和 --input_dir")
    
    # 初始化提取器
    try:
        extractor = VARTokenExtractor(args.vae_ckpt, args.device)
    except Exception as e:
        print(f"❌ 初始化提取器失败: {str(e)}")
        return
    
    # 准备图像路径列表
    image_paths = []
    
    if args.image_path:
        if os.path.exists(args.image_path):
            image_paths = [args.image_path]
        else:
            print(f"❌ 图像文件不存在: {args.image_path}")
            return
    else:
        extensions = args.extensions.lower().split(',')
        input_dir = Path(args.input_dir)
        
        if not input_dir.exists():
            print(f"❌ 输入目录不存在: {args.input_dir}")
            return
        
        for ext in extensions:
            image_paths.extend(input_dir.glob(f"*.{ext}"))
            image_paths.extend(input_dir.glob(f"*.{ext.upper()}"))
        
        image_paths = [str(p) for p in image_paths]
        
        if not image_paths:
            print(f"❌ 在目录 {args.input_dir} 中未找到支持的图像文件")
            return
    
    # 处理图像
    if len(image_paths) == 1:
        output_path = extractor.process_single_image(image_paths[0], args.output_dir)
        output_paths = [output_path]
    else:
        output_paths = extractor.process_image_list(image_paths, args.output_dir)
    
    # 可选：演示验证
    if args.demo_verify and output_paths:
        extractor.demo_load_and_verify(output_paths[0])


if __name__ == "__main__":
    main()
