import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from captum.attr import IntegratedGradients
import numpy as np
from tqdm import tqdm
import warnings
import os
from argparse import Namespace
from collections import OrderedDict
from sklearn.metrics import accuracy_score, f1_score
import pytorch_lightning as pl

# --- 导入您的模型和 Dataset ---
from models.Tip_utils.Tip_downstream import TIPBackbone
from models.Tip_utils.Tip_downstream_ensemble import TIPBackboneEnsemble
from datasets.ImagingAndTabularDataset import ImagingAndTabularDataset
from models.Tip_utils.pieces import DotDict
from models.Evaluator import Evaluator
from models.TabularModel import TabularModel
from models.ImagingModel import ImagingModel
from models.MultimodalModel import MultimodalModel
from models.Tip_utils.Tip_downstream import TIPBackbone
from models.Tip_utils.Tip_downstream_ensemble import TIPBackboneEnsemble
from models.DAFT import DAFT
from models.MultimodalModelMUL import MultimodalModelMUL
from models.MultimodalModelTransformer import MultimodalModelTransformer


# =========================================================================
#
# PART 2: 模型适配器 (Model Adapter)
# (这是解决 API 不同问题的关键)
#
# =========================================================================

class ModelAdapter:
    """
    包装一个模型 (TIP, TIP-Ensemble, DAFT, CONCAT/MAX) 
    以提供一个统一的 API 供我们的分析函数使用。
    """
    def __init__(self, model: nn.Module, hparams: Namespace):
        self.model = model
        self.hparams = hparams
        self.model_type = None

        if hparams.eval_datatype == 'multimodal':
            # --- [新增] 处理 Ensemble 的逻辑 ---
            if hparams.finetune_ensemble:
                self.model_type = "TIP_Ensemble"
                assert isinstance(model, TIPBackboneEnsemble)
                self.image_encoder = model.encoder_imaging
                self.tabular_encoder = model.encoder_tabular
                self.fusion_classifier_func = self._tip_ensemble_fusion_classifier
            else:
                self.model_type = "TIP_Standard"
                assert isinstance(model, TIPBackbone)
                self.image_encoder = model.encoder_imaging
                self.tabular_encoder = model.encoder_tabular
                self.fusion_classifier_func = self._tip_fusion_classifier
            
        elif hparams.eval_datatype == 'imaging_and_tabular':
            if hparams.algorithm_name == 'DAFT':
                self.model_type = "DAFT"
                assert isinstance(model, DAFT)
                self.image_encoder = model.imaging_encoder
                self.tabular_encoder = model.tabular_encoder
                self.fusion_classifier_func = self._daft_fusion_classifier
            elif hparams.algorithm_name in ['CONCAT', 'MAX']:
                self.model_type = hparams.algorithm_name
                assert isinstance(model, MultimodalModel)
                self.image_encoder = model.imaging_model.encoder
                self.tabular_encoder = model.tabular_model.encoder
                self.fusion_classifier_func = self._multimodal_fusion_classifier
            else:
                # 排除 MUL (不兼容) 和 MultimodalModelTransformer (您保证不会出现)
                raise ValueError(f"不支持的 algorithm_name: {hparams.algorithm_name}")
        else:
            raise TypeError(f"不支持的 eval_datatype: {hparams.eval_datatype}")
            
        print(f"ModelAdapter 为模型类型初始化: {self.model_type}")

    def get_image_embedding(self, img_input: torch.Tensor) -> torch.Tensor:
        """返回 z_i (图像嵌入)"""
        # TIP 和 TIP_Ensemble 共享相同的逻辑
        if self.model_type in ["TIP_Standard", "TIP_Ensemble"]:
            return self.image_encoder(img_input)[-1]
        elif self.model_type == "DAFT":
            return self.image_encoder(img_input)[-1]
        elif self.model_type in ["CONCAT", "MAX"]:
            return self.image_encoder(img_input)[0].squeeze()
        
    def get_tabular_embedding(self, tab_input: torch.Tensor) -> torch.Tensor:
        """返回 z_t (表格嵌入)"""
        # TIP, TIP_Ensemble, 和 CONCAT/MAX 共享相同的逻辑
        if self.model_type in ["TIP_Standard", "TIP_Ensemble"]:
            return self.tabular_encoder(tab_input)
        elif self.model_type == "DAFT":
            return self.tabular_encoder(tab_input) # DAFT's encoder 是 Identity
        elif self.model_type in ["CONCAT", "MAX"]:
            return self.tabular_encoder(tab_input).squeeze()

    # --- 内部包装器 (用于 IG) ---
    
    def _tip_fusion_classifier(self, img_embed, tab_embed):
        """标准 TIP 模型的融合/分类路径"""
        multimodal_features = self.model.encoder_multimodal(
            x=tab_embed, image_features=img_embed
        )
        return self.model.classifier(multimodal_features[:, 0, :])

    # --- [新增] Ensemble 的融合函数 ---
    def _tip_ensemble_fusion_classifier(self, img_embed, tab_embed):
        """
        Ensemble 模型的融合/分类路径。
       
        """
        # 1. 多模态路径
        x_m = self.model.encoder_multimodal(x=tab_embed, image_features=img_embed)
        out_m = self.model.classifier_multimodal(x_m[:,0,:])
        
        # 2. 纯图像路径
        if self.model.encoder_imaging_type == 'resnet':
            out_i = self.model.classifier_imaging(F.adaptive_avg_pool2d(img_embed, (1, 1)).flatten(1))
        elif self.model.encoder_imaging_type == 'vit':
            out_i = self.model.classifier_imaging(img_embed[:,0,:])
        
        # 3. 纯表格路径
        out_t = self.model.classifier_tabular(tab_embed[:,0,:])
        
        # 4. 平均
        x = (out_i + out_t + out_m) / 3.0
        return x

    def _daft_fusion_classifier(self, img_embed, tab_embed):
        """DAFT 模型的融合/分类路径"""
        x_fused = self.model.daft(x_im=img_embed, x_tab=tab_embed)
        x_fused = self.model.residual(x_fused)
        x_fused = x_fused + self.model.shortcut(img_embed)
        x_fused = self.model.act(x_fused)
        x_pool = F.adaptive_avg_pool2d(x_fused, (1, 1)).flatten(1)
        return self.model.head(x_pool)

    def _multimodal_fusion_classifier(self, img_embed, tab_embed):
        """CONCAT/MAX 模型的融合/分类路径"""
        if self.model_type == 'CONCAT':
            x_fused = torch.cat([img_embed, tab_embed], dim=1)
        elif self.model_type == 'MAX':
            x_im_proj = self.model.imaging_proj(img_embed)
            x_fused = torch.stack([x_im_proj, tab_embed], dim=1)
            x_fused, _ = torch.max(x_fused, dim=1)
        return self.model.head(x_fused)


# =========================================================================
#
# PART 3: 适配器驱动的分析函数
# (这部分代码无需更改，因为 Adapter 已经处理了所有复杂性)
#
# =========================================================================

def compute_mean_embeddings(adapter: ModelAdapter, 
                            data_loader: DataLoader, 
                            device: torch.device,
                            baseline_file_path: str): # <-- 新增参数
    """
    计算数据集的平均嵌入 (z_i, z_t)。
    如果 'baseline_file_path' 存在，则加载它。
    否则，计算它并保存。
    """
    
    # --- [新增] 检查文件是否存在 ---
    if os.path.exists(baseline_file_path):
        print(f"--- 正在从文件加载已保存的基线 ---")
        print(f"路径: {baseline_file_path}")
        try:
            baselines = torch.load(baseline_file_path, map_location=device)
            print("基线加载成功。")
            return baselines['img_embed'], baselines['tab_embed']
        except Exception as e:
            print(f"加载基线失败 (文件可能已损坏): {e}")
            print("将重新计算基线...")
    # ----------------------------

    adapter.model.to(device)
    adapter.model.eval()
    all_img_embeds, all_tab_embeds = [], []
    print(f"Calculating mean embeddings baseline from {len(data_loader)} batches...")
    with torch.no_grad():
        for batch in tqdm(data_loader):
            img_input = batch[0][0].to(device)
            tab_input = batch[0][1].to(device)
            if img_input.dtype == torch.uint8:
                img_input = img_input.float() / 255.0
            
            img_embed = adapter.get_image_embedding(img_input)
            tab_embed = adapter.get_tabular_embedding(tab_input)
            all_img_embeds.append(img_embed.cpu())
            all_tab_embeds.append(tab_embed.cpu())
            
    mean_img_embed = torch.cat(all_img_embeds, dim=0).mean(dim=0)
    mean_tab_embed = torch.cat(all_tab_embeds, dim=0).mean(dim=0)
    print("Mean embeddings calculated.")

    # --- [新增] 保存计算出的基线 ---
    try:
        print(f"正在保存新基线到: {baseline_file_path}")
        baselines_to_save = {
            'img_embed': mean_img_embed.to(device),
            'tab_embed': mean_tab_embed.to(device)
        }
        torch.save(baselines_to_save, baseline_file_path)
        print("基线保存成功。")
    except Exception as e:
        print(f"警告：保存基线失败: {e}")
    # ---------------------------------

    return mean_img_embed.to(device), mean_tab_embed.to(device)

def evaluate_model_with_ablation(adapter: ModelAdapter, 
                                 data_loader: DataLoader, 
                                 device: torch.device,
                                 img_embed_baseline: torch.Tensor,
                                 tab_embed_baseline: torch.Tensor,
                                 ablate_image: bool = False,
                                 ablate_tabular: bool = False):
    """使用模态消融评估模型性能。"""
    adapter.model.to(device)
    adapter.model.eval()
    all_labels, all_preds = [], []
    
    with torch.no_grad():
        for batch in tqdm(data_loader):
            img_input = batch[0][0].to(device)
            tab_input = batch[0][1].to(device)
            labels = batch[1].to(device)
            
            if img_input.dtype == torch.uint8:
                img_input = img_input.float() / 255.0

            img_embed = adapter.get_image_embedding(img_input)
            tab_embed = adapter.get_tabular_embedding(tab_input)
            
            if ablate_image:
                img_embed = img_embed_baseline.expand_as(img_embed)
            if ablate_tabular:
                if adapter.model_type == "DAFT":
                    tab_embed = torch.zeros_like(tab_embed)
                else:
                    tab_embed = tab_embed_baseline.expand_as(tab_embed)

            logits = adapter.fusion_classifier_func(img_embed, tab_embed)
            preds = torch.argmax(logits, dim=1)
            all_labels.append(labels.cpu())
            all_preds.append(preds.cpu())

    all_labels = torch.cat(all_labels).numpy()
    all_preds = torch.cat(all_preds).numpy()
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return {'accuracy': accuracy, 'f1_macro': f1_macro}

def run_ablation_analysis(adapter: ModelAdapter, 
                          data_loader: DataLoader, 
                          device: torch.device,
                          img_embed_baseline: torch.Tensor,
                          tab_embed_baseline: torch.Tensor):
    """运行完整的消融研究。"""
    print("\n--- [ANALYSIS 1/2] Running Modality Ablation ---")
    
    print("Evaluating: Full Model (Baseline)")
    metrics_full = evaluate_model_with_ablation(
        adapter, data_loader, device, img_embed_baseline, tab_embed_baseline
    )
    print("Evaluating: Image-Only (Tabular Ablated)")
    metrics_img_only = evaluate_model_with_ablation(
        adapter, data_loader, device, img_embed_baseline, tab_embed_baseline, ablate_tabular=True
    )
    print("Evaluating: Tabular-Only (Image Ablated)")
    metrics_tab_only = evaluate_model_with_ablation(
        adapter, data_loader, device, img_embed_baseline, tab_embed_baseline, ablate_image=True
    )

    base_f1 = metrics_full['f1_macro']
    f1_loss_tab_ablated = base_f1 - metrics_img_only['f1_macro']
    f1_loss_img_ablated = base_f1 - metrics_tab_only['f1_macro']
    total_loss = f1_loss_tab_ablated + f1_loss_img_ablated + 1e-10
    img_contrib_pct = (f1_loss_img_ablated / total_loss) * 100.0
    tab_contrib_pct = (f1_loss_tab_ablated / total_loss) * 100.0

    print("\n--- Ablation Analysis Results (Global Contribution) ---")
    print(f"{'Experiment':<20} | {'Accuracy':<10} | {'Macro F1':<10} | {'Δ (F1)':<10}")
    print("-" * 56)
    print(f"{'Full Model':<20} | {metrics_full['accuracy']:<10.4f} | {metrics_full['f1_macro']:<10.4f} | {'-':<10}")
    print(f"{'Image-Only':<20} | {metrics_img_only['accuracy']:<10.4f} | {metrics_img_only['f1_macro']:<10.4f} | {-f1_loss_tab_ablated:<10.4f}")
    print(f"{'Tabular-Only':<20} | {metrics_tab_only['accuracy']:<10.4f} | {metrics_tab_only['f1_macro']:<10.4f} | {-f1_loss_img_ablated:<10.4f}")
    print("-" * 56)
    print(f"Global Contribution (F1-based):")
    print(f"  > Image Modality: {img_contrib_pct:.2f}%")
    print(f"  > Tabular Modality: {tab_contrib_pct:.2f}%")
    print("-" * 56)

def calculate_attribution_magnitude(attributions_tensor):
    return torch.sum(torch.abs(attributions_tensor))

def calculate_relative_contribution(img_attr_score, tab_attr_score, epsilon=1e-10):
    total_attribution = img_attr_score + tab_attr_score + epsilon
    return (img_attr_score / total_attribution) * 100.0, (tab_attr_score / total_attribution) * 100.0

def get_ig_contribution(adapter: ModelAdapter, 
                        img_sample: torch.Tensor, 
                        tab_sample: torch.Tensor, 
                        img_embed_baseline: torch.Tensor, 
                        tab_embed_baseline: torch.Tensor, 
                        target_class: int,
                        m_steps=100):
    """计算单个样本的 IG。"""
    ig = IntegratedGradients(adapter.fusion_classifier_func)
    adapter.model.eval()
    with torch.no_grad():
        if img_sample.dtype == torch.uint8:
            img_sample = img_sample.float() / 255.0
        img_embed_sample = adapter.get_image_embedding(img_sample)
        tab_embed_sample = adapter.get_tabular_embedding(tab_sample)

    inputs_tuple = (img_embed_sample, tab_embed_sample)
    
    if adapter.model_type == "DAFT":
        tab_baseline_final = torch.zeros_like(tab_embed_sample)
    else:
        tab_baseline_final = tab_embed_baseline
    baselines_tuple = (img_embed_baseline, tab_baseline_final)

    attributions_tuple = ig.attribute(
        inputs=inputs_tuple, baselines=baselines_tuple,
        target=target_class, n_steps=m_steps,
        return_convergence_delta=False
    )
    img_attributions = attributions_tuple[0].squeeze(0)
    tab_attributions = attributions_tuple[1].squeeze(0)
    img_score_l1 = calculate_attribution_magnitude(img_attributions)
    tab_score_l1 = calculate_attribution_magnitude(tab_attributions)
    return calculate_relative_contribution(img_score_l1, tab_score_l1)

def run_ig_analysis_on_dataset(adapter: ModelAdapter, 
                               data_loader: DataLoader, 
                               device: torch.device,
                               img_embed_baseline: torch.Tensor,
                               tab_embed_baseline: torch.Tensor):
    """运行完整的 IG 分析。"""
    print("\n--- [ANALYSIS 2/2] Running Integrated Gradients ---")
    adapter.model.to(device)
    adapter.model.eval()
    all_results = []
    
    if data_loader.batch_size != 1:
        warnings.warn("IG analysis requires batch_size=1. Skipping.")
        return

    print(f"Calculating IG contributions for {len(data_loader)} samples...")
    for batch in tqdm(data_loader):
        img_sample = batch[0][0].to(device)
        tab_sample = batch[0][1].to(device)
        target_class = batch[1].item()
        try:
            img_pct, tab_pct = get_ig_contribution(
                adapter, img_sample, tab_sample,
                img_embed_baseline, tab_embed_baseline,
                target_class, m_steps=100
            )
            all_results.append((img_pct, tab_pct))
        except Exception as e:
            all_results.append((np.nan, np.nan))

    print("\n--- IG Analysis Results (Average Local Contribution) ---")
    results_array = np.array(all_results)
    mean_img_contrib = np.nanmean(results_array[:, 0])
    mean_tab_contrib = np.nanmean(results_array[:, 1])
    std_img = np.nanstd(results_array[:, 0])
    n_samples = results_array.shape[0]
    ci_img = 1.96 * (std_img / np.sqrt(n_samples))
    ci_tab = 1.96 * (np.nanstd(results_array[:, 1]) / np.sqrt(n_samples))

    print(f"Total samples: {n_samples}")
    print(f"  > Image Modality: {mean_img_contrib:.2f}% (± {ci_img:.2f} 95% CI)")
    print(f"  > Tabular Modality: {mean_tab_contrib:.2f}% (± {ci_tab:.2f} 95% CI)")
    print("-" * 56)


# =========================================================================
#
# PART 4: 主执行脚本
# (已更新为使用 Evaluator.load_from_checkpoint)
#
# =========================================================================

def grab_arg_from_checkpoint(args: Namespace, arg_name: str):
    """
   
    """
    if args.checkpoint:
        try:
            ckpt = torch.load(args.checkpoint) 
            load_args = ckpt['hyper_parameters']
        except FileNotFoundError:
            load_args = args
    else:
        load_args = args
    if isinstance(load_args, Namespace):
        return getattr(load_args, arg_name, None) # 使用 .get(..., None) 更安全
    else:
        return load_args.get(arg_name, None)

def load_analysis_datasets(cfg: Namespace):
    """
    使用您更正后的 'test_dataset' 逻辑加载 DataLoaders。
   
    """
    
    # 共同参数
    shared_args = {
        'delete_segmentation': cfg.delete_segmentation,
        'eval_one_hot': False, 'train': False, 'target': cfg.target,
        'corruption_rate': 0.0, 'data_base': cfg.data_base,
        'missing_tabular': False, 'missing_strategy': 'None',
        'missing_rate': 0.0,
        'augmentation_speedup': cfg.augmentation_speedup,
        'algorithm_name': cfg.algorithm_name # (来自 ImagingAndTabularDataset 的 __init__)
    }
    
    # --- 1. 创建训练数据集 (用于基线计算) ---
    if cfg.eval_datatype == 'multimodal':
        img_size_train = grab_arg_from_checkpoint(cfg, 'img_size')
        aug_rate_train = 0.0
    else: # imaging_and_tabular
        img_size_train = cfg.img_size
        aug_rate_train = 0.0
        
    train_dataset = ImagingAndTabularDataset(
        data_path_imaging=cfg.data_train_eval_imaging,
        eval_train_augment_rate=aug_rate_train, 
        data_path_tabular=cfg.data_train_eval_tabular,
        field_lengths_tabular=cfg.field_lengths_tabular,
        labels_path=cfg.labels_train_eval_imaging, 
        img_size=img_size_train, 
        live_loading=cfg.live_loading,
        **shared_args
    )
    
    # --- 2. 创建测试数据集 (用于分析) ---
    if cfg.eval_datatype == 'multimodal':
        img_size_test = grab_arg_from_checkpoint(cfg, 'img_size')
        aug_rate_test = 0.0
    else: # imaging_and_tabular
        img_size_test = cfg.img_size
        aug_rate_test = 0.0
        
    test_dataset = ImagingAndTabularDataset(
        data_path_imaging=cfg.data_test_eval_imaging,
        eval_train_augment_rate=aug_rate_test, 
        data_path_tabular=cfg.data_test_eval_tabular,
        field_lengths_tabular=cfg.field_lengths_tabular,
        labels_path=cfg.labels_test_eval_imaging, 
        img_size=img_size_test,
        live_loading=cfg.live_loading,
        **shared_args
    )
    
    # --- 3. 创建 DataLoaders ---
    train_loader = DataLoader(
        train_dataset, batch_size=256, shuffle=False, num_workers=cfg.num_workers
    )
    test_loader_ablation = DataLoader(
        test_dataset, batch_size=256, shuffle=False, num_workers=cfg.num_workers
    )
    test_loader_ig = DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=cfg.num_workers
    )
    print("DataLoaders prepared.")
    return train_loader, test_loader_ablation, test_loader_ig

# ----------------------------------------------------
# 6. 如何使用 (请替换为您自己的路径和 Dataloader)
# ----------------------------------------------------

def load_and_run_from_checkpoint(checkpoints:str):
    """
    加载所有配置、实例化对象并运行分析。
    """
    
    # --- 1. 加载配置 ---
    checkpoints_path = checkpoints
    DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    print(f"Loading checkpoint from: {checkpoints_path}")
    checkpoint = torch.load(checkpoints_path, map_location=DEVICE)
    
    hparams = Namespace(**checkpoint['hyper_parameters'])
    state_dict = checkpoint['state_dict']
    
    # --- 2. 应用修复补丁 (Hacks) ---
    # hparams.checkpoint = None      # 避免 FileNotFoundError
    # hparams.share_weights = False  # 避免 AttributeError
    hparams.missing_tabular = False
    hparams.eval_one_hot = False

    # --- 3. 实例化 Evaluator (并自动实例化模型) ---
    # 我们使用 'hparams' 手动实例化 Evaluator，
    # 而不是使用 .load_from_checkpoint
    print("Instantiating Evaluator (this will load the model)...")
    eval_module = Evaluator(hparams).to(DEVICE)
    
    # --- 4. 手动加载 State Dict ---
    # (因为我们绕过了 .load_from_checkpoint)
    print("Loading state_dict into Evaluator...")
    eval_module.load_state_dict(state_dict)
    eval_module.eval()
    print("Model loaded successfully.")

    # --- 5. 获取模型并创建适配器 ---
    model_to_analyze = eval_module.model
    adapter = ModelAdapter(model_to_analyze, hparams)

    # --- 6. 加载 Data ---
    train_loader, test_loader_ablation, test_loader_ig = load_analysis_datasets(hparams)

    # --- 7. 运行完整分析 ---
    print("\n--- Computing Shared Baselines ---")
    model_id_name = hparams.algorithm_name if hparams.eval_datatype == 'imaging_and_tabular' else 'TIP'
    dataset_name = hparams.target
    BASELINE_FILE_PATH = f"./{dataset_name}_{model_id_name}_baselines.pt"
    # --------------------------

    print("\n--- Computing Shared Baselines ---")
    img_baseline, tab_baseline = compute_mean_embeddings(
        adapter, 
        train_loader, 
        DEVICE,
        BASELINE_FILE_PATH  # <-- 传入文件路径
    )
    
    run_ablation_analysis(
        adapter, test_loader_ablation, DEVICE,
        img_baseline, tab_baseline
    )
    
    run_ig_analysis_on_dataset(
        adapter, test_loader_ig, DEVICE,
        img_baseline, tab_baseline
    )
    
    print("\n\n--- Full Analysis Complete ---")

if __name__ == "__main__":
    
    
    print("\n--- 脚本已准备就绪 ---")

    checkpoints_path = [
        # "/mnt/hdd/jiazy/checkpoints/dvm/Concat/checkpoint_best_acc.ckpt",
        # '/mnt/hdd/jiazy/checkpoints/dvm/DAFT/checkpoint_best_acc.ckpt',
        # '/mnt/hdd/jiazy/checkpoints/dvm/Max/checkpoint_best_acc.ckpt'
        '/mnt/hdd/jiazy/checkpoints/dvm/TIP/checkpoint_best_acc.ckpt'
    ]
    for checkpoint in checkpoints_path:
        load_and_run_from_checkpoint(checkpoint)