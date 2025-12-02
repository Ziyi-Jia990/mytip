from typing import Tuple

import torch
import torchmetrics
import pytorch_lightning as pl
# 引入 R2Score 和 MeanSquaredError
from torchmetrics import R2Score, MeanSquaredError

from models.TabularModel import TabularModel
from models.ImagingModel import ImagingModel
from models.MultimodalModel import MultimodalModel


class Evaluator_Regression(pl.LightningModule):
  def __init__(self, hparams):
    super().__init__()
    self.save_hyperparameters(hparams)

    # 模型初始化保持不变
    if self.hparams.datatype == 'imaging' or self.hparams.datatype == 'multimodal':
      self.model = ImagingModel(self.hparams)
    if self.hparams.datatype == 'tabular':
      self.model = TabularModel(self.hparams)
    if self.hparams.datatype == 'imaging_and_tabular':
      self.model = MultimodalModel(self.hparams)
    
    # 损失函数
    self.criterion = torch.nn.MSELoss()

    # --- [修改点 1] 初始化指标: MAE, RMSE, R2 ---
    # 1. MAE (原代码已有)
    self.mae_train = torchmetrics.MeanAbsoluteError()
    self.mae_val = torchmetrics.MeanAbsoluteError()
    self.mae_test = torchmetrics.MeanAbsoluteError()

    # 2. RMSE (设置 squared=False 即为 RMSE)
    self.rmse_train = MeanSquaredError(squared=False)
    self.rmse_val = MeanSquaredError(squared=False)
    self.rmse_test = MeanSquaredError(squared=False)

    # 3. R2 Score
    self.r2_train = R2Score()
    self.r2_val = R2Score()
    self.r2_test = R2Score()

    # 保留原有的 PCC (如果不想要可以删除)
    self.pcc_train = torchmetrics.PearsonCorrCoef(num_outputs=1) # 通常回归输出为1
    self.pcc_val = torchmetrics.PearsonCorrCoef(num_outputs=1)
    self.pcc_test = torchmetrics.PearsonCorrCoef(num_outputs=1)
    
    # 修改用于 ModelCheckpoint 的监控指标，建议使用 val_rmse 或 val_r2
    self.best_val_score = -float('inf') # R2 越大越好，MAE/RMSE 越小越好，这里假设监控 R2

    print(self.model)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    y_hat = self.model(x)
    return y_hat

  def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> torch.Tensor:
    x, y = batch
    y_hat = self.forward(x)
    
    # 确保维度匹配，防止广播错误
    if y_hat.shape != y.shape:
        y = y.view_as(y_hat)

    loss = self.criterion(y_hat, y)
    y_hat = y_hat.detach()

    # 更新指标
    self.mae_train(y_hat, y)
    self.rmse_train(y_hat, y)
    self.r2_train(y_hat, y)
    self.pcc_train(y_hat, y)

    # Logging
    self.log('eval.train.loss', loss, on_epoch=True, on_step=False)
    self.log('eval.train.mae', self.mae_train, on_epoch=True, on_step=False)
    self.log('eval.train.rmse', self.rmse_train, on_epoch=True, on_step=False)
    self.log('eval.train.r2', self.r2_train, on_epoch=True, on_step=False)

    return loss
  
  def training_epoch_end(self, _) -> None:
    # 只需要重置，logging 在 training_step 设置 on_epoch=True 已经被自动处理了
    # 如果你需要手动处理 PCC mean:
    epoch_pcc_train = self.pcc_train.compute()
    self.log('eval.train.pcc.mean', epoch_pcc_train, on_epoch=True, on_step=False)
    
    self.mae_train.reset()
    self.rmse_train.reset()
    self.r2_train.reset()
    self.pcc_train.reset()

  def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> torch.Tensor:
    x, y = batch
    y_hat = self.forward(x)
    
    if y_hat.shape != y.shape:
        y = y.view_as(y_hat)

    loss = self.criterion(y_hat, y)
    y_hat = y_hat.detach()

    # 更新验证集指标
    self.mae_val(y_hat, y)
    self.rmse_val(y_hat, y)
    self.r2_val(y_hat, y)
    self.pcc_val(y_hat, y)
    
    self.log('eval.val.loss', loss, on_epoch=True, on_step=False)

  def validation_epoch_end(self, _) -> None:
    if self.trainer.sanity_checking:
      return  

    # 计算指标
    epoch_mae_val = self.mae_val.compute()
    epoch_rmse_val = self.rmse_val.compute()
    epoch_r2_val = self.r2_val.compute()
    epoch_pcc_val = self.pcc_val.compute()

    # Log
    self.log('eval.val.mae', epoch_mae_val, on_epoch=True, on_step=False)
    self.log('eval.val.rmse', epoch_rmse_val, on_epoch=True, on_step=False)
    self.log('eval.val.r2', epoch_r2_val, on_epoch=True, on_step=False)
    self.log('eval.val.pcc.mean', epoch_pcc_val, on_epoch=True, on_step=False)
    
    # 更新 Best Score (这里以 R2 为例，R2 越大越好)
    self.best_val_score = max(self.best_val_score, epoch_r2_val)
    
    # 重置
    self.mae_val.reset()
    self.rmse_val.reset()
    self.r2_val.reset()
    self.pcc_val.reset()

  def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _) -> None:
    x, y = batch 
    y_hat = self.forward(x)
    
    if y_hat.shape != y.shape:
        y = y.view_as(y_hat)
        
    y_hat = y_hat.detach()

    # 更新测试集指标
    self.mae_test(y_hat, y)
    self.rmse_test(y_hat, y)
    self.r2_test(y_hat, y)
    self.pcc_test(y_hat, y)

  def test_epoch_end(self, _) -> None:
    test_mae = self.mae_test.compute()
    test_rmse = self.rmse_test.compute()
    test_r2 = self.r2_test.compute()
    test_pcc = self.pcc_test.compute()

    self.log('test.mae', test_mae)
    self.log('test.rmse', test_rmse)
    self.log('test.r2', test_r2)
    self.log('test.pcc.mean', test_pcc)
    
    self.mae_test.reset()
    self.rmse_test.reset()
    self.r2_test.reset()
    self.pcc_test.reset()

  def configure_optimizers(self):
    optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.lr_eval, weight_decay=self.hparams.weight_decay_eval)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=int(10/self.hparams.check_val_every_n_epoch), min_lr=self.hparams.lr*0.0001)
    
    return {
       "optimizer": optimizer, 
       "lr_scheduler": {
         "scheduler": scheduler,
         "monitor": 'eval.val.loss', # 或者 'eval.val.rmse'
         "strict": False
       }
     }