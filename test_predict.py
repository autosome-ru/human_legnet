import torch

import pandas as pd
import lightning.pytorch as pl

from lib.human_legnet.datamodule import SeqDataModule
from pathlib import Path


def save_predict(trainer: pl.Trainer,
                 model: pl.LightningModule, 
                 data: SeqDataModule,
                 save_dir: Path,
                 pref: str = ""):
    
    df = data.test.copy()
    
    for pred_name, dl in data.dls_for_predictions():
        
        y_preds =  trainer.predict(model,
                                   dataloaders=dl)
        y_preds = torch.concat(y_preds).cpu().numpy() #type: ignore
        df[pred_name] = y_preds

    if pref != "":
        df.to_csv(save_dir / f"predictions_{pref}.tsv", 
                  sep='\t', 
                  index=False)
    else:
        df.to_csv(save_dir / f"predictions.tsv", 
                  sep='\t', 
                  index=False)
    return df
