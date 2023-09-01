import torch 

import lightning.pytorch as pl



from datamodule import SeqDataModule
from test_predict import save_predict
from trainer import LitModel, TrainingConfig
from utils import set_global_seed
from lightning.pytorch.callbacks import ModelCheckpoint
from pathlib import Path 
import glob



import argparse 
parser = argparse.ArgumentParser()

general = parser.add_argument_group('general args', 
                                    'general_argumens')
general.add_argument("--model_dir",
                     type=str,
                     required=True)
general.add_argument("--data_path", 
                     type=str, 
                     required=True)
general.add_argument("--outdir",
                     type=str,
                     required=True)
general.add_argument("--device", 
                     type=int,
                     default=0)
general.add_argument("--num_workers",
                     type=int, 
                     default=8)
general.add_argument("--fraction",
                     type=float,
                    default=1.0)




args = parser.parse_args()


model_dir = Path(args.model_dir)
train_cfg = TrainingConfig.from_json(model_dir / "config.json")
train_cfg.data_path = args.data_path

outdir = Path(args.outdir)
outdir.mkdir(parents=True, exist_ok=True)


torch.set_float32_matmul_precision('medium') # type: ignore 

for test_fold in range(1, 11):
    for val_fold in range(1, 11):
        if test_fold == val_fold:
            continue

        data = SeqDataModule(val_fold=val_fold,
                             test_fold=test_fold,
                             cfg=train_cfg)
        
    
        dump_dir = model_dir / f"model_{val_fold}_{test_fold}"
      

        trainer = pl.Trainer(accelerator='gpu',
                            devices=[train_cfg.device], 
                            precision='16-mixed')

        

        models = glob.glob(str(dump_dir / "lightning_logs" / "version_0" / "checkpoints" / "pearson*") )
        
        save_dir = outdir / f"model_{val_fold}_{test_fold}"
        save_dir.mkdir(parents=True, exist_ok=True)
        assert len(models) == 1
        model_path = models[0]
        
        model = LitModel.load_from_checkpoint(model_path, 
                                              tr_cfg=train_cfg)
        
        df_pred = save_predict(trainer, 
                               model, 
                               data,
                               save_dir=save_dir, 
                               pref="new_format")
