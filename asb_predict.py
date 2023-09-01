import argparse 
from asb import ASBDataset
from trainer import LitModel, TrainingConfig
import lightning.pytorch as pl
from Bio import SeqIO
from Bio.Seq import Seq
from torch.utils.data import DataLoader
import torch
import numpy as np 
from asb import  read_asb
import os
import glob





def _asb_predict_one_strand(trainer: pl.Trainer,
               model: pl.LightningModule, 
               cfg: TrainingConfig,
               data_path: str,
               genome: dict[str, Seq],
               ref: bool,
               reverse: bool,
               shift: int,
               window: int,
               one_indexed: bool):
    assert not cfg.use_reverse_channel, "Not implemented for models with reverse channel"
        
    ds = ASBDataset(asb_path=data_path,
                            genome=genome, 
                            return_ref=ref,
                            reverse=reverse,
                            window=window,
                            one_indexed=one_indexed,
                            shift=shift)
    dl = DataLoader(ds, 
                    batch_size=cfg.valid_batch_size,
                    num_workers=cfg.num_workers,
                    shuffle=False)
        
    y_preds =  trainer.predict(model,
                                dataloaders=dl)
    y_preds = torch.concat(y_preds).cpu().numpy() #type: ignore
    return y_preds 


def _shift_single_asb_predict(trainer: pl.Trainer,
               model: pl.LightningModule, 
               cfg: TrainingConfig,
               data_path: str,
               genome: dict[str, Seq],
               ref: bool,
               shift: int,
               window: int,
               one_indexed: bool) -> dict[str, np.ndarray]:
    
     
    forw_y_preds = _asb_predict_one_strand(trainer=trainer,
                                           model=model,
                                           cfg=cfg,
                                           data_path=data_path,
                                           genome=genome,
                                           ref=ref,
                                           window=window,
                                           one_indexed=one_indexed,
                                           shift=shift,
                                           reverse=False)
    rev_y_preds = _asb_predict_one_strand(trainer=trainer,
                                           model=model,
                                           cfg=cfg,
                                           data_path=data_path,
                                           genome=genome,
                                           ref=ref,
                                           window=window,
                                           one_indexed=one_indexed,
                                           shift=shift,
                                           reverse=True)
    
    return {"forw": forw_y_preds, "rev": rev_y_preds}


def _single_asb_predict(trainer: pl.Trainer,
                        model: pl.LightningModule, 
                        cfg: TrainingConfig,
                        data_path: str,
                        genome: dict[str, Seq],
                        ref: bool,
                        window: int,
                        one_indexed: bool,
                        max_shift: int,
                        shift_step: int) -> dict[str, np.ndarray]:
    dt = {}
    
    half_shift_range = list(range(0, max_shift+1, shift_step))
    shift_range = [-s for s in reversed(half_shift_range[1:])] + half_shift_range
    for shift in shift_range:
        shift_scores = _shift_single_asb_predict(trainer=trainer, 
                                            model=model, 
                                            cfg=cfg, 
                                            data_path=data_path, 
                                            genome=genome, 
                                            window=window, 
                                            one_indexed=one_indexed,
                                            shift=shift,
                                            ref=ref)
        for key, value in shift_scores.items():
            dt[f"{shift}_{key}"] = value
        
    return dt

def _model_asb_predict(trainer: pl.Trainer,
                       model: pl.LightningModule, 
                       cfg: TrainingConfig,
                       data_path: str,
                       genome: dict[str, Seq],
                       max_shift: int,
                       shift_step: int,
                       window: int = 231,
                       one_indexed: bool = False) -> dict[str, np.ndarray]:
    dt = {}
    ref_scores = _single_asb_predict(trainer=trainer, 
                                     model=model, 
                                     cfg=cfg, 
                                     data_path=data_path, 
                                     genome=genome, 
                                     window=window, 
                                     one_indexed=one_indexed,
                                     max_shift=max_shift,
                                     shift_step=shift_step,
                                     ref=True)
    for key, value in ref_scores.items():
        dt[f"ref_{key}"] = value
    alt_scores = _single_asb_predict(trainer=trainer, 
                                     model=model, 
                                     cfg=cfg, 
                                     data_path=data_path, 
                                     genome=genome, 
                                     window=window, 
                                     one_indexed=one_indexed,
                                     max_shift=max_shift,
                                     shift_step=shift_step,
                                     ref=False)
    for key, value in alt_scores.items():
        dt[f"alt_{key}"] = value    
        
    return dt

def asb_predict(model_paths: dict[str, str], 
                cfg: TrainingConfig,
                data_path: str,
                genome: dict[str, Seq],
                max_shift: int,
                shift_step: int,
                window: int = 231,
                one_indexed: bool = False) -> dict[str, np.ndarray]:
    
    dt = {}
    for name, m_path in model_paths.items():
        model = LitModel.load_from_checkpoint(m_path, 
                                      tr_cfg=train_cfg)

        trainer = pl.Trainer(accelerator='gpu',
                            devices=[args.device], 
                            precision='16-mixed')
        preds = _model_asb_predict(trainer=trainer, 
                            model=model, 
                            cfg=train_cfg, 
                            data_path=data_path,
                            genome=genome,
                            window=window, 
                            max_shift=max_shift,
                            shift_step=shift_step,
                            one_indexed=one_indexed)
        for key, value in preds.items():
            if key.startswith("ref"):
                key = key.replace("ref", name)
                key = f"ref_{key}"
            elif key.startswith("alt"):
                key = key.replace("alt", name)
                key = f"alt_{key}"
            else:
                raise NotImplementedError()
            dt[key] = value
    return dt


parser = argparse.ArgumentParser()
parser.add_argument("--config",
                    type=str,
                    help="path to model training config",
                    required=True)
parser.add_argument("--models_dir",
                    type=str,
                    help="path to dir with models checkpoints",
                    required=True)
parser.add_argument("--asb_path",
                    type=str,
                    help="path to asb info",
                    required=True)
parser.add_argument("--genome",
                    type=str,
                    help="path to genome in fasta",
                    required=True)
parser.add_argument("--out_path",
                    type=str,
                    help="path to output file",
                    required=True)
parser.add_argument("--device",
                    type=int,
                    required=True)
parser.add_argument("--max_shift",
                    type=int, 
                    required=True)
parser.add_argument("--shift_step",
                    type=int,
                    default=1)
parser.add_argument("--window",
                    type=int,
                    default=231)

args = parser.parse_args()
assert args.max_shift <= args.window // 2

genome = SeqIO.to_dict(SeqIO.parse(args.genome,
                                   format="fasta"))

train_cfg = TrainingConfig.from_json(args.config)

torch.set_float32_matmul_precision('medium') # type: ignore 

model_paths = {}
for p in glob.glob(os.path.join(args.models_dir, "*.ckpt")):
    name = os.path.basename(p).replace(".ckpt", "").replace("best_model_", "")
    model_paths[name] = p

preds = asb_predict(model_paths=model_paths,
                    cfg=train_cfg, 
                    data_path=args.asb_path,
                    genome=genome,
                    window=args.window, 
                    max_shift=args.max_shift,
                    shift_step=args.shift_step,
                    one_indexed=True)
data = read_asb(args.asb_path, for_eval=True)

for key, value in preds.items():
    data[f"pred_{key}"] = value

data.to_csv(args.out_path, 
            sep="\t",
            index=False)


