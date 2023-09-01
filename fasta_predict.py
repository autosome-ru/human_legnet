import argparse
import torch
import lightning.pytorch as pl


from trainer import LitModel, TrainingConfig
from fasta import FastaDataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()
parser.add_argument("--fasta", 
                    type=str,
                    help="path to fasta file, sequence must be of the same size",
                    required=True)
parser.add_argument("--config",
                    type=str,
                    help="path to model training config",
                    required=True)
parser.add_argument("--model",
                    type=str,
                    help="path to model checkpoints",
                    required=True)
parser.add_argument("--out_path",
                    type=str,
                    help="path to output file",
                    required=True)
parser.add_argument("--device",
                    type=int,
                    required=True)
parser.add_argument("--batch_size",
                    type=int,
                    default=512)

def run_pred(model, trainer, fasta_path: str, batch_size: int, reverse: bool):
    dataset = FastaDataset(fasta_path, 
                           reverse=reverse)
    dl = DataLoader(dataset, batch_size=batch_size)
    y_preds =  trainer.predict(model,
                           dataloaders=dl)
    y_preds = torch.concat(y_preds).cpu().numpy() 
    return y_preds
    

args = parser.parse_args()

train_cfg = TrainingConfig.from_json(args.config)

model = LitModel.load_from_checkpoint(args.model, 
                                      tr_cfg=train_cfg)

trainer = pl.Trainer(accelerator='gpu',
                     devices=[args.device], 
                     precision='16-mixed')



y_preds =  run_pred(model=model, 
                    trainer=trainer,
                    fasta_path=args.fasta,
                    batch_size=args.batch_size,
                    reverse=False)
if train_cfg.reverse_augment:
    y_preds_rev = run_pred(model=model, 
                    trainer=trainer,
                    fasta_path=args.fasta,
                    batch_size=args.batch_size,
                    reverse=True)
    y_preds = (y_preds + y_preds_rev) / 2

ds = FastaDataset(args.fasta)
names = ds.seq_names()

with open(args.out_path, "w") as out:
    for name, score in zip(names, y_preds):
        print(name, 
              score, 
              sep="\t", 
              file=out)
