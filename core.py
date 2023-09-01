import torch 

import lightning.pytorch as pl



from datamodule import SeqDataModule
from test_predict import save_predict
from trainer import LitModel, TrainingConfig
from utils import set_global_seed, parameter_count
from lightning.pytorch.callbacks import ModelCheckpoint
from pathlib import Path 

cell_types = ['HepG2', 'K562', 'WTC11']
cell_type = cell_types[2]


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
general.add_argument("--device", 
                     type=int,
                     default=0)
general.add_argument("--num_workers",
                     type=int, 
                     default=8)
general.add_argument("--fraction",
                     type=float,
                    default=1.0)
general.add_argument("--seed",
                     type=int,
                     default=777)

aug = parser.add_argument_group('aug args', 
                                'augmentation arguments')
aug.add_argument("--reverse_augment", 
                 action="store_true")
aug.add_argument("--use_reverse_channel",
                 action="store_true")
aug.add_argument("--use_shift", 
                 action="store_true")
aug.add_argument("--max_shift",
                 default=None, 
                 nargs=2,
                 type=int)

model_args =  parser.add_argument_group('model arguments', 
                                'model architecture arguments')
model_args.add_argument("--stem_ch", 
                   type=int,
                   default=64)
model_args.add_argument("--stem_ks",
                   type=int,
                   default=11)
model_args.add_argument("--ef_ks",
                        type=int,
                        default=9)
model_args.add_argument("--ef_block_sizes", 
                        type=int,
                        nargs="+",
                        default=[80, 96, 112, 128])
model_args.add_argument("--resize_factor",
                        type=int,
                        default=4)
model_args.add_argument("--pool_sizes", 
                        type=int,
                        nargs="+",
                        default=[2, 2, 2, 2])

scheduler_args =  parser.add_argument_group('scheduler arguments', 
                                'One cycle scheduler arguments')
scheduler_args.add_argument("--max_lr", 
                            type=float,
                            default=0.01)
scheduler_args.add_argument("--weight_decay",
                            type=float,
                            default=0.1)
scheduler_args.add_argument("--epoch_num",
                            type=int,
                            default=20)
scheduler_args.add_argument("--train_batch_size",
                            type=int, 
                            default=1024)

valid_args =  parser.add_argument_group('valid arguments', 
                                'Validation arguments')
valid_args.add_argument("--valid_batch_size",
                            type=int,
                            default=1024)

args = parser.parse_args()

train_cfg = TrainingConfig(
    # general options 
    training=True,
    model_dir=args.model_dir,
    data_path=args.data_path,
    num_workers = args.num_workers,
    device=args.device,
    seed=args.seed,
    # aug options
    reverse_augment=args.reverse_augment,
    use_reverse_channel=args.use_reverse_channel,
    use_shift=args.use_shift,
    max_shift=args.max_shift,     
    # model architecture
    stem_ch = args.stem_ch,
    stem_ks = args.stem_ks,
    ef_ks = args.ef_ks,
    ef_block_sizes = args.ef_block_sizes,
    resize_factor = args.resize_factor,
    pool_sizes = args.pool_sizes,
    # scheduler options
    max_lr = args.max_lr,
    weight_decay = args.weight_decay,
    epoch_num=args.epoch_num,
    train_batch_size=args.train_batch_size,
    # validation options
    valid_batch_size=args.valid_batch_size)


model_dir = Path(train_cfg.model_dir)
model_dir.mkdir(exist_ok=True,
                parents=True)

train_cfg.dump()

torch.set_float32_matmul_precision('medium') # type: ignore 

for test_fold in range(1, 11):
    for val_fold in range(1, 11):
        if test_fold == val_fold:
            continue
        set_global_seed(train_cfg.seed)
        
        model = LitModel(tr_cfg=train_cfg)
        print(parameter_count(model))

        data = SeqDataModule(val_fold=val_fold,
                             test_fold=test_fold,
                             cfg=train_cfg)
        
        train_dl = data.train_dataloader()
        valid_dl = data.val_dataloader()
        
    
        dump_dir = model_dir / f"model_{val_fold}_{test_fold}"
        last_checkpoint_callback = pl.callbacks.ModelCheckpoint(   #type: ignore
            save_top_k=1,
            monitor="step",
            mode="max",
            filename="last_model-{epoch}",
            save_on_train_epoch_end=True,
        )
        
        best_checkpoint_callback = ModelCheckpoint(
            save_top_k=1,
            monitor="val_pearson",
            mode="max",
            filename="pearson-{epoch:02d}-{val_pearson:.2f}",
        )

        trainer = pl.Trainer(accelerator='gpu',
                            enable_checkpointing=True,
                            devices=[train_cfg.device], 
                            precision='16-mixed', 
                            max_epochs=train_cfg.epoch_num,
                            callbacks=[last_checkpoint_callback,  best_checkpoint_callback],
                            gradient_clip_val=1,
                            default_root_dir=dump_dir)

        trainer.fit(model, 
                    datamodule=data)
        model = LitModel.load_from_checkpoint(best_checkpoint_callback.best_model_path, 
                                              tr_cfg=train_cfg)
        
        df_pred = save_predict(trainer, 
                               model, 
                               data,
                               save_dir=dump_dir, 
                               pref="new_format")
