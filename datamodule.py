import lightning.pytorch as pl
import pandas as pd

from torch.utils.data import DataLoader
from dataset import TrainSeqDatasetProb, TestSeqDatasetProb

from training_config import TrainingConfig

class SeqDataModule(pl.LightningDataModule):
    def __init__(self, 
                 val_fold: int,
                 test_fold: int,
                 cfg: TrainingConfig):
        super().__init__()
        self.cfg = cfg
        
        df = pd.read_csv(self.cfg.data_path, 
                 sep='\t')
        df.columns = ['seq_id', 'seq', 'mean_value', 'fold_num', 'rev'][0:len(df.columns)]
        
        if "rev" in df.columns:
            df = df[df.rev == 0]
            
        self.train = df[~df.fold_num.isin([val_fold, test_fold])]
        self.valid = df[df.fold_num == val_fold]
        self.test = df[df.fold_num == test_fold]
        
    def train_dataloader(self):
        
        train_ds =  TrainSeqDatasetProb(self.train,
                                   use_reverse=self.cfg.reverse_augment,
                                   use_reverse_channel=self.cfg.use_reverse_channel,
                                   use_shift=self.cfg.use_shift,
                                   max_shift=self.cfg.max_shift)
        
        return DataLoader(train_ds, 
                          batch_size=self.cfg.train_batch_size,
                          num_workers=self.cfg.num_workers,
                          shuffle=True) 
    
    def val_dataloader(self):
        valid_ds = TestSeqDatasetProb(self.valid, 
                                  use_reverse_channel=self.cfg.use_reverse_channel,
                                  shift=0,
                                  reverse=False)

        return DataLoader(valid_ds, 
                          batch_size=self.cfg.valid_batch_size,
                          num_workers=self.cfg.num_workers,
                          shuffle=False)
        
    def dls_for_predictions(self):
        
        test_ds = TestSeqDatasetProb(self.test,
                                  use_reverse_channel=self.cfg.use_reverse_channel,
                                  shift=0,
                                  reverse=False)
        test_dl =  DataLoader(test_ds,
                              batch_size=self.cfg.valid_batch_size,
                              num_workers=self.cfg.num_workers,
                              shuffle=False)
        yield "forw_pred", test_dl
        if self.cfg.reverse_augment:
            rev_test_ds = TestSeqDatasetProb(self.test,
                                  use_reverse_channel=self.cfg.use_reverse_channel,
                                  shift=0,
                                  reverse=True)
            rev_test_dl =  DataLoader(rev_test_ds,
                              batch_size=self.cfg.valid_batch_size,
                              num_workers=self.cfg.num_workers,
                              shuffle=False)
            yield "rev_pred", rev_test_dl
