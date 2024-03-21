import pandas as pd
import numpy as np
import pandas as pd

import torch

from torch.utils.data import  Dataset
from utils import Seq2Tensor, reverse_complement

class TrainSeqDatasetProb(Dataset):
    
    """ Sequence dataset. """
    
    def __init__(self, 
                 ds: pd.DataFrame,
                 use_reverse: bool,
                 use_shift: bool,
                 use_reverse_channel: bool,  
                 seqsize=230,
                 max_shift: tuple[int, int] | None = None, 
                 training=True):
        """
        Parameters
        ----------
        ds : pd.DataFrame
            Training dataset.
        use_reverse : bool
            If True, additional reverse augmentation is used.
        use_reverse_channel : bool
            If True, extra channel indicating reversed sequences is added.
        seqsize : int
            Constant sequence length.
        """
        self.training = training

        self.ds = ds
        self.totensor = Seq2Tensor() 
        self.use_reverse = use_reverse
        self.use_shift = use_shift
        self.use_reverse_channel = use_reverse_channel
        self.forward_side = "GGCCCGCTCTAGACCTGCAGG"
        self.reverse_side = "CACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGT"
        self.seqsize = seqsize 
        if max_shift is None:
            self.max_shift = (0, len(self.forward_side))
        else:
            self.max_shift = max_shift
            
    def transform(self, x):
        assert isinstance(x, str)
        return self.totensor(x)
    
    def __getitem__(self, i):

        seq = self.ds.seq.values[i]
        
        if self.use_shift:
            shift = torch.randint(size=(1,), low=-self.max_shift[0], high=self.max_shift[1] + 1).item()
            if shift < 0: # use forward primer
                seq = seq[:shift]
                seq = self.forward_side[shift:] + seq
            elif shift > 0:
                seq = seq[shift:]
                seq = seq + self.reverse_side[:shift]
            else: # shift = 0
                pass # nothing to do

        if self.use_reverse:
            r = torch.rand((1,)).item()
            if  r > 0.5:
                seq = reverse_complement(seq)
                rev = 1.0
            else:
                rev = 0.0
        else:
            rev = 0.0
            
        seq = self.transform(seq)
        to_concat = [seq]
        
        # add reverse augmentation channel
        if self.use_reverse_channel:
            rev = torch.full( (1, self.seqsize), rev, dtype=torch.float32)
            to_concat.append(rev)
            
        # create final tensor
        if len(to_concat) > 1:
            X = torch.concat(to_concat, dim=0)
        else:
            X = seq
            
        mean = self.ds.mean_value.values[i]
        
        return X, mean.astype(np.float32)
    
    def __len__(self):
        return len(self.ds.seq)
    
    
class TestSeqDatasetProb(Dataset):
    
    """ Sequence dataset. """
    
    def __init__(self, 
                 ds: pd.DataFrame,
                 reverse: bool,
                 shift: int,  
                 use_reverse_channel: bool = True,
                 seqsize=230):
        """
        Parameters
        ----------
        ds : pd.DataFrame
            Training dataset.
        use_reverse_channel : bool
            If True, extra channel indicating reversed sequences is added.
        seqsize : int
            Constant sequence length.
        """
       
        self.ds = ds
        self.totensor = Seq2Tensor()
        self.use_reverse_channel = use_reverse_channel 
        self.reverse = reverse
        self.shift = shift
        self.forward_side = "GGCCCGCTCTAGACCTGCAGG"
        self.reverse_side = "CACTAGAGGGTATATAATGGAAGCTCGACTTCCAGCTTGGCAATCCGGTACTGT"
        self.seqsize = seqsize 

        
    def transform(self, x):
        assert isinstance(x, str)
        return self.totensor(x)
    
    def __getitem__(self, i):
        """
        Output
        ----------
        X: torch.Tensor    
            Create one-hot encoding tensor with reverse channel if required.   
        bin: float 
            Training expression value
        """
        seq = self.ds.seq.values[i]
        
        if self.shift < 0: # use forward primer
            seq = seq[:self.shift]
            seq = self.forward_side[self.shift:] + seq
        elif self.shift > 0:
            seq = seq[self.shift:]
            seq = seq + self.reverse_side[:self.shift]
        else: # shift = 0
            pass # nothing to do

        
        if self.reverse:
            seq = reverse_complement(seq)
            rev = 1.0
        else:
            rev = 0.0

        seq = self.transform(seq)
        to_concat = [seq]
        
        # add reverse augmentation channel
        if self.use_reverse_channel:
            rev = torch.full( (1, self.seqsize), rev, dtype=torch.float32)
            to_concat.append(rev)
            
        # create final tensor
        if len(to_concat) > 1:
            X = torch.concat(to_concat, dim=0)
        else:
            X = seq
            
        mean = self.ds.mean_value.values[i]
        
        return X, mean.astype(np.float32)
    
    def __len__(self):
        return len(self.ds.seq)


