from torch.utils.data import Dataset

import numpy as np 
import pandas as pd

from Bio.Seq import Seq

from utils import encode_seq, reverse_complement

def read_asb(asb_path: str, pos_fdr_upper: float=0.05, neg_fdr_lower:float = 0.5, for_eval=False, eps=1e-10):
    t = pd.read_table(asb_path)
    t = t.rename({"#chr": "chr"}, axis=1)
   
    
    if for_eval:
        vals =  t[['fdrp_bh_ref', 'fdrp_bh_alt']].values
        fdr_argmin = vals.argmin(axis=1)
        fdr_min = np.take_along_axis(vals, fdr_argmin[:,None], axis=1).squeeze()
        s = 1 - fdr_argmin * 2 
        score = -np.log2(fdr_min + eps) * s 
        t["min_fdr"] = fdr_min
        t["score"] = score
        t['cls'] = t['min_fdr'].apply(lambda x: 0 if x > neg_fdr_lower else 1 if x < pos_fdr_upper else -1)
    return t


class ASBDataset(Dataset): 
    def __init__(self, 
                 asb_path: str, 
                 genome: dict[str, Seq],
                 return_ref: bool = True,
                 reverse: bool = False, 
                 window: int=231, 
                 shift: int = 0,
                 one_indexed: bool = True): 
        super().__init__()
        self.table = read_asb(asb_path) 
        self.genome = genome 
        self.window = window
        assert self.window % 2 == 1
        self.shift = shift
        self.halfwindow = window // 2
        assert self.halfwindow > self.shift
        self.one_indexed = one_indexed
        self.return_ref = return_ref
        self.reverse = reverse
    
    def __getitem__(self, idx):
        row = self.table.iloc[idx, :]
        #print(row)
        ch = self.genome[row.chr]  
        pos = int(row.pos) - self.one_indexed
        assert ch[pos] == row.ref, f"{ch[pos]} vs {row.ref}"
        pos = pos + self.shift
        start = pos - self.halfwindow
        end = pos + self.halfwindow + 1
        
        seq = ch[start:end]
        seq = seq.seq._data.decode() # type: ignore
        if not self.return_ref: # return alt
            #print(seq)
            seq = list(seq)
            offset = self.halfwindow - self.shift
            #print(offset, seq[offset])
            seq[offset] = row.alt
            seq = "".join(seq)
            #print(seq)
        
        if self.reverse:
            seq = reverse_complement(seq)
        seq = encode_seq(seq)
        
        return seq, -1
    
    def __len__(self):
        return self.table.shape[0]