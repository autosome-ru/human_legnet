from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from Bio.Seq import Seq
from torch.utils.data import Dataset
from utils import reverse_complement, encode_seq


def read_coverage(asb_path: str,  pos_fdr_upper: float=0.05, neg_fdr_lower: float = 0.5, for_eval: bool=False, eps: float=1e-10):
    t = pd.read_table(asb_path)
    t = t.rename({"#chr": "chr"}, axis=1)
    
    if for_eval:
        fdr_min = t['fdr_comb_pval'].values
        s = t['pref_allele'].apply(lambda x: -1 if x == "ref" else 1).values
        score = np.log2(fdr_min + eps) * s # type: ignore

        t["score"] = score
        t['cls'] = t['fdr_comb_pval'].apply(lambda x: 0 if x > neg_fdr_lower else 1 if x < pos_fdr_upper else -1)
        
        
    return t


class CoverageDataset(Dataset): 
    def __init__(self, 
                 asb_path: str, 
                 genome: dict[str, Seq],
                 return_ref: bool = True,
                 reverse: bool = False, 
                 window: int=231, 
                 shift: int = 0,
                 one_indexed: bool = False): 
        super().__init__()
        self.table = read_coverage(asb_path) 
        self.genome = genome 
        self.window = window
        assert self.window % 2 == 1
        self.shift = shift
        self.halfwindow = window // 2
        assert self.halfwindow >= self.shift
        self.one_indexed = one_indexed
        self.return_ref = return_ref
        self.reverse = reverse
    
    def __getitem__(self, idx):
        row = self.table.iloc[idx, :]
        #print(row)
        ch = self.genome[row.chr]  
        pos = int(row.start) - self.one_indexed
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
            #print(offset)
            seq[offset] = row.alt
            seq = "".join(seq)
            #print(seq)
        
        if self.reverse:
            seq = reverse_complement(seq)
        seq = encode_seq(seq)
        
        return seq, -1
    
    def __len__(self):
        return self.table.shape[0]