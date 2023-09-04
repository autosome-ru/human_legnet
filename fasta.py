from Bio import SeqIO
from torch.utils.data import Dataset
from utils import encode_seq, reverse_complement

class FastaDataset(Dataset):
    def __init__(self, 
                 file_path: str,
                 reverse: bool = False):
        super().__init__()
        self.file_path = file_path
        self.seqs = list(SeqIO.parse(file_path, format="fasta"))
        self.reverse = reverse
     
    def seq_names(self) -> list[str]:
        return [seq.id for seq in self.seqs]
    
    def raw_seqs(self) -> list[str]:
        return [self._get_seq(i) for i in range(0, len(self))]
    
    def _get_seq(self, idx):
        seq = self.seqs[idx]
        seq = seq.seq._data
        if isinstance(seq, bytes):
            seq = seq.decode()
        
        if self.reverse:
            seq = reverse_complement(seq)
        return seq
    
    def __getitem__(self, idx):
        seq = self._get_seq(idx)
        seq = encode_seq(seq)
        return seq
    
    def __len__(self) -> int:
        return len(self.seqs)