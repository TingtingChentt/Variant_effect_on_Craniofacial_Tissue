import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

import polars as pl
import numpy as np
from random import randrange, random
from pathlib import Path
from pyfaidx import Fasta
import pyBigWig
import os

# helper functions

def exists(val):
    return val is not None

def identity(t):
    return t

def cast_list(t):
    return t if isinstance(t, list) else [t]

def coin_flip():
    return random() > 0.5

# genomic function transforms

seq_indices_embed = torch.zeros(256).long()
seq_indices_embed[ord('a')] = 0
seq_indices_embed[ord('c')] = 1
seq_indices_embed[ord('g')] = 2
seq_indices_embed[ord('t')] = 3
seq_indices_embed[ord('n')] = 4
seq_indices_embed[ord('A')] = 0
seq_indices_embed[ord('C')] = 1
seq_indices_embed[ord('G')] = 2
seq_indices_embed[ord('T')] = 3
seq_indices_embed[ord('N')] = 4
seq_indices_embed[ord('.')] = -1

one_hot_embed = torch.zeros(256, 4)
one_hot_embed[ord('a')] = torch.Tensor([1., 0., 0., 0.])
one_hot_embed[ord('c')] = torch.Tensor([0., 1., 0., 0.])
one_hot_embed[ord('g')] = torch.Tensor([0., 0., 1., 0.])
one_hot_embed[ord('t')] = torch.Tensor([0., 0., 0., 1.])
one_hot_embed[ord('n')] = torch.Tensor([0., 0., 0., 0.])
one_hot_embed[ord('A')] = torch.Tensor([1., 0., 0., 0.])
one_hot_embed[ord('C')] = torch.Tensor([0., 1., 0., 0.])
one_hot_embed[ord('G')] = torch.Tensor([0., 0., 1., 0.])
one_hot_embed[ord('T')] = torch.Tensor([0., 0., 0., 1.])
one_hot_embed[ord('N')] = torch.Tensor([0., 0., 0., 0.])
one_hot_embed[ord('.')] = torch.Tensor([0.25, 0.25, 0.25, 0.25])

reverse_complement_map = torch.Tensor([3, 2, 1, 0, 4]).long()

def torch_fromstring(seq_strs):
    batched = not isinstance(seq_strs, str)
    seq_strs = cast_list(seq_strs)
    np_seq_chrs = list(map(lambda t: np.frombuffer(t.encode(), dtype = np.uint8).copy(), seq_strs))
    seq_chrs = list(map(torch.from_numpy, np_seq_chrs))
    return torch.stack(seq_chrs) if batched else seq_chrs[0]

def str_to_seq_indices(seq_strs):
    seq_chrs = torch_fromstring(seq_strs)
    return seq_indices_embed[seq_chrs.long()]

def str_to_one_hot(seq_strs):
    seq_chrs = torch_fromstring(seq_strs)
    return one_hot_embed[seq_chrs.long()]

def seq_indices_to_one_hot(t, padding = -1):
    is_padding = t == padding
    t = t.clamp(min = 0)
    one_hot = F.one_hot(t, num_classes = 5)
    out = one_hot[..., :4].float()
    out = out.masked_fill(is_padding[..., None], 0.25)
    return out

# augmentations

def seq_indices_reverse_complement(seq_indices):
    complement = reverse_complement_map[seq_indices.long()]
    return torch.flip(complement, dims = (-1,))

def one_hot_reverse_complement(one_hot):
    *_, n, d = one_hot.shape
    assert d == 4, 'must be one hot encoding with last dimension equal to 4'
    return torch.flip(one_hot, (-1, -2))

# processing bed files

class FastaInterval():
    def __init__(
        self,
        *,
        fasta_file,
        context_length = None,
        return_seq_indices = False,
        shift_augs = None,
        rc_aug = False
    ):
        fasta_file = Path(fasta_file)
        assert fasta_file.exists(), 'path to fasta file must exist'

        self.seqs = Fasta(str(fasta_file))
        self.return_seq_indices = return_seq_indices
        self.context_length = context_length
        self.shift_augs = shift_augs
        self.rc_aug = rc_aug

    def __call__(self, chr_name, start, end, return_augs = False):
        interval_length = end - start
        chromosome = self.seqs[chr_name]
        chromosome_length = len(chromosome)

        if exists(self.shift_augs):
            min_shift, max_shift = self.shift_augs
            max_shift += 1

            min_shift = min(max(start + min_shift, 0) - start, 0)
            max_shift = max(min(end + max_shift, chromosome_length) - end, 1)

            rand_shift = randrange(min_shift, max_shift)
            start += rand_shift
            end += rand_shift

        left_padding = right_padding = 0

        if exists(self.context_length) and interval_length < self.context_length:
            extra_seq = self.context_length - interval_length

            extra_left_seq = extra_seq // 2
            extra_right_seq = extra_seq - extra_left_seq

            start -= extra_left_seq
            end += extra_right_seq

        if start < 0:
            left_padding = -start
            start = 0

        if end > chromosome_length:
            right_padding = end - chromosome_length
            end = chromosome_length

        seq = ('.' * left_padding) + str(chromosome[start:end]) + ('.' * right_padding)

        should_rc_aug = self.rc_aug and coin_flip()

        if self.return_seq_indices:
            seq = str_to_seq_indices(seq)

            if should_rc_aug:
                seq = seq_indices_reverse_complement(seq)

            return seq

        one_hot = str_to_one_hot(seq)

        if should_rc_aug:
            one_hot = one_hot_reverse_complement(one_hot)

        if not return_augs:
            return one_hot

        # returns the shift integer as well as the bool (for whether reverse complement was activated)
        # for this particular genomic sequence

        # rand_shift_tensor = torch.tensor([rand_shift])
        rand_aug_bool_tensor = torch.tensor([should_rc_aug])

        return one_hot, rand_aug_bool_tensor

# For compatibility with DataLoader collate, return intervals as plain tuples (start, end)


# class GenomeIntervalDataset(Dataset):
#     def __init__(
#         self,
#         label_data,
#         fasta_file,
#         chr_bed_to_fasta_map = dict(),
#         context_length = None,
#         test=None,
#         return_seq_indices = False,
#         shift_augs = None,
#         rc_aug = False,
#         return_augs = False
#     ):
#         super().__init__()
#         self.test = test
#         self.label_data = label_data
#         self.context_length = context_length

#         # if the chromosome name in the bed file is different than the keyname in the fasta
#         # can remap on the fly
#         self.chr_bed_to_fasta_map = chr_bed_to_fasta_map

#         self.fasta = FastaInterval(
#             fasta_file = fasta_file,
#             context_length = context_length,
#             return_seq_indices = return_seq_indices,
#             shift_augs = shift_augs,
#             rc_aug = rc_aug
#         )

#         self.return_augs = return_augs

#     def __len__(self):
#         return len(self.label_data)

#     def __getitem__(self, ind):
#         # data keys: gene_id, gene_name, chrom, start, end, strand, tss, bin_index, target
#         data_i = self.label_data[ind]
#         chr_name, start, end = (data_i['chrom'], data_i['start'], data_i['end'])

#         chr_name = self.chr_bed_to_fasta_map.get(chr_name, chr_name)
#         seq = self.fasta(chr_name, start, end, return_augs = self.return_augs)

#         # If seq shape is (L, C)
#         if seq.ndim == 2:
#             L, _ = seq.shape
#             if L < self.context_length:
#                 pad_amount = self.context_length - L
#                 # pad: pad = (pad_left_w, pad_right_w, pad_top_h, pad_bottom_h)
#                 seq = F.pad(seq, (0, 0, 0, pad_amount))   # pad bottom of axis0
#             elif L > self.context_length:
#                 seq = seq[:self.context_length]
#         else:
#             # handle seq shape conventions if different
#             pass

#         assert seq.shape[-2] == self.context_length, f"Sequence length {seq.shape[-2]} does not match expected context length {self.context_length} at index {ind} (chr {chr_name}:{start}-{end})"
#         target = torch.from_numpy(data_i['target'].T).float()

#         if self.test:
#             interval = (int(start), int(end))
#             seq_info = data_i['gene_name'] + '_' + chr_name + f':{start}-{end}' + '_tss_' + str(data_i['tss'])
#             return seq, target, interval, seq_info
        
#         return seq, target


def get_target_from_bigwig(data_folder, bigwig_files, chr_name, start, end, n_bins=896, clip_pct=99.9):
    target = []
    
    for bigwig_file in bigwig_files:
        bigwig_file = data_folder + bigwig_file
        bw = pyBigWig.open(bigwig_file)
        
        try:
            values = bw.stats(chr_name, start, end, nBins=n_bins, type="sum")
            bw.close()
            values = np.array(values, dtype=float)
        except Exception as e:
            print(f"Error getting values for {chr_name}:{start}-{end}: {e}")
            values = np.zeros(n_bins)
        
        # Replace NaN values with zeros
        values = np.nan_to_num(values, nan=0.0)
        values = np.log1p(values)

        # optional clipping of extreme outliers
        if clip_pct is not None:
            cap = np.percentile(values, clip_pct)
            if np.isfinite(cap) and cap > 0:
                values = np.minimum(values, cap)

        target.append(values)
        
    target = np.stack(target, axis=1)  # shape: (n_bins, num_tracks)
    return target

class GenomeIntervalDataset(Dataset):
    def __init__(
        self,
        data_folder,
        bed_file,
        bigwig_files,
        fasta_file,
        filter_df_fn = identity,
        chr_bed_to_fasta_map = dict(),
        context_length = None,
        test=None,
        return_seq_indices = False,
        shift_augs = None,
        rc_aug = False,
        return_augs = False
    ):
        super().__init__()
        bed_path = os.path.join(data_folder, bed_file)
        bed_path = Path(bed_path)
        assert bed_path.exists(), 'path to .bed file must exist'

        df = pl.read_csv(str(bed_path), separator = '\t', has_header = False)
        
        # Ensure proper data types for bed file columns
        # Column 0: chromosome (string), Column 1: start (int), Column 2: end (int)
        column_names = df.columns
        if len(column_names) >= 3:
            df = df.with_columns([
                pl.col(column_names[0]).cast(pl.Utf8).alias('chrom'),
                pl.col(column_names[1]).cast(pl.Int64).alias('start'), 
                pl.col(column_names[2]).cast(pl.Int64).alias('end')
            ]).select(['chrom', 'start', 'end'])
        
        # Add row_index column for filtering
        df = df.with_row_index('row_index')
        df = filter_df_fn(df)
        self.df = df

        # if the chromosome name in the bed file is different than the keyname in the fasta
        # can remap on the fly
        self.chr_bed_to_fasta_map = chr_bed_to_fasta_map

        self.fasta = FastaInterval(
            fasta_file = fasta_file,
            context_length = context_length,
            return_seq_indices = return_seq_indices,
            shift_augs = shift_augs,
            rc_aug = rc_aug
        )

        self.return_augs = return_augs
        self.data_folder = data_folder
        self.bigwig_files = bigwig_files
        self.context_length = context_length
        self.test = test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ind):
        # Access columns by name to avoid indexing issues
        chr_name = str(self.df.row(ind, named=True)['chrom'])
        start = int(self.df.row(ind, named=True)['start'])  
        end = int(self.df.row(ind, named=True)['end'])
        
        chr_name = self.chr_bed_to_fasta_map.get(chr_name, chr_name)

        # Get sequence with augmentation info if return_augs is True
        if self.return_augs:
            seq, rand_aug_bool_tensor = self.fasta(chr_name, start, end, return_augs = True)
            rc_applied = rand_aug_bool_tensor.item()
        else:
            seq = self.fasta(chr_name, start, end, return_augs = False)
            rc_applied = False
        
        # If seq shape is (L, C)
        if seq.ndim == 2:
            L, _ = seq.shape
            if L < self.context_length:
                pad_amount = self.context_length - L
                # pad: pad = (pad_left_w, pad_right_w, pad_top_h, pad_bottom_h)
                seq = F.pad(seq, (0, 0, 0, pad_amount))   # pad bottom of axis0
            elif L > self.context_length:
                seq = seq[:self.context_length]
        else:
            # handle seq shape conventions if different
            pass
            
        assert seq.shape[-2] == self.context_length, f"Sequence length {seq.shape[-2]} does not match expected context length {self.context_length} at index {ind} (chr {chr_name}:{start}-{end})"

        # print(chr_name, target_start, target_end)
        effective_length = 114688
        target = get_target_from_bigwig(
            data_folder=self.data_folder,
            bigwig_files=self.bigwig_files,
            chr_name=chr_name,
            start=start,
            end=end,
            n_bins=effective_length//128,
            clip_pct=99.9
        )
        target = torch.from_numpy(target).float()

        # If reverse complement was applied to sequence, also reverse the target signal
        if rc_applied:
            target = torch.flip(target, dims=(0,))  # Flip along the sequence dimension (now dim 0)
        
        if self.test:
            interval = (int(start), int(end))
            seq_info = chr_name + f':{start}-{end}'
            return seq, target, interval, seq_info
        
        return seq, target