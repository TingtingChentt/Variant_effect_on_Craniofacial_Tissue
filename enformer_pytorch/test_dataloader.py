from data import GenomeIntervalDataset
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader


label_path = 'data/stage_labels.pth'
fasta_file = '/home/chent9/projects/enformer-tf/data/genome.fa'
context_length = 196608
batch_size = 8
num_workers = 0
data = torch.load(label_path, weights_only=False, map_location='cpu')
train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

train_ds = GenomeIntervalDataset(label_data=train_data, fasta_file=fasta_file, context_length=context_length)
val_ds = GenomeIntervalDataset(label_data=val_data, fasta_file=fasta_file, context_length=context_length)

# train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
# val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)


def find_bad_intervals(dataset, expected_length=context_length, max_checks=50000):
	"""Iterate dataset (single-worker) and print any intervals where the sequence length != expected_length."""
	print(f"Scanning up to {max_checks} samples for sequences != {expected_length}...")
	bad = []
	for i in range(min(len(dataset), max_checks)):
		item = dataset[i]
		# dataset returns (seq, target) where seq may be a tensor or tuple
		seq = item[0] if isinstance(item, (list, tuple)) else item

		# if dataset returns (seq, augs, ...) unpack
		if isinstance(seq, (list, tuple)):
			seq = seq[0]

		# seq expected shape: (L, C) or (C, L) depending on dataset
		if hasattr(seq, 'shape'):
			L = seq.shape[0]
		else:
			try:
				L = len(seq)
			except Exception:
				L = None

		if L != expected_length:
			# print some useful info if available
			info = None
			try:
				info = dataset.label_data[i]
			except Exception:
				info = 'N/A'

			print(f"Bad sample index {i}: seq_len={L}, info={info}")
			bad.append((i, L, info))

	print(f"Found {len(bad)} bad samples")
	return bad


if __name__ == '__main__':
	# run a quick scan on the train dataset with a single worker to debug shapes
	bad = find_bad_intervals(train_ds, expected_length=context_length, max_checks=50000)
	if len(bad) > 0:
		print('Example bad intervals (first 10):')
		for b in bad[:10]:
			print(b)

