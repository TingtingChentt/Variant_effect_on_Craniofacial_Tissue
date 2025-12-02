import argparse
import os
from pathlib import Path

import torch
from torch import amp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from finetune import *
from data import GenomeIntervalDataset
from enformer_pytorch.modeling_enformer import from_pretrained
from enformer_pytorch.finetune import HeadAdapterWrapper
from enformer_pytorch.modeling_enformer import pearson_corr_coef
import numpy as np
import random
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import seaborn as sns


def plot_tracks(tracks, interval, seq_info, height=1.5):
    """tracks: Ordered mapping title -> 1D array-like OR title -> (series1, series2,...)
    If the value is a sequence, each element will be plotted on the same axes (overlay).
    interval: object with .start and .end attributes (or tuple)
    returns matplotlib.Figure
    """
    items = list(tracks.items())
    n = len(items)
    fig, axes = plt.subplots(n, 1, figsize=(20, height * n), sharex=True)
    if n == 1:
        axes = [axes]
        
    def _interval_start_end(interval, length):
        # interval may be a tuple/list (start, end) or an object with .start/.end
        if isinstance(interval, (list, tuple)):
            try:
                s, e = int(interval[0]), int(interval[1])
            except Exception:
                s, e = 0, length - 1
        else:
            s = getattr(interval, 'start', 0)
            e = getattr(interval, 'end', length - 1)
        return s, e

    labels = ['pred', 'target']
    for ax, (title, y) in zip(axes, items):
        # y may be a single 1D array or a sequence of arrays to overlay
        if isinstance(y, (list, tuple)):
            # take length from first series
            first = y[0]
            s, e = _interval_start_end(interval, len(first))
            x = np.linspace(s, e, num=len(first))
            for idx, series in enumerate(y):
                try:
                    arr = np.array(series)
                    ax.plot(x, arr, label=labels[idx])
                except Exception:
                    pass
            ax.legend()
        else:
            arr = np.array(y)
            s, e = _interval_start_end(interval, len(arr))
            x = np.linspace(s, e, num=len(arr))
            ax.fill_between(x, arr)
        ax.set_title(title)
        try:
            sns.despine(top=True, right=True, bottom=True)
        except Exception:
            pass

    # set xlabel: prefer a human-readable seq_info when provided, else fall back to interval
    try:
        if seq_info is not None:
            axes[-1].set_xlabel(seq_info)
        else:
            s_all, e_all = _interval_start_end(interval, 0)
            axes[-1].set_xlabel(f"{s_all}:{e_all}")
    except Exception:
        try:
            axes[-1].set_xlabel(str(seq_info if seq_info is not None else interval))
        except Exception:
            axes[-1].set_xlabel('')
    plt.tight_layout()
    return fig



def train_epoch(
    model,
    loader,
    optimizer,
    scaler,
    device,
    writer=None,
    global_step_start=0,
    clip_grad_norm=None,
    freeze_enformer=False,
    finetune_enformer_ln_only=False,
    print_every=1,
    log_every=1,
    log_grads=False,
    grad_log_every=100,
):
    model.train()
    total_loss = 0.0
    iters = 0
    global_step = global_step_start

    for item in loader:
        seq, target = item

        # dataset may optionally return augment info as tuple
        if isinstance(seq, (list, tuple)):
            seq = seq[0]

        seq = seq.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad()

        # run forward under autocast only if we have a scaler (AMP enabled)
        if scaler is not None:
            with amp.autocast(device_type=device.type):
                loss = model(seq, target=target, freeze_enformer=freeze_enformer, finetune_enformer_ln_only=finetune_enformer_ln_only)
        else:
            # no AMP: run in full precision
            loss = model(seq, target=target, freeze_enformer=freeze_enformer, finetune_enformer_ln_only=finetune_enformer_ln_only)

        # model returns a scalar loss (poisson_loss); ensure it's a scalar tensor
        if hasattr(loss, 'mean') and loss.numel() > 1:
            loss = loss.mean()

        # backward: use scaler if available, else normal backward
        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # ensure we only call unscale_ once per step
        unscaled = False
        if (clip_grad_norm is not None) or (log_grads and ((global_step % grad_log_every) == 0)):
            try:
                if (scaler is not None) and hasattr(scaler, 'unscale_'):
                    scaler.unscale_(optimizer)
                    unscaled = True
            except Exception:
                pass

        if clip_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad_norm)

        # step optimizer: use scaler if present
        if scaler is not None:
            try:
                scaler.step(optimizer)
                scaler.update()
            except Exception:
                # if scaler fails for any reason, fall back to a plain step
                optimizer.step()
        else:
            optimizer.step()

        total_loss += float(loss.detach().cpu().item())
        iters += 1

        # log per-step training loss to TensorBoard if provided (controlled by log_every)
        if writer is not None and (log_every <= 1 or (global_step % log_every == 0)):
            writer.add_scalar('train/loss_step', float(loss.detach().cpu().item()), global_step)

        # compute and log grad norm if requested (after unscale and clipping)
        grad_norm = None
        if log_grads and ((global_step % grad_log_every) == 0):
            total_norm_sq = 0.0
            for p in model.parameters():
                if p.grad is None:
                    continue
                try:
                    gnorm = float(p.grad.detach().data.norm(2).item())
                except Exception:
                    continue
                total_norm_sq += gnorm * gnorm
            grad_norm = total_norm_sq ** 0.5
            if writer is not None:
                writer.add_scalar('train/grad_norm', grad_norm, global_step)

        # print per-step training loss (controlled by print_every)
        if (print_every <= 1) or (global_step % print_every == 0):
            try:
                if grad_norm is not None:
                    print(f"Train step {global_step}: loss={float(loss.detach().cpu().item()):.6f}, grad_norm={grad_norm:.6f}")
                else:
                    print(f"Train step {global_step}: loss={float(loss.detach().cpu().item()):.6f}")
            except Exception:
                print(f"Train step {global_step}: loss=<unavailable>")

        global_step += 1

    return total_loss / max(1, iters), global_step


def validate(model, loader, device):
    model.eval()
    total_loss = 0.0
    iters = 0

    # assume number of tracks equals last dimension of target
    # peek one batch to get channels
    # validate using pearson_corr_coef from modeling_enformer
    with torch.no_grad():
        pearson_sum = 0.0
        pearson_count = 0

        for item in loader:
            seq, target = item

            if isinstance(seq, (list, tuple)):
                seq = seq[0]

            seq = seq.to(device)
            target = target.to(device)

            preds = model(seq)

            # preds, target expected shapes: (b, n, channels)
            loss = poisson_loss(preds, target)

            # pearson_corr_coef returns per-sample Pearson (one value per batch element)
            pearson_vals = pearson_corr_coef(preds, target)  # shape (b,)

            pearson_sum += float(pearson_vals.sum().cpu().item())
            pearson_count += int(pearson_vals.numel())

            total_loss += float(loss.cpu().item())
            iters += 1

            # print per-validation-batch loss and pearson mean for this batch
            try:
                batch_pearson_mean = float(pearson_vals.mean().cpu().item())
            except Exception:
                batch_pearson_mean = None
            print(f"Val batch {iters}: loss={float(loss.cpu().item()):.6f}, pearson_mean={batch_pearson_mean}")

    mean_corr = (pearson_sum / pearson_count) if pearson_count > 0 else None
    return total_loss / max(1, iters), mean_corr



def save_checkpoint(state, out_dir: Path, name: str = "checkpoint.pt"):
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, out_dir / name)


def split_by_chromosomes(data):
    """Split list-of-records by fixed chromosome assignments.

    Returns (train_list, val_list, test_list).
    Uses predefined chromosome sets for deterministic splitting.
    """
    train_chroms = [
        "chr1", "chr3", "chr4", "chr5", "chr6", "chr7", "chr9",
        "chr11", "chr12", "chr13", "chr14", "chr15", "chr16",
        "chr17", "chr18", "chr19", "chr20", "chr21", "chr22", "chrX"
    ]
    val_chroms = ["chr8"]
    test_chroms = ["chr2", "chr10"]
    
    by_chr = defaultdict(list)
    for rec in data:
        chrom = None
        if isinstance(rec, dict):
            chrom = rec.get('chrom') or rec.get('chr') or rec.get('chromosome')
        # fall back to 'unknown' for items without chromosome info
        if chrom is None:
            chrom = 'unknown'
        by_chr[chrom].append(rec)

    print(f"Available chromosomes in data: {sorted(by_chr.keys())}")

    test = []
    val = []
    train = []

    # Allocate based on predefined chromosome sets
    for c in by_chr.keys():
        if c in test_chroms:
            print(f"Allocating chromosome {c} to test set ({len(by_chr[c])} samples)")
            test.extend(by_chr[c])
        elif c in val_chroms:
            print(f"Allocating chromosome {c} to val set ({len(by_chr[c])} samples)")
            val.extend(by_chr[c])
        elif c in train_chroms:
            print(f"Allocating chromosome {c} to train set ({len(by_chr[c])} samples)")
            train.extend(by_chr[c])
        else:
            print(f"Warning: chromosome {c} not in any predefined set, skipping")
    
    print(f"Final split sizes: train={len(train)}, val={len(val)}, test={len(test)}")
    
    return train, val, test


def build_dataloaders(
    data_folder,
    bed_file,
    bigwig_files,
    fasta_file,
    batch_size,
    context_length=196608,
    num_workers=4,
    rc_aug=False,
    return_augs=False
):
    """
    Build DataLoaders. Supports:
      - split_method='chrom'  : chromosome holdout split into train/val/test

    Returns (train_loader, val_loader, test_loader_or_None)
    """
    import polars as pl
    
    bed_path = os.path.join(data_folder, bed_file)
    bed_path = Path(bed_path)
    assert bed_path.exists(), f'Bed file must exist: {bed_path}'
    
    # Read the bed file and convert to list of dictionaries for splitting
    df = pl.read_csv(str(bed_path), separator='\t', has_header=False)
    
    # Convert polars DataFrame to list of dicts with chromosome info
    data = []
    for i in range(len(df)):
        row = df.row(i)
        data.append({
            'chrom': row[0],
            'start': row[1], 
            'end': row[2],
            'row_index': i  # Keep track of original row index
        })

    # Split data by chromosomes
    train_data, val_data, test_data = split_by_chromosomes(data)

    # Create datasets using the new GenomeIntervalDataset that reads bed files directly
    def create_filter_fn(indices_to_keep):
        def filter_fn(df):
            return df.filter(pl.col('row_index').is_in(indices_to_keep))
        return filter_fn
    
    train_indices = [x['row_index'] for x in train_data]
    val_indices = [x['row_index'] for x in val_data]
    
    train_ds = GenomeIntervalDataset(
        data_folder=data_folder,
        bed_file=bed_file,
        bigwig_files=bigwig_files,
        fasta_file=fasta_file,
        context_length=context_length,
        filter_df_fn=create_filter_fn(train_indices),
        rc_aug=rc_aug,
        return_augs=return_augs
    )
    
    # set rc_aug and return_augs to False for validation dataset
    val_ds = GenomeIntervalDataset(
        data_folder=data_folder,
        bed_file=bed_file, 
        bigwig_files=bigwig_files,
        fasta_file=fasta_file,
        context_length=context_length,
        filter_df_fn=create_filter_fn(val_indices)
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    # set rc_aug and return_augs to False for test dataset
    test_indices = [x['row_index'] for x in test_data]
    test_ds = GenomeIntervalDataset(
        data_folder=data_folder,
        bed_file=bed_file,
        bigwig_files=bigwig_files, 
        fasta_file=fasta_file,
        context_length=context_length,
        filter_df_fn=create_filter_fn(test_indices),
        test=True
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, help='name of the model to use')
    parser.add_argument('--labels', type=str, default='data/GSE197513_stage_logcpm_labels.pth')
    parser.add_argument('--data-folder', type=str, default='/home/chent9/Datasets/impute_H3K27ac_downloads')
    parser.add_argument('--bed-file', type=str, default='All_stages_H3K27ac.union_peaks.gap10000.centered_ext114688.in_chrom_bounds.bed')
    parser.add_argument('--bigwig-files', type=list, default=["/CS13_H3K27ac.mean.pval.bw", "/CS14_H3K27ac.mean.pval.bw", "/CS15_H3K27ac.mean.pval.bw", "/CS17_H3K27ac.mean.pval.bw","/impute_CS20-12104_H3K27ac.pval.signal.bigWig"])
    parser.add_argument('--fasta', type=str, default='/home/chent9/projects/enformer-tf/data/genome.fa')
    parser.add_argument('--rc-aug', action='store_true', help='use reverse-complement augmentation during training')
    parser.add_argument('--return-augs', action='store_true', help='return augmented samples in addition to original samples')
    parser.add_argument('--pretrained', type=str, default='EleutherAI/enformer-official-rough')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--out-dir', type=str, default='results/')
    parser.add_argument('--num-tracks', type=int, default=6)
    parser.add_argument('--clip-grad', type=float, default=1.0)
    parser.add_argument('--debug', action='store_true', help='run a single-batch debug and exit')
    parser.add_argument('--print-every', type=int, default=50, help='print train loss every N steps')
    parser.add_argument('--log-every', type=int, default=50, help='log to tensorboard every N steps')
    parser.add_argument('--log-grads', action='store_true', help='log gradient norm every grad-log-every steps')
    parser.add_argument('--grad-log-every', type=int, default=100, help='how often to log gradient norms')
    parser.add_argument('--no-amp', action='store_true', help='disable automatic mixed precision (AMP) and GradScaler')
    parser.add_argument('--seed', type=int, default=42, help='random seed for splits')
    parser.add_argument('--test-plot-samples', type=int, default=5, help='how many random test samples to plot in TensorBoard')
    parser.add_argument('--test-plot-seed', type=int, default=42, help='seed for selecting test samples to plot')
    parser.add_argument('--test-only', action='store_true', help='run test inference only using best checkpoint and exit')
    parser.add_argument('--best-checkpoint', type=str, default='', help='path to best checkpoint to use for test-only inference (overrides default best.pt location)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    train_loader, val_loader, test_loader = build_dataloaders(
        data_folder=args.data_folder,
        bed_file=args.bed_file,
        bigwig_files=args.bigwig_files,
        fasta_file=args.fasta,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        rc_aug=args.rc_aug,
        return_augs=args.return_augs
    )

    # ensure a model name
    model_name = args.model_name or 'run'
    print('running with model:', model_name)

    base_out = Path(args.out_dir)
    model_out_dir = base_out / model_name / 'model'
    log_dir = base_out / model_name / 'logs'

    model_out_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # set up TensorBoard writer
    writer = SummaryWriter(str(log_dir))

    # finetune enformer on a limited budget by setting use_checkpointing = True
    # use_tf_gamma = False, use_checkpointing = True
    enformer = from_pretrained(args.pretrained)
    model = HeadAdapterWrapper(enformer=enformer, num_tracks=args.num_tracks, post_transformer_embed=False).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, min_lr=1e-6)
    # If user disabled AMP, use a None scaler and run in full precision.
    if args.no_amp:
        scaler = None
    else:
        # create a GradScaler (keep simple; some older torch versions differ in constructor args)
        try:
            scaler = amp.GradScaler(device='cuda' if device.type == 'cuda' else 'cpu')
        except Exception:
            # fallback to default constructor
            try:
                scaler = amp.GradScaler()
            except Exception:
                # last resort: no scaler (behave like --no-amp)
                scaler = None

    # # If user requested test-only, load checkpoint (if present) and run test inference, then exit
    # if args.test_only:
    #     if test_loader is None:
    #         print('No test_loader available. To run test-only, provide a test split (e.g. --test-split 0.1) or a dataset with test records.')
    #         return
    #     # prefer explicit path if provided
    #     best_path = Path(args.best_checkpoint) if getattr(args, 'best_checkpoint', None) else (model_out_dir / 'best.pt')
    #     if best_path.exists():
    #         print('Loading best checkpoint for test-only inference:', best_path)
    #         ckpt = torch.load(best_path, map_location=device)
    #         try:
    #             model.load_state_dict(ckpt['model_state_dict'])
    #         except Exception:
    #             model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
    #     else:
    #         print('Best checkpoint not found at', best_path, '; using current model weights for test-only inference')

    #     # Run the test block (same logic as the previous inline test implementation)
    #     model.eval()
    #     sample_idx = 0
    #     pearson_sum = 0.0
    #     pearson_count = 0
    #     plot_k = max(0, int(getattr(args, 'test_plot_samples', 0)))
    #     plot_seed = int(getattr(args, 'test_plot_seed', 42))
    #     rng = random.Random(plot_seed)
    #     reservoir = []
    #     # save_pred_target = []
    #     with torch.no_grad():
    #         for item in test_loader:
    #             seq, target, interval, seq_info = item

    #             if isinstance(seq, (list, tuple)):
    #                 seq = seq[0]

    #             seq = seq.to(device)
    #             target = target.to(device)

    #             preds = model(seq)
    #             pearson_vals = pearson_corr_coef(preds, target)
    #             # save_pred_target.append({'preds': preds.cpu(), 'target': target.cpu()})

    #             if pearson_vals is not None:
    #                 for v in pearson_vals.cpu().tolist():
    #                     writer.add_scalar('test/pearson_sample', float(v), sample_idx)
    #                     sample_idx += 1
    #                 pearson_sum += float(pearson_vals.sum().cpu().item())
    #                 pearson_count += int(pearson_vals.numel())

    #             if plot_k > 0:
    #                 batch_size = preds.shape[0]
    #                 for i in range(batch_size):
    #                     sample_interval = None
    #                     try:
    #                         if isinstance(interval, (list, tuple)):
    #                             sample_interval = interval[i]
    #                         else:
    #                             sample_interval = interval
    #                     except Exception:
    #                         sample_interval = None

    #                     entry = (seq[i].detach().cpu().clone(), target[i].detach().cpu().clone(), preds[i].detach().cpu().clone(), sample_interval, seq_info[i])
    #                     if len(reservoir) < plot_k:
    #                         reservoir.append(entry)
    #                     else:
    #                         j = rng.randrange(sample_idx) if sample_idx > 0 else 0
    #                         if j < plot_k:
    #                             reservoir[j] = entry
        
    #     # torch.save(save_pred_target, model_out_dir / 'test_predictions.pth')
    #     mean_test_corr = (pearson_sum / pearson_count) if pearson_count > 0 else None
    #     print('Test set mean Pearson:', mean_test_corr)
    #     if mean_test_corr is not None:
    #         writer.add_scalar('test/mean_pearson', mean_test_corr, 0)

    #     # plotting section (reuse existing naming)
    #     TRACKS_5 = ['CS13','CS14','CS15','CS17','CS20']

    #     print('Plotting', len(reservoir), 'test samples to TensorBoard...')
    #     if plot_k > 0 and len(reservoir) > 0:
    #         for i, (seq_cpu, target_cpu, preds_cpu, interval, seq_info) in enumerate(reservoir):
    #             try:
    #                 ch = None
    #                 if preds_cpu.ndim == 3:
    #                     ch = preds_cpu.shape[1]
    #                 elif preds_cpu.ndim == 2:
    #                     ch = preds_cpu.shape[1] if preds_cpu.shape[0] > preds_cpu.shape[1] else preds_cpu.shape[1]
    #                 else:
    #                     ch = 1

    #                 names = TRACKS_5 if ch == 5 else [f'track_{k}' for k in range(ch)]

    #                 tracks = {}
    #                 for idx, name in enumerate(names[:ch]):
    #                     try:
    #                         if preds_cpu.ndim == 3:
    #                             pred_arr = preds_cpu[:, idx].numpy()
    #                             target_arr = target_cpu[:, idx].numpy()
    #                         elif preds_cpu.ndim == 2:
    #                             pred_arr = preds_cpu[:, idx].numpy()
    #                             target_arr = target_cpu[:, idx].numpy()
    #                         else:
    #                             pred_arr = preds_cpu.flatten().numpy()
    #                             target_arr = target_cpu.flatten().numpy()
    #                         tracks[name] = (pred_arr, target_arr)
    #                     except Exception:
    #                         continue

    #                 plot_interval = interval if (interval is not None) else (0, len(next(iter(tracks.values()))[0]) - 1)
    #                 fig = plot_tracks({k: v for k, v in tracks.items()}, plot_interval, seq_info)
    #                 writer.add_figure(f'test/pred_vs_target_{i}', fig, global_step=0)
    #                 plt.close(fig)
    #             except Exception:
    #                 pass
    #     writer.close()

    #     return
    
    best_val_loss = float('inf')
    global_step = 0
    for epoch in range(1, args.epochs + 1):
        train_loss, global_step = train_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            writer=writer,
            global_step_start=global_step,
            clip_grad_norm=args.clip_grad,
            print_every=args.print_every,
            log_every=args.log_every,
            log_grads=args.log_grads,
            grad_log_every=args.grad_log_every,
        )
        val_loss, mean_corr = validate(model, val_loader, device)

        # log epoch metrics
        writer.add_scalar('val/loss_epoch', val_loss, epoch)
        writer.add_scalar('train/loss_epoch', train_loss, epoch)
        if mean_corr is not None:
            writer.add_scalar('val/mean_pearson', mean_corr, epoch)

        # step scheduler on validation loss
        scheduler.step(val_loss)

        print(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")
        if mean_corr is not None:
            print(f"Mean Pearson per channel: {mean_corr}")

        # save checkpoint
        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict() if (scaler is not None) else {},
            'val_loss': val_loss
        }, model_out_dir, name=f'checkpoint_epoch_{epoch}.pt')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint({'model_state_dict': model.state_dict(), 'val_loss': val_loss}, model_out_dir, name='best.pt')
    
    # After training, run test inference with best checkpoint if test_loader exists
    if test_loader is not None:
        print('\n=== Running test inference with best checkpoint ===')
        best_path = model_out_dir / 'best.pt'
        if best_path.exists():
            print('Loading best checkpoint for test inference:', best_path)
            ckpt = torch.load(best_path, map_location=device)
            try:
                model.load_state_dict(ckpt['model_state_dict'])
            except Exception:
                model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
        else:
            print('Best checkpoint not found at', best_path, '; using current model weights for test inference')

        # Run test evaluation
        model.eval()
        sample_idx = 0
        pearson_sum = 0.0
        pearson_count = 0
        plot_k = max(0, int(getattr(args, 'test_plot_samples', 5)))
        plot_seed = int(getattr(args, 'test_plot_seed', 42))
        rng = random.Random(plot_seed)
        reservoir = []
        
        with torch.no_grad():
            for item in test_loader:
                seq, target, interval, seq_info = item

                if isinstance(seq, (list, tuple)):
                    seq = seq[0]

                seq = seq.to(device)
                target = target.to(device)

                preds = model(seq)
                pearson_vals = pearson_corr_coef(preds, target)

                if pearson_vals is not None:
                    for v in pearson_vals.cpu().tolist():
                        writer.add_scalar('test/pearson_sample', float(v), sample_idx)
                        sample_idx += 1
                    pearson_sum += float(pearson_vals.sum().cpu().item())
                    pearson_count += int(pearson_vals.numel())

                if plot_k > 0:
                    batch_size = preds.shape[0]
                    for i in range(batch_size):
                        # Extract interval and seq_info for each sample in the batch
                        sample_interval = None
                        sample_seq_info = None
                        try:
                            if isinstance(interval, (list, tuple)) and len(interval) > i:
                                sample_interval = interval[i]
                            elif not isinstance(interval, (list, tuple)):
                                sample_interval = interval
                            
                            if isinstance(seq_info, (list, tuple)) and len(seq_info) > i:
                                sample_seq_info = seq_info[i]
                            elif not isinstance(seq_info, (list, tuple)):
                                sample_seq_info = seq_info
                        except Exception:
                            sample_interval = (0, preds.shape[-1] - 1)
                            sample_seq_info = f'test_sample_{sample_idx + i}'

                        entry = (seq[i].detach().cpu().clone(), target[i].detach().cpu().clone(), preds[i].detach().cpu().clone(), sample_interval, sample_seq_info)
                        if len(reservoir) < plot_k:
                            reservoir.append(entry)
                        else:
                            j = rng.randrange(sample_idx) if sample_idx > 0 else 0
                            if j < plot_k:
                                reservoir[j] = entry
        
        mean_test_corr = (pearson_sum / pearson_count) if pearson_count > 0 else None
        print('Test set mean Pearson:', mean_test_corr)
        if mean_test_corr is not None:
            writer.add_scalar('test/mean_pearson', mean_test_corr, 0)

        # Plot test samples
        TRACKS_5 = ['CS13','CS14','CS15','CS17','CS20']
        print('Plotting', len(reservoir), 'test samples to TensorBoard...')
        if plot_k > 0 and len(reservoir) > 0:
            for i, (seq_cpu, target_cpu, preds_cpu, interval, seq_info) in enumerate(reservoir):
                try:
                    ch = None
                    if preds_cpu.ndim == 3:
                        ch = preds_cpu.shape[1]
                    elif preds_cpu.ndim == 2:
                        ch = preds_cpu.shape[1] if preds_cpu.shape[0] > preds_cpu.shape[1] else preds_cpu.shape[1]
                    else:
                        ch = 1

                    names = TRACKS_5 if ch == 5 else [f'track_{k}' for k in range(ch)]

                    tracks = {}
                    for idx, name in enumerate(names[:ch]):
                        try:
                            if preds_cpu.ndim == 3:
                                pred_arr = preds_cpu[:, idx].numpy()
                                target_arr = target_cpu[:, idx].numpy()
                            elif preds_cpu.ndim == 2:
                                pred_arr = preds_cpu[:, idx].numpy()
                                target_arr = target_cpu[:, idx].numpy()
                            else:
                                pred_arr = preds_cpu.flatten().numpy()
                                target_arr = target_cpu.flatten().numpy()
                            tracks[name] = (pred_arr, target_arr)
                        except Exception:
                            continue

                    plot_interval = interval if (interval is not None) else (0, len(next(iter(tracks.values()))[0]) - 1)
                    fig = plot_tracks({k: v for k, v in tracks.items()}, plot_interval, seq_info)
                    writer.add_figure(f'test/pred_vs_target_{i}', fig, global_step=0)
                    plt.close(fig)
                except Exception:
                    pass
    
    writer.close()
    


if __name__ == '__main__':
    main()
