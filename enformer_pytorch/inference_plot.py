import torch
import os
import argparse
from data import GenomeIntervalDataset
from torch.utils.data import DataLoader
from enformer_pytorch import from_pretrained
from enformer_pytorch.finetune import HeadAdapterWrapper
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
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

    # Calculate global y-axis limits across all tracks (center 100 bins only)
    all_values = []
    center_bins = 100
    
    for title, y in items:
        if isinstance(y, (list, tuple)):
            for series in y:
                try:
                    arr = np.array(series)
                    # Extract center 100 bins
                    total_length = len(arr)
                    if total_length > center_bins:
                        start_idx = (total_length - center_bins) // 2
                        end_idx = start_idx + center_bins
                        arr_center = arr[start_idx:end_idx]
                    else:
                        arr_center = arr
                    all_values.extend(arr_center.flatten())
                except Exception:
                    pass
        else:
            try:
                arr = np.array(y)
                # Extract center 100 bins
                total_length = len(arr)
                if total_length > center_bins:
                    start_idx = (total_length - center_bins) // 2
                    end_idx = start_idx + center_bins
                    arr_center = arr[start_idx:end_idx]
                else:
                    arr_center = arr
                all_values.extend(arr_center.flatten())
            except Exception:
                pass
    
    if all_values:
        y_min = np.min(all_values)
        y_max = np.max(all_values)
        # Add some padding (5% on each side)
        y_range = y_max - y_min
        y_padding = y_range * 0.05
        y_min_padded = y_min - y_padding
        y_max_padded = y_max + y_padding
    else:
        y_min_padded, y_max_padded = 0, 1

    labels = ['pred', 'target']
    for ax, (title, y) in zip(axes, items):
        # y may be a single 1D array or a sequence of arrays to overlay
        if isinstance(y, (list, tuple)):
            # take length from first series
            first = y[0]
            s, e = _interval_start_end(interval, len(first))
            
            # Calculate center 100 bins
            total_length = len(first)
            center_bins = 100
            if total_length > center_bins:
                start_idx = (total_length - center_bins) // 2
                end_idx = start_idx + center_bins
            else:
                start_idx = 0
                end_idx = total_length
            
            # Create x coordinates for center region only
            x_full = np.linspace(s, e, num=total_length)
            x = x_full[start_idx:end_idx]
            
            for idx, series in enumerate(y):
                try:
                    arr = np.array(series)
                    # Trim to center 100 bins
                    arr_trimmed = arr[start_idx:end_idx]
                    ax.plot(x, arr_trimmed, label=labels[idx])
                except Exception:
                    pass
            ax.legend()
        else:
            arr = np.array(y)
            s, e = _interval_start_end(interval, len(arr))
            
            # Calculate center 100 bins
            total_length = len(arr)
            center_bins = 100
            if total_length > center_bins:
                start_idx = (total_length - center_bins) // 2
                end_idx = start_idx + center_bins
            else:
                start_idx = 0
                end_idx = total_length
            
            # Create x coordinates for center region only
            x_full = np.linspace(s, e, num=total_length)
            x = x_full[start_idx:end_idx]
            arr_trimmed = arr[start_idx:end_idx]
            
            ax.fill_between(x, arr_trimmed)
        
        # Set the same y-axis limits for all subplots
        ax.set_ylim(y_min_padded, y_max_padded)
        
        # Add grid for better value reading
        ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)  # Ensure grid is behind the plot lines
        
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

def main():
    TRACKS_48 = [
            'mesenchyme_CS12','mesenchyme_CS13','mesenchyme_CS14','mesenchyme_CS16','mesenchyme_CS17','mesenchyme_CS20',
            'ectoderm_CS12','ectoderm_CS13','ectoderm_CS14','ectoderm_CS16','ectoderm_CS17','ectoderm_CS20',
            'muscle_CS12','muscle_CS13','muscle_CS14','muscle_CS16','muscle_CS17','muscle_CS20',
            'erythrocytes_CS12','erythrocytes_CS13','erythrocytes_CS14','erythrocytes_CS16','erythrocytes_CS17','erythrocytes_CS20',
            'endothelium_CS12','endothelium_CS13','endothelium_CS14','endothelium_CS16','endothelium_CS17','endothelium_CS20',
            'cncc_CS12','cncc_CS13','cncc_CS14','cncc_CS16','cncc_CS17','cncc_CS20',
            'immune_CS12','immune_CS13','immune_CS14','immune_CS16','immune_CS17','immune_CS20',
            'progenitor_CS12','progenitor_CS13','progenitor_CS14','progenitor_CS16','progenitor_CS17','progenitor_CS20'
        ]
    TRACKS_6 = ['NCC','CS13','CS14','CS15','CS17','CS22']


    results_folder = '/home/chent9/projects/enformer-pytorch/results'
    pt_path = '/home/chent9/projects/enformer-pytorch/data'
    pt_name = 'gse_chr_holdout_test_cranioficial_genes.pt'
    num_tracks = 6
    model_name = 'gse_batchsize_4_test_grad_lr1e-5_clip0.5_noamp_chromsplit811'
    # pt_name = 'single_chr_holdout_test_cranioficial_genes.pt'
    
    # num_tracks = 48
    # model_name = 'single_batchsize_4_test_grad_lr1e-5_clip0.5_noamp_chromsplit811'

    fasta_file = '/home/chent9/projects/enformer-tf/data/genome.fa'
    context_length = 196608
    test = True
    pretrained = 'EleutherAI/enformer-official-rough'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    data_pt = torch.load(os.path.join(pt_path, pt_name), weights_only=False, map_location='cpu')

    ds = GenomeIntervalDataset(label_data=data_pt, fasta_file=fasta_file, context_length=context_length, test=test)
    test_loader = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)


    enformer = from_pretrained(pretrained)
    model = HeadAdapterWrapper(enformer=enformer, num_tracks=num_tracks, post_transformer_embed=False).to(device)
    model_path = os.path.join(results_folder, model_name, 'model', 'best.pt')
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Create output directory for plots
    plot_output_dir = os.path.join(results_folder, model_name, 'plots_sameYscale_trimX')
    os.makedirs(plot_output_dir, exist_ok=True)

    plots = []

    with torch.no_grad():
        for item in test_loader:
            seq, target, interval, seq_info = item

            if isinstance(seq, (list, tuple)):
                seq = seq[0]

            seq = seq.to(device)
            target = target.to(device)

            preds = model(seq)

            batch_size = preds.shape[0]
            for i in range(batch_size):
                sample_interval = None
                try:
                    if isinstance(interval, (list, tuple)):
                        sample_interval = interval[i]
                    else:
                        sample_interval = interval
                except Exception:
                    sample_interval = None

                entry = (seq[i].detach().cpu().clone(), target[i].detach().cpu().clone(), preds[i].detach().cpu().clone(), sample_interval, seq_info[i])
                plots.append(entry)
    
    for i, (seq_cpu, target_cpu, preds_cpu, interval, seq_info) in enumerate(plots):
        try:
            ch = None
            if preds_cpu.ndim == 3:
                ch = preds_cpu.shape[1]
            elif preds_cpu.ndim == 2:
                ch = preds_cpu.shape[1] if preds_cpu.shape[0] > preds_cpu.shape[1] else preds_cpu.shape[1]
            else:
                ch = 1

            names = TRACKS_48 if ch >= 48 else TRACKS_6 if ch == 6 else [f'track_{k}' for k in range(ch)]

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
            
            # Generate filename for the plot
            safe_seq_info = str(seq_info).replace(':', '_').replace('/', '_').replace(' ', '_') if seq_info else f"sample_{i}"
            plot_filename = f"plot_{i:04d}_{safe_seq_info}.png"
            plot_path = os.path.join(plot_output_dir, plot_filename)
            
            # Save the plot
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot: {plot_path}")
            
            plt.close(fig)
        except Exception:
            pass



if __name__ == '__main__':
    main()