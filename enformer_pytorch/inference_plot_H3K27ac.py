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
from data import str_to_one_hot


def plot_tracks_batch(tracks_dict, seq_names, target_data, seq_mut_infos=None, height=1.5):
    """
    Plot multiple sequences (reference + alterations) for each track on the same figure
    with enhanced visualization to show differences clearly
    """

    # Compute global y-limits over all targets and predictions (full length)
    items = list(tracks_dict.items())
    n = len(items)
    fig, axes = plt.subplots(n, 2, figsize=(30, height * n), sharex=True)
    if n == 1:
        axes = axes.reshape(1, -1)

    # Color and style setup
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(seq_names))))
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']

    # find reference index
    ref_idx = 0
    for i, name in enumerate(seq_names):
        try:
            if 'ref' in name.lower() or 'reference' in name.lower():
                ref_idx = i
                break
        except Exception:
            continue

    all_values_orig = []
    all_values_diff = []
    for track_idx, (track_name, pred_arrays) in enumerate(items):
        # target
        if len(target_data.shape) > 1:
            target_for_track = target_data[:, track_idx]
        else:
            target_for_track = target_data[track_idx]
        if hasattr(target_for_track, '__len__') and len(target_for_track) > 1:
            all_values_orig.extend(np.asarray(target_for_track).flatten())
        else:
            all_values_orig.append(target_for_track)

        # predictions
        ref_pred = pred_arrays[ref_idx] if pred_arrays else None
        for i, pred_arr in enumerate(pred_arrays):
            if hasattr(pred_arr, '__len__') and len(pred_arr) > 1:
                all_values_orig.extend(np.asarray(pred_arr).flatten())
                # diffs vs ref
                if i != ref_idx and ref_pred is not None and hasattr(ref_pred, '__len__') and len(ref_pred) > 1:
                    min_len = min(len(pred_arr), len(ref_pred))
                    diff_full = np.asarray(pred_arr[:min_len]) - np.asarray(ref_pred[:min_len])
                    all_values_diff.extend(diff_full.flatten())
            else:
                all_values_orig.append(pred_arr)

    # y-limits for original
    if all_values_orig:
        y_min_orig = np.min(all_values_orig)
        y_max_orig = np.max(all_values_orig)
        y_padding_orig = (y_max_orig - y_min_orig) * 0.05 if y_max_orig != y_min_orig else 0.1
        y_min_orig_padded = y_min_orig - y_padding_orig
        y_max_orig_padded = y_max_orig + y_padding_orig
    else:
        y_min_orig_padded, y_max_orig_padded = 0, 1

    # y-limits for diffs
    if all_values_diff:
        y_min_diff = np.min(all_values_diff)
        y_max_diff = np.max(all_values_diff)
        y_padding_diff = (y_max_diff - y_min_diff) * 0.05 if y_max_diff != y_min_diff else 0.1
        y_min_diff_padded = y_min_diff - y_padding_diff
        y_max_diff_padded = y_max_diff + y_padding_diff
    else:
        y_min_diff_padded, y_max_diff_padded = -1, 1

    for track_idx, (track_name, pred_arrays) in enumerate(items):
        ax_orig = axes[track_idx, 0]
        ax_diff = axes[track_idx, 1]

        # target (full length)
        if len(target_data.shape) > 1:
            target_for_track = target_data[:, track_idx]
        else:
            target_for_track = target_data[track_idx]
        if hasattr(target_for_track, '__len__') and len(target_for_track) > 1:
            x_t = np.arange(len(target_for_track))
            ax_orig.plot(x_t, np.asarray(target_for_track), label='target', color='black', linewidth=3, alpha=0.9)
        else:
            ax_orig.axhline(y=target_for_track, label='target', color='black', linewidth=3, alpha=0.9)

        # plot predictions (full length) and compute diffs vs ref
        ref_pred = None
        if len(pred_arrays) > 0:
            ref_pred = pred_arrays[ref_idx]

        for i, (pred_arr, seq_name) in enumerate(zip(pred_arrays, seq_names)):
            try:
                if hasattr(pred_arr, '__len__') and len(pred_arr) > 1:
                    x_p = np.arange(len(pred_arr))
                    ax_orig.plot(x_p, np.asarray(pred_arr), label=seq_name, color=colors[i % len(colors)],
                                 linestyle=line_styles[i % len(line_styles)], linewidth=(3 if i==ref_idx else 2), alpha=(0.95 if i==ref_idx else 0.7))

                    # difference vs ref
                    if i != ref_idx and ref_pred is not None and hasattr(ref_pred, '__len__') and len(ref_pred) > 1:
                        min_len = min(len(pred_arr), len(ref_pred))
                        diff = np.asarray(pred_arr[:min_len]) - np.asarray(ref_pred[:min_len])
                        ax_diff.plot(np.arange(min_len), diff, label=f'{seq_name} vs ref', color=colors[i % len(colors)],
                                     linestyle=line_styles[i % len(line_styles)], linewidth=1.5, alpha=0.8)
                else:
                    # scalar
                    ax_orig.axhline(y=pred_arr, label=seq_name, color=colors[i % len(colors)],
                                   linestyle=line_styles[i % len(line_styles)], alpha=0.7)
            except Exception as e:
                print(f"Error plotting {seq_name} for {track_name}: {e}")
                continue

        # add zero line to diff axis
        ax_diff.axhline(y=0, color='gray', linestyle='-', alpha=0.5)

        # Mark center bin for the mutation (simplified: mutation is at center bin)
        try:
            # determine number of bins from any prediction array
            n_bins = None
            for p in pred_arrays:
                if hasattr(p, '__len__') and len(p) > 1:
                    n_bins = len(p)
                    break
            if n_bins is not None:
                center_bin = n_bins // 2
                # draw a dashed vertical line at the center bin for each sequence (no text labels)
                for j in range(len(seq_names)):
                    ax_orig.axvline(x=center_bin, color=colors[j % len(colors)], linestyle=':', linewidth=1.2, alpha=0.9)
                    ax_diff.axvline(x=center_bin, color=colors[j % len(colors)], linestyle=':', linewidth=1.2, alpha=0.9)
        except Exception:
            pass

        # Set consistent y-axis limits and grid
        ax_orig.set_ylim(y_min_orig_padded, y_max_orig_padded)
        ax_diff.set_ylim(y_min_diff_padded, y_max_diff_padded)

        ax_orig.set_title(f'{track_name} - Original Predictions')
        ax_diff.set_title(f'{track_name} - Differences from Reference')

        if track_idx == 0:
            ax_orig.legend(loc='upper right', fontsize=10)
            ax_diff.legend(loc='upper right', fontsize=10)

        ax_orig.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        ax_diff.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
        ax_orig.set_axisbelow(True)
        ax_diff.set_axisbelow(True)

        try:
            sns.despine(top=True, right=True, bottom=True, ax=ax_orig)
            sns.despine(top=True, right=True, bottom=True, ax=ax_diff)
        except Exception:
            pass

    plt.tight_layout()
    return fig

def main():
    TRACKS_5 = ['CS13','CS14','CS15','CS17','CS20']


    results_folder = '/home/chent9/projects/enformer-pytorch/results'
    data_folder = "/home/chent9/Datasets/impute_H3K27ac_downloads"
    pt_name = 'snp_mutation_IRF6_H3K27ac.pt'  
    num_tracks = 5
    model_name = 'H3K27ac_batchsize_4_lr1e-5_clip0.5_noamp_chromsplit811_log1p_1'

    data_pt = torch.load(os.path.join(data_folder, pt_name), weights_only=False, map_location='cpu')

    pretrained = 'EleutherAI/enformer-official-rough'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enformer = from_pretrained(pretrained)
    model = HeadAdapterWrapper(enformer=enformer, num_tracks=num_tracks, post_transformer_embed=False).to(device)
    model_path = os.path.join(results_folder, model_name, 'model', 'best.pt')
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Create output directory for plots
    plot_output_dir = os.path.join(results_folder, model_name, 'plots_mutation_IRF6_H3K27ac')
    os.makedirs(plot_output_dir, exist_ok=True)

    data_with_preds = []
    with torch.no_grad():
        for batch_idx in range(len(data_pt)):
            data_i = data_pt[batch_idx]
            # load target as provided; orientation may vary, normalize after inference
            target = torch.from_numpy(data_i['target']).float()
            all_seq = data_i['seqs']
            rsid = str(data_i['rs_id'])

            batch_seqs = []
            seq_names = []
            seq_strs = []
            for seq_name, seq_str in all_seq.items():
                if 'REVCOMP' in seq_name:
                    continue
                seq_one_hot = str_to_one_hot(seq_str)
                batch_seqs.append(seq_one_hot)
                seq_names.append(seq_name)
                seq_strs.append(seq_str)
                
            # Stack the batch sequences into a tensor
            batch_seqs = torch.stack(batch_seqs)  # Shape: [batch_size, seq_len, 4]

            # Determine reference sequence string (try to find 'ref' in name, else use first)
            ref_idx_local = 0
            for i, name in enumerate(seq_names):
                if 'ref' in name.lower() or 'reference' in name.lower():
                    ref_idx_local = i
                    break
            ref_seq_str = seq_strs[ref_idx_local]

            # attempt to get genomic start and chromosome from data_i
            genomic_start = None
            chrom_name = None
            if isinstance(data_i, dict):
                if 'interval' in data_i and isinstance(data_i['interval'], (list, tuple)) and len(data_i['interval']) >= 1:
                    try:
                        genomic_start = int(data_i['interval'][0])
                    except Exception:
                        genomic_start = None
                if 'chrom' in data_i:
                    chrom_name = str(data_i['chrom'])
                if 'seq_info' in data_i and isinstance(data_i['seq_info'], str):
                    try:
                        parts = data_i['seq_info'].split(':')
                        if len(parts) > 0:
                            chrom_name = parts[0]
                        if len(parts) > 1:
                            rng = parts[1].split('-')[0]
                            genomic_start = int(rng)
                    except Exception:
                        pass

            context_len = batch_seqs.shape[1] if batch_seqs.ndim > 1 else None

            # Find a primary mutation position (from any alt seq) to use for naming the ref
            primary_pos = None
            primary_ref = None
            primary_alt = None
            for seq_str in seq_strs:
                try:
                    L = min(len(seq_str), len(ref_seq_str))
                    diffs = [idx for idx in range(L) if seq_str[idx] != ref_seq_str[idx]]
                    if diffs:
                        center_idx = L // 2
                        diffs_sorted = sorted(diffs, key=lambda x: abs(x - center_idx))
                        p = diffs_sorted[0]
                        primary_pos = p
                        primary_ref = ref_seq_str[p]
                        primary_alt = seq_str[p]
                        break
                except Exception:
                    continue

            # Simplify naming: construct a single mutation label (from any alt) and apply to all sequences
            label_base = None
            if primary_pos is not None:
                try:
                    p = primary_pos
                    ref_base = primary_ref
                    alt_base = primary_alt
                    if genomic_start is not None:
                        genomic_pos = genomic_start + p
                    else:
                        if context_len is not None:
                            genomic_pos = (p - (context_len // 2))
                        else:
                            genomic_pos = p
                    if chrom_name is not None:
                        label_base = f"{chrom_name}:{genomic_pos}_{ref_base}>{alt_base}"
                    else:
                        label_base = f"{genomic_pos}_{ref_base}>{alt_base}"
                except Exception:
                    label_base = None

            # we no longer build seq_mut_infos (plotting will use center bin)
            seq_mut_infos = None

            # Run model inference and prepare data for plotting
            batch_seqs = batch_seqs.to(device)
            target = target.to(device)
            preds = model(batch_seqs)  # Shape: [batch_size, num_bins, num_tracks]

            # Move to CPU for plotting
            preds_cpu = preds.detach().cpu().numpy()
            target_cpu = target.detach().cpu().numpy()

            preds_dict = {}
            for i in range(len(seq_names)):
                preds_dict[seq_names[i]] = preds_cpu[i]

            data_i['preds'] = preds_dict
            data_with_preds.append(data_i)

            # Ensure target has shape [num_bins, num_tracks]. Model preds shape: [batch, num_bins, num_tracks]
            try:
                if target_cpu.ndim == 2:
                    num_bins_pred = preds_cpu.shape[1]
                    num_tracks_pred = preds_cpu.shape[2]
                    if target_cpu.shape[0] != num_bins_pred and target_cpu.shape[1] == num_bins_pred:
                        # transpose if orientation swapped
                        target_cpu = target_cpu.T
            except Exception:
                pass

            # Determine number of tracks and get track names
            num_tracks = preds_cpu.shape[2] if len(preds_cpu.shape) > 2 else preds_cpu.shape[1]
            track_names = TRACKS_5

            # Organize data for plotting: each track gets predictions from all sequences
            tracks_dict = {}
            for track_idx, track_name in enumerate(track_names):
                pred_arrays_for_track = []
                for seq_idx in range(len(seq_names)):
                    if len(preds_cpu.shape) > 2:
                        pred_arr = preds_cpu[seq_idx, :, track_idx]  # [num_bins]
                    else:
                        pred_arr = preds_cpu[seq_idx, track_idx]  # scalar or 1D
                    pred_arrays_for_track.append(pred_arr)
                tracks_dict[track_name] = pred_arrays_for_track

            # Target data for plotting (same for all tracks in terms of structure)
            if len(target_cpu.shape) > 1:
                target_for_plotting = target_cpu  # [num_bins, num_tracks]
            else:
                target_for_plotting = target_cpu  # [num_tracks]

            # Create the plot for this batch (pass mutation infos)
            fig = plot_tracks_batch(tracks_dict, seq_names, target_for_plotting, seq_mut_infos=seq_mut_infos)
            
            # Generate filename for the plot
            gene_name = seq_names[0].split('|')[0]  # Extract gene name from first sequence name
            plot_filename = f"plot_{batch_idx:04d}_{rsid}_{gene_name}.png"
            plot_path = os.path.join(plot_output_dir, plot_filename)
            
            # Save the plot
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Saved batch plot: {plot_path}")
            
            plt.close(fig)
        
        torch.save(data_with_preds, os.path.join(results_folder, model_name, 'data_with_predictions.pt'))


if __name__ == '__main__':
    main()