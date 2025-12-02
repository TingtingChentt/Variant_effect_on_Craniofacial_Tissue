import torch
import os
import argparse
from data import str_to_one_hot
from torch.utils.data import DataLoader
from enformer_pytorch import from_pretrained
from enformer_pytorch.finetune import HeadAdapterWrapper
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
import numpy as np
import seaborn as sns


import seaborn as sns

def clean_seq_name(seq_name):
    """
    Clean sequence name by removing gene ID prefix.
    Example: 'ENSG00000199466|Y_RNA|chr10:124724553C>T|ALT' -> 'Y_RNA|chr10:124724553C>T|ALT'
    """
    if '|' in seq_name:
        parts = seq_name.split('|')
        if len(parts) > 1:
            # Remove the first part (gene ID) and join the rest
            return '|'.join(parts[1:])
    return seq_name

def plot_tracks_batch(tracks_dict, seq_names, target_data, height=1.5):
    """
    Plot multiple sequences (reference + alterations) for each track on the same figure
    with enhanced visualization to show differences clearly
    """
    # Clean the sequence names for display
    cleaned_seq_names = [clean_seq_name(name) for name in seq_names]
    
    items = list(tracks_dict.items())
    n = len(items)
    
    # Create subplots: 2 columns (original + difference plot)
    fig, axes = plt.subplots(n, 2, figsize=(30, height * n), sharex=True)
    if n == 1:
        axes = axes.reshape(1, -1)
    
    # Color palette for different sequences
    colors = plt.cm.tab10(np.linspace(0, 1, len(cleaned_seq_names)))
    line_styles = ['-', '--', '-.', ':', '-', '--', '-.', ':']  # Different line styles
    
    # Find reference sequence (usually first or contains 'ref' in name)
    ref_idx = 0
    for i, name in enumerate(cleaned_seq_names):
        if 'ref' in name.lower() or 'reference' in name.lower():
            ref_idx = i
            break
    
    # Calculate global y-axis limits for original plots (center 60 bins only)
    all_values_orig = []
    all_values_diff = []
    center_bins = 60
    
    for track_idx, (track_name, pred_arrays) in enumerate(items):
        # Extract target for this specific track
        if len(target_data.shape) > 1:
            target_for_track = target_data[:, track_idx]
        else:
            target_for_track = target_data[track_idx]
        
        # Add target values (trimmed to center)
        if hasattr(target_for_track, '__len__') and len(target_for_track) > 1:
            total_length = len(target_for_track)
            if total_length > center_bins:
                start_idx = (total_length - center_bins) // 2
                end_idx = start_idx + center_bins
                target_center = target_for_track[start_idx:end_idx]
            else:
                target_center = target_for_track
            all_values_orig.extend(target_center.flatten())
        else:
            all_values_orig.append(target_for_track)
        
        ref_pred = pred_arrays[ref_idx] if pred_arrays else None
        
        # Process all predictions for this track
        for i, pred_arr in enumerate(pred_arrays):
            if hasattr(pred_arr, '__len__') and len(pred_arr) > 1:
                total_length = len(pred_arr)
                if total_length > center_bins:
                    start_idx = (total_length - center_bins) // 2
                    end_idx = start_idx + center_bins
                    pred_center = pred_arr[start_idx:end_idx]
                else:
                    pred_center = pred_arr
                all_values_orig.extend(pred_center.flatten())
                
                # Calculate differences for difference plot scaling
                if i != ref_idx and ref_pred is not None:
                    if hasattr(ref_pred, '__len__') and len(ref_pred) > 1:
                        if len(ref_pred) > center_bins:
                            ref_center = ref_pred[start_idx:end_idx]
                        else:
                            ref_center = ref_pred
                        diff_center = pred_center - ref_center
                        all_values_diff.extend(diff_center.flatten())
            else:
                all_values_orig.append(pred_arr)
                if i != ref_idx and ref_pred is not None:
                    diff = pred_arr - ref_pred
                    all_values_diff.append(diff)
    
    # Calculate y-axis limits for original plots
    if all_values_orig:
        y_min_orig = np.min(all_values_orig)
        y_max_orig = np.max(all_values_orig)
        y_range_orig = y_max_orig - y_min_orig
        y_padding_orig = y_range_orig * 0.05
        y_min_orig_padded = y_min_orig - y_padding_orig
        y_max_orig_padded = y_max_orig + y_padding_orig
    else:
        y_min_orig_padded, y_max_orig_padded = 0, 1
    
    # Calculate y-axis limits for difference plots
    if all_values_diff:
        y_min_diff = np.min(all_values_diff)
        y_max_diff = np.max(all_values_diff)
        y_range_diff = y_max_diff - y_min_diff
        y_padding_diff = y_range_diff * 0.05
        y_min_diff_padded = y_min_diff - y_padding_diff
        y_max_diff_padded = y_max_diff + y_padding_diff
    else:
        y_min_diff_padded, y_max_diff_padded = -1, 1
    
    for track_idx, (track_name, pred_arrays) in enumerate(items):
        # Get the corresponding axes
        ax_orig = axes[track_idx, 0]  # Original predictions
        ax_diff = axes[track_idx, 1]  # Difference from reference
        
        # Extract target for this specific track
        if len(target_data.shape) > 1:
            target_for_track = target_data[:, track_idx]
        else:
            target_for_track = target_data[track_idx]
        
        # Get reference prediction for difference calculation
        ref_pred = pred_arrays[ref_idx] if pred_arrays else None
        
        # Calculate center region for trimming
        if hasattr(target_for_track, '__len__') and len(target_for_track) > 1:
            total_length = len(target_for_track)
            if total_length > center_bins:
                start_idx = (total_length - center_bins) // 2
                end_idx = start_idx + center_bins
            else:
                start_idx = 0
                end_idx = total_length
        else:
            start_idx = 0
            end_idx = 1
        
        # Plot target on original axis (trimmed)
        if hasattr(target_for_track, '__len__') and len(target_for_track) > 1:
            x = np.arange(start_idx, end_idx)
            target_trimmed = target_for_track[start_idx:end_idx]
            ax_orig.plot(x, target_trimmed, label='target', color='black', linewidth=3, alpha=0.9)
        else:
            x = np.arange(start_idx, end_idx)
            ax_orig.axhline(y=target_for_track, label='target', color='black', linewidth=3, alpha=0.9)
        
        # Plot predictions with different styles and offsets for visibility
        for i, (pred_arr, seq_name) in enumerate(zip(pred_arrays, cleaned_seq_names)):
            try:
                if hasattr(pred_arr, '__len__') and len(pred_arr) > 1:
                    # Trim to center region
                    pred_trimmed = pred_arr[start_idx:end_idx]
                    x = np.arange(start_idx, end_idx)
                    
                    # Original plot with different line styles
                    line_style = line_styles[i % len(line_styles)]
                    line_width = 3 if i == ref_idx else 2
                    alpha = 0.9 if i == ref_idx else 0.7
                    
                    # Show labels for all sequences in original plots
                    ax_orig.plot(x, pred_trimmed, label=seq_name, color=colors[i], 
                               linestyle=line_style, linewidth=line_width, alpha=alpha)
                    
                    # Difference plot (only if not reference)
                    if i != ref_idx and ref_pred is not None:
                        ref_trimmed = ref_pred[start_idx:end_idx]
                        diff = pred_trimmed - ref_trimmed
                        
                        # Show labels for all mutations in difference plots
                        ax_diff.plot(x, diff, label=f'{seq_name} vs ref', 
                                   color=colors[i], linestyle=line_style, linewidth=2, alpha=0.8)
                        
                        # Add zero line for reference
                        ax_diff.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
                    
                else:
                    # Handle scalar predictions
                    ax_orig.axhline(y=pred_arr, label=seq_name, color=colors[i], 
                                  linestyle=line_styles[i % len(line_styles)], alpha=0.7)
                    
                    if i != ref_idx and ref_pred is not None:
                        diff = pred_arr - ref_pred
                        ax_diff.axhline(y=diff, label=f'{seq_name} vs ref', 
                                      color=colors[i], alpha=0.7)
                        ax_diff.axhline(y=0, color='gray', linestyle='-', alpha=0.5)
                        
            except Exception as e:
                print(f"Error plotting {seq_name} for {track_name}: {e}")
                continue
        
        # Set consistent y-axis limits and add grid
        ax_orig.set_ylim(y_min_orig_padded, y_max_orig_padded)
        ax_diff.set_ylim(y_min_diff_padded, y_max_diff_padded)
        
        # Set titles and labels
        ax_orig.set_title(f'{track_name} - Original Predictions')
        ax_diff.set_title(f'{track_name} - Differences from Reference')
        
        # Show legend only for the first track (since sequence names are the same across tracks)
        if track_idx == 0:
            ax_orig.legend(loc='upper right', fontsize=10)
            ax_diff.legend(loc='upper right', fontsize=10)
        
        # Add grid for better readability (y-axis only)
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
    pt_name = 'gse_chr_holdout_test_enhancer_DNM_seqs_craniofacial_genes.pt'
    num_tracks = 6
    model_name = 'gse_batchsize_4_test_grad_lr1e-5_clip0.5_noamp_chromsplit811'
    # pt_name = 'single_chr_holdout_test_all_DNM_seqs_craniofacial_genes.pt'
    
    # num_tracks = 48
    # model_name = 'single_batchsize_4_test_grad_lr1e-5_clip0.5_noamp_chromsplit811'


    pretrained = 'EleutherAI/enformer-official-rough'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    data_pt = torch.load(os.path.join(pt_path, pt_name), weights_only=False, map_location='cpu')
    enformer = from_pretrained(pretrained)
    model = HeadAdapterWrapper(enformer=enformer, num_tracks=num_tracks, post_transformer_embed=False).to(device)
    model_path = os.path.join(results_folder, model_name, 'model', 'best.pt')
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Create output directory for plots
    plot_output_dir = os.path.join(results_folder, model_name, 'plots_enhancer_DNM_seqs_sameYscale_trimX')
    os.makedirs(plot_output_dir, exist_ok=True)

    with torch.no_grad():
        for batch_idx in range(len(data_pt)):
            data_i = data_pt[batch_idx]
            target = torch.from_numpy(data_i['target'].T).float()  # Shape: [num_bins, num_tracks]
            all_seq = data_i['seq']

            batch_seqs = []
            seq_names = []
            for seq_name, seq_str in all_seq.items():
                if 'REVCOMP' in seq_name:
                    continue
                seq_one_hot = str_to_one_hot(seq_str)
                batch_seqs.append(seq_one_hot)
                seq_names.append(seq_name)
                
            # Stack the batch sequences into a tensor
            batch_seqs = torch.stack(batch_seqs)  # Shape: [batch_size, seq_len, 4]

            batch_seqs = batch_seqs.to(device)
            target = target.to(device)
            preds = model(batch_seqs)  # Shape: [batch_size, num_bins, num_tracks]

            # Move to CPU for plotting
            preds_cpu = preds.detach().cpu().numpy()
            target_cpu = target.detach().cpu().numpy()

            # Determine number of tracks and get track names
            num_tracks = preds_cpu.shape[2] if len(preds_cpu.shape) > 2 else preds_cpu.shape[1]
            track_names = TRACKS_48[:num_tracks] if num_tracks >= 48 else TRACKS_6[:num_tracks] if num_tracks == 6 else [f'track_{k}' for k in range(num_tracks)]

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

            # Create the plot for this batch
            fig = plot_tracks_batch(tracks_dict, seq_names, target_for_plotting)
            
            # Generate filename for the plot
            gene_name = seq_names[0].split('|')[1]  # Extract gene name from first sequence name
            plot_filename = f"plot_{batch_idx:04d}_{gene_name}.png"
            plot_path = os.path.join(plot_output_dir, plot_filename)
            
            # Save the plot
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            print(f"Saved batch plot: {plot_path}")
            
            plt.close(fig)

if __name__ == '__main__':
    main()