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


def plot_tracks_batch(tracks_dict, seq_names, target_data, height=1.5):
    """
    Plot multiple sequences (reference + alterations) for each track on the same figure
    tracks_dict: dict mapping track_name -> list of prediction arrays (one per sequence)
    seq_names: list of sequence names for labeling
    target_data: target array [num_bins, num_tracks] or [num_tracks]
    """
    items = list(tracks_dict.items())
    n = len(items)
    fig, axes = plt.subplots(n, 1, figsize=(20, height * n), sharex=True)
    if n == 1:
        axes = [axes]
    
    # Color palette for different sequences
    colors = plt.cm.tab10(np.linspace(0, 1, len(seq_names)))
    
    for track_idx, (track_name, pred_arrays) in enumerate(items):
        # Get the corresponding axis
        ax = axes[track_idx]
        
        # Extract target for this specific track
        if len(target_data.shape) > 1:
            target_for_track = target_data[:, track_idx]  # [num_bins]
        else:
            target_for_track = target_data[track_idx]  # scalar
        
        # Plot target first (as reference)
        if hasattr(target_for_track, '__len__') and len(target_for_track) > 1:
            x = np.arange(len(target_for_track))
            ax.plot(x, target_for_track, label='target', color='black', linewidth=2, alpha=0.8)
        else:
            # Handle scalar targets
            x = np.arange(len(pred_arrays[0]) if pred_arrays else 1)
            ax.axhline(y=target_for_track, label='target', color='black', linewidth=2, alpha=0.8)
        
        # Plot predictions for each sequence
        for i, (pred_arr, seq_name) in enumerate(zip(pred_arrays, seq_names)):
            try:
                if hasattr(pred_arr, '__len__') and len(pred_arr) > 1:
                    x = np.arange(len(pred_arr))
                    ax.plot(x, pred_arr, label=seq_name, color=colors[i], alpha=0.7)
                else:
                    # Handle scalar predictions
                    ax.axhline(y=pred_arr, label=seq_name, color=colors[i], alpha=0.7)
            except Exception as e:
                print(f"Error plotting {seq_name} for {track_name}: {e}")
                continue
        
        ax.set_title(track_name)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        try:
            sns.despine(top=True, right=True, bottom=True)
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
    # pt_name = 'single_chr_holdout_test_cranioficial_genes.pt'
    
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
    plot_output_dir = os.path.join(results_folder, model_name, 'plots_enhancer_DNM_seqs')
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