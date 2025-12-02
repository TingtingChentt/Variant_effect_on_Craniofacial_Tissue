import torch
import os
import argparse
import pandas as pd
import numpy as np
from data import str_to_one_hot
from torch.utils.data import DataLoader
from enformer_pytorch import from_pretrained
from enformer_pytorch.finetune import HeadAdapterWrapper


def calculate_enformer_scores_organized(tracks_dict, seq_names, target_data, gene_name):
    """
    Calculate Enformer scores with organized format: one reference row + mutation rows per gene
    Returns a list of dictionaries with reference first, then mutations
    """
    results = []
    
    # Find reference sequence (usually first or contains 'ref' in name)
    ref_idx = 0
    for i, name in enumerate(seq_names):
        if 'ref' in name.lower() or 'reference' in name.lower() or 'original' in name.lower():
            ref_idx = i
            break
    
    ref_seq_name = seq_names[ref_idx]
    
    for track_idx, (track_name, pred_arrays) in enumerate(tracks_dict.items()):
        # Extract target for this specific track
        if len(target_data.shape) > 1:
            target_for_track = target_data[:, track_idx]
        else:
            target_for_track = target_data[track_idx]
        
        # Calculate target score (sum or mean of target values)
        if hasattr(target_for_track, '__len__') and len(target_for_track) > 1:
            target_score = np.sum(target_for_track)
            target_mean = np.mean(target_for_track)
        else:
            target_score = float(target_for_track)
            target_mean = float(target_for_track)
        
        ref_pred = pred_arrays[ref_idx] if pred_arrays else None
        
        # Calculate reference scores
        if hasattr(ref_pred, '__len__') and len(ref_pred) > 1:
            ref_score_sum = np.sum(ref_pred)
            ref_score_mean = np.mean(ref_pred)
        else:
            ref_score_sum = float(ref_pred)
            ref_score_mean = float(ref_pred)
        
        # First, add the reference row
        ref_result = {
            'gene_name': gene_name,
            'track_name': track_name,
            'sequence_name': ref_seq_name,
            'sequence_type': 'reference',
            'target_score_sum': target_score,
            'target_score_mean': target_mean,
            'predicted_score_sum': ref_score_sum,
            'predicted_score_mean': ref_score_mean,
            'diff_from_ref_sum': 0.0,
            'diff_from_ref_mean': 0.0,
        #     'percent_change_sum': 0.0,
        #     'percent_change_mean': 0.0,
        #     'correlation_with_target': np.corrcoef(ref_pred.flatten() if hasattr(ref_pred, 'flatten') else [ref_pred], 
        #                                          target_for_track.flatten() if hasattr(target_for_track, 'flatten') else [target_for_track])[0,1] if len(ref_pred) > 1 and len(target_for_track) > 1 else 0.0
        }
        results.append(ref_result)
        
        # Then, add all mutation rows
        for seq_idx, (pred_arr, seq_name) in enumerate(zip(pred_arrays, seq_names)):
            if seq_idx == ref_idx:  # Skip reference, already added
                continue
                
            # Calculate mutation prediction scores
            if hasattr(pred_arr, '__len__') and len(pred_arr) > 1:
                pred_score = np.sum(pred_arr)
                pred_mean = np.mean(pred_arr)
            else:
                pred_score = float(pred_arr)
                pred_mean = float(pred_arr)
            
            # Calculate differences from reference
            diff_from_ref_score = pred_score - ref_score_sum
            diff_from_ref_mean = pred_mean - ref_score_mean
            
            # Calculate percent change
            # percent_change_score = (diff_from_ref_score / ref_score_sum * 100) if ref_score_sum != 0 else 0.0
            # percent_change_mean = (diff_from_ref_mean / ref_score_mean * 100) if ref_score_mean != 0 else 0.0
            
            # Store mutation result
            mut_result = {
                'gene_name': gene_name,
                'track_name': track_name,
                'sequence_name': seq_name,
                'sequence_type': 'mutation',
                'target_score_sum': target_score,
                'target_score_mean': target_mean,
                'predicted_score_sum': pred_score,
                'predicted_score_mean': pred_mean,
                'diff_from_ref_sum': diff_from_ref_score,
                'diff_from_ref_mean': diff_from_ref_mean,
            #     'percent_change_sum': percent_change_score,
            #     'percent_change_mean': percent_change_mean,
            #     'correlation_with_target': np.corrcoef(pred_arr.flatten() if hasattr(pred_arr, 'flatten') else [pred_arr], 
            #                                          target_for_track.flatten() if hasattr(target_for_track, 'flatten') else [target_for_track])[0,1] if len(pred_arr) > 1 and len(target_for_track) > 1 else 0.0
            }
            results.append(mut_result)
    
    return results

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

    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, help='name of the model to use')
    parser.add_argument('--num-tracks', type=int, default=6, help='number of output tracks')
    parser.add_argument('--pt-name', type=str, help='name of the data pt file')
    parser.add_argument('--save-table-folder-name', type=str, help='path to save the output table')
    args = parser.parse_args()


    results_folder = '/home/chent9/projects/enformer-pytorch/results'
    pt_path = '/home/chent9/projects/enformer-pytorch/data'
    pt_name = args.pt_name
    num_tracks = args.num_tracks
    model_name = args.model_name


    pretrained = 'EleutherAI/enformer-official-rough'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    data_pt = torch.load(os.path.join(pt_path, pt_name), weights_only=False, map_location='cpu')
    enformer = from_pretrained(pretrained)
    model = HeadAdapterWrapper(enformer=enformer, num_tracks=num_tracks, post_transformer_embed=False).to(device)
    model_path = os.path.join(results_folder, model_name, 'model', 'best.pt')
    ckpt = torch.load(model_path, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()

    # Create output directory for tables
    table_folder_name = args.save_table_folder_name
    table_output_dir = os.path.join(results_folder, model_name, table_folder_name)
    os.makedirs(table_output_dir, exist_ok=True)

    # Store all results for final comprehensive table
    all_results = []

    with torch.no_grad():
        for batch_idx in range(100):  #len(data_pt)):
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

            # Move to CPU for calculation
            preds_cpu = preds.detach().cpu().numpy()
            target_cpu = target.detach().cpu().numpy()

            # Determine number of tracks and get track names
            num_tracks = preds_cpu.shape[2] if len(preds_cpu.shape) > 2 else preds_cpu.shape[1]
            track_names = TRACKS_48[:num_tracks] if num_tracks >= 48 else TRACKS_6[:num_tracks] if num_tracks == 6 else [f'track_{k}' for k in range(num_tracks)]

            # Organize data for calculation: each track gets predictions from all sequences
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

            # Target data for calculation
            if len(target_cpu.shape) > 1:
                target_for_calculation = target_cpu  # [num_bins, num_tracks]
            else:
                target_for_calculation = target_cpu  # [num_tracks]

            # Extract gene name from sequence names
            try:
                gene_name = seq_names[0].split('|')[1] if '|' in seq_names[0] else f"gene_{batch_idx}"
            except:
                gene_name = f"gene_{batch_idx}"

            # Calculate Enformer scores for this batch (organized format)
            batch_results = calculate_enformer_scores_organized(tracks_dict, seq_names, target_for_calculation, gene_name)
            all_results.extend(batch_results)
            
            # Save individual gene table (organized format)
            gene_df = pd.DataFrame(batch_results)
            gene_table_path = os.path.join(table_output_dir, f"gene_{gene_name}_scores.xlsx")
            gene_df.to_excel(gene_table_path, index=False, engine='openpyxl')
            print(f"Saved gene table: {gene_table_path}")

    # Save comprehensive table with all results (organized format)
    comprehensive_df = pd.DataFrame(all_results)
    comprehensive_table_path = os.path.join(table_output_dir, "all_genes_scores.xlsx")
    comprehensive_df.to_excel(comprehensive_table_path, index=False, engine='openpyxl')
    print(f"Saved comprehensive table: {comprehensive_table_path}")

    # Create analysis summaries
    if not comprehensive_df.empty:
        # Separate reference and mutation data
        references_df = comprehensive_df[comprehensive_df['sequence_type'] == 'reference'].copy()
        mutations_df = comprehensive_df[comprehensive_df['sequence_type'] == 'mutation'].copy()
        
        # Summary statistics for mutations
        if not mutations_df.empty:
            mutation_summary = mutations_df.groupby(['gene_name', 'track_name']).agg({
                'sequence_name': 'count',  # Number of mutations per gene-track
                'diff_from_ref_sum': ['mean', 'std', 'min', 'max'],
                'diff_from_ref_mean': ['mean', 'std', 'min', 'max']
                # 'percent_change_sum': ['mean', 'std', 'min', 'max'],
                # 'percent_change_mean': ['mean', 'std', 'min', 'max']
            }).round(4)
            
            mutation_summary.columns = ['_'.join(col).strip() for col in mutation_summary.columns]
            mutation_summary = mutation_summary.reset_index()
            
            summary_table_path = os.path.join(table_output_dir, "mutation_summary_stats.xlsx")
            mutation_summary.to_excel(summary_table_path, index=False, engine='openpyxl')
            print(f"Saved mutation summary: {summary_table_path}")

    # Save a comprehensive Excel file with multiple sheets
    comprehensive_excel_path = os.path.join(table_output_dir, "enformer_analysis_organized.xlsx")
    with pd.ExcelWriter(comprehensive_excel_path, engine='openpyxl') as writer:
        # All data sheet (organized: reference first, then mutations)
        comprehensive_df.to_excel(writer, sheet_name='All_Data_Organized', index=False)
        
        # Reference sequences only
        if not references_df.empty:
            references_df.to_excel(writer, sheet_name='References_Only', index=False)
        
        # Mutations only
        if not mutations_df.empty:
            mutations_df.to_excel(writer, sheet_name='Mutations_Only', index=False)
            
            # Top effects (mutations with largest absolute changes)
            top_effects = mutations_df.nlargest(100, 'diff_from_ref_sum', keep='all')[
                ['gene_name', 'track_name', 'sequence_name', 'predicted_score_sum', 
                 'diff_from_ref_sum', 'percent_change_sum']
            ].copy()
            top_effects.to_excel(writer, sheet_name='Top_100_Effects', index=False)
            
            # Summary statistics
            if 'mutation_summary' in locals():
                mutation_summary.to_excel(writer, sheet_name='Summary_Statistics', index=False)
    
    print(f"Saved comprehensive Excel file: {comprehensive_excel_path}")

    print(f"\nProcessed {len(data_pt)} genes with {len(all_results)} total sequence-track combinations")
    print(f"Tables saved to: {table_output_dir}")

if __name__ == '__main__':
    main()