import sys
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from aop_dataloader import *
from seq_model_def import *
from graph_model_def import *
from aop_def import *
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import csv

# Define global constants
seq_length = 50
batch_size = 500

normal_aa = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']


def fasta2csv(fasta_file, csv_file):
    normal_aa = ['A', 'R', 'N', 'D', 'C', 'E', 'Q', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    normal_aa_set = set(normal_aa)  # Convert to set for faster lookup

    with open(fasta_file, 'r') as f, open(csv_file, 'w', newline='') as csvf:
        writer = csv.writer(csvf)
        writer.writerow(['name', 'SEQUENCE'])
        seq_id = ''
        seq = ''

        # Read the FASTA file
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Check if previous sequence is valid before writing
                if seq_id and len(seq) >= 2 and len(seq) <= 30:
                    # Only write if all amino acids are in the normal set
                    if all(aa in normal_aa_set for aa in seq):
                        writer.writerow([seq_id, seq])
                    else:
                        print(f"Skipped {seq_id}: contains non-standard amino acids")
                seq_id = line[1:]
                seq = ''
            else:
                seq += line.upper()  # Convert to uppercase and append
        if seq_id and len(seq) >= 2 and len(seq) <= 30:
            if all(aa in normal_aa_set for aa in seq):
                writer.writerow([seq_id, seq])
            else:
                print(f"Skipped {seq_id}: contains non-standard amino acids")

def csv2fasta_func(csv_path, fasta_path):
    # get .csv info
    seq_data = pd.read_csv(csv_path)
    # .csv to .fasta
    fast_file = open(fasta_path, "w")
    for i in range(len(seq_data.SEQUENCE)):
        fast_file.write(">" + str(seq_data.name[i]) + "\n")
        fast_file.write(seq_data.SEQUENCE[i] + "\n")
    fast_file.close()


def aop_predict(model_path, csv_path, out_csv_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = CombinedModel()
    # Load model checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    test_df = pd.read_csv(csv_path)
    if 'label' not in test_df.columns:
        test_df['label'] = [0] * len(test_df)
        test_df.to_csv(csv_path, index=False)

    test_df = pd.read_csv(csv_path)
    test_loader = get_data_loader(csv_path, batch_size=batch_size, seq_length=seq_length)

    test_prob_list = []
    test_pred_list = []
    test_target_list = []

    print("Extracting features...")
    with torch.no_grad():
        for batch in test_loader:
            sequences = batch['sequences'].to(device)
            x = batch['x'].to(device)
            edge_index = batch['edge_index'].to(device)
            edge_attr = batch['edge_attr'].to(device)
            batch_idx_tensor = batch['batch'].to(device)
            labels = batch['labels'].to(device)

            seq_features, pooled_seq, graph_features, fused_features, last_hidden, outputs = model(
                sequences, x, edge_index, edge_attr, batch_idx_tensor
            )
            probs = outputs.squeeze().cpu().numpy()
            preds = (probs > 0.5).astype(float)

            test_prob_list.extend(probs.tolist() if probs.ndim > 0 else [probs.item()])
            test_pred_list.extend(preds.tolist() if preds.ndim > 0 else [preds.item()])
            test_target_list.extend(labels.cpu().numpy().tolist())

    test_df['probs'] = test_prob_list
    test_df['preds'] = test_pred_list
    test_df.to_csv(out_csv_path, index=False)
    return test_df

if __name__ == "__main__":
    model_path = '/home/jianxiu/OneDrive/aop_pipeline/model/best_model_Oct13.pth'
    fasta_path = 'eggshell_out_Oct6.fasta'
    csv_path = 'eggshell_out_Oct6.csv'
    fasta2csv(fasta_path, csv_path)
    out_csv_path = 'eggshell_predict.csv'

    # csv_path = '/home/jianxiu/OneDrive/amp_rf/nov13/nov13_sa_ec.csv'
    # out_csv_path = '/home/jianxiu/OneDrive/amp_rf/nov13/nov13_sa_ec_aop.csv'

    aop_predict(model_path, csv_path, out_csv_path)