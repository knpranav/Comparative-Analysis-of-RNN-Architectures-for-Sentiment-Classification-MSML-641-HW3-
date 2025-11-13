import pandas as pd
import matplotlib.pyplot as plt
import os

def generate_report_artifacts():
    """Loads metrics and generates the required plots."""
    
    os.makedirs('results/plots', exist_ok=True)
    
    try:
        df = pd.read_csv('results/metrics.csv')
    except FileNotFoundError:
        print("Error: metrics.csv not found. Run train.py first.")
        return

    # --- 1. Summary Table for the Report (Picking Final Epoch Metrics) ---
    # Find the last epoch for each unique configuration
    final_epoch_df = df.sort_values(by='Epoch', ascending=False).drop_duplicates(
        subset=['Model', 'Activation', 'Optimizer', 'Seq_Length', 'Grad_Clipping'], keep='first'
    )
    
    # Save the final summary table (formatted)
    summary_cols = ['Model', 'Activation', 'Optimizer', 'Seq_Length', 'Grad_Clipping', 
                    'Accuracy', 'F1', 'Epoch_Time_s']
    summary_df = final_epoch_df[summary_cols].round(4)
    summary_df.rename(columns={'Epoch_Time_s': 'Epoch Time (s)', 
                               'Seq_Length': 'Seq Length'}, inplace=True)
    summary_df.to_csv('results/summary_metrics.csv', index=False)
    
    # --- 2. Plot: Accuracy/F1 vs. Sequence Length ---
    # Filter the experiments used for the Seq_Length comparison
    seq_df = df[(df['Model'] == 'LSTM') & (df['Activation'] == 'relu') & 
                (df['Optimizer'] == 'Adam') & (df['Grad_Clipping'] == 'No')]
    
    # Group by sequence length for final epoch results
    seq_plot_data = seq_df.sort_values(by='Epoch', ascending=False).drop_duplicates(
        subset=['Seq_Length'], keep='first'
    ).sort_values('Seq_Length')

    plt.figure(figsize=(10, 6))
    plt.plot(seq_plot_data['Seq_Length'], seq_plot_data['Accuracy'], 
             marker='o', label='Accuracy')
    plt.plot(seq_plot_data['Seq_Length'], seq_plot_data['F1'], 
             marker='o', label='F1-score (Macro)')
    plt.xlabel('Sequence Length')
    plt.ylabel('Performance')
    plt.title('Accuracy/F1 vs. Sequence Length (Fixed LSTM, ReLU, Adam, No Clip)')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/acc_f1_vs_seqlen.png')
    plt.close()
    print("Generated plots/acc_f1_vs_seqlen.png")


    # --- 3. Plot: Training Loss vs. Epochs (for best and worst models) ---
    
    # Find the best and worst final-epoch F1 score
    best_config = final_epoch_df.loc[final_epoch_df['F1'].idxmax()]
    worst_config = final_epoch_df.loc[final_epoch_df['F1'].idxmin()]

    # Helper function to filter history for a specific config
    def get_history(config, df):
        return df[(df['Model'] == config['Model']) & 
                  (df['Activation'] == config['Activation']) & 
                  (df['Optimizer'] == config['Optimizer']) & 
                  (df['Seq_Length'] == config['Seq_Length']) & 
                  (df['Grad_Clipping'] == config['Grad_Clipping'])]
                  
    best_history = get_history(best_config, df)
    worst_history = get_history(worst_config, df)

    plt.figure(figsize=(10, 6))
    plt.plot(best_history['Epoch'], best_history['Loss'], marker='o', 
             label=f'Best Model (F1: {best_config["F1"]:.4f})')
    plt.plot(worst_history['Epoch'], worst_history['Loss'], marker='x', 
             label=f'Worst Model (F1: {worst_config["F1"]:.4f})')
    
    plt.xlabel('Epoch')
    plt.ylabel('Validation Loss')
    plt.title('Validation Loss vs. Epochs for Best and Worst Configurations')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/plots/loss_vs_epochs.png')
    plt.close()
    print("Generated plots/loss_vs_epochs.png")

    print("\nReport artifacts generated. Review results/summary_metrics.csv and results/plots/.")

if __name__ == '__main__':
    generate_report_artifacts()