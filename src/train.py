import pandas as pd
from preprocess import load_and_preprocess_data, set_seeds
from models import build_model, VOCAB_SIZE
from utiles import Metrics
import os

# --- Configuration ---
# All experiments will run for a fixed, small number of epochs (e.g., 3) 
# to save time, as per your constraint.
N_EPOCHS = 3 
BATCH_SIZE = 32

# Define the set of experiments (14 total)
EXPERIMENTS = [
    # 1. Sequence Length Variation (Fixed: LSTM, ReLU, Adam, No Clip)
    {'arch': 'LSTM', 'act': 'relu', 'opt': 'Adam', 'seq_len': 25, 'clip': None},
    {'arch': 'LSTM', 'act': 'relu', 'opt': 'Adam', 'seq_len': 50, 'clip': None},
    {'arch': 'LSTM', 'act': 'relu', 'opt': 'Adam', 'seq_len': 100, 'clip': None},
    
    # 2. Architecture Variation (Fixed: Seq 50, ReLU, Adam, No Clip)
    {'arch': 'RNN', 'act': 'relu', 'opt': 'Adam', 'seq_len': 50, 'clip': None},
    {'arch': 'Bidirectional LSTM', 'act': 'relu', 'opt': 'Adam', 'seq_len': 50, 'clip': None}, # LSTM already in Seq Var
    
    # 3. Optimizer Variation (Fixed: Seq 50, LSTM, ReLU, No Clip)
    {'arch': 'LSTM', 'act': 'relu', 'opt': 'SGD', 'seq_len': 50, 'clip': None},
    {'arch': 'LSTM', 'act': 'relu', 'opt': 'RMSProp', 'seq_len': 50, 'clip': None}, # Adam already in Seq Var
    
    # 4. Activation Variation (Fixed: Seq 50, LSTM, Adam, No Clip)
    {'arch': 'LSTM', 'act': 'sigmoid', 'opt': 'Adam', 'seq_len': 50, 'clip': None},
    {'arch': 'LSTM', 'act': 'tanh', 'opt': 'Adam', 'seq_len': 50, 'clip': None}, # ReLU already in Seq Var
    
    # 5. Stability/Clipping Variation (Fixed: Seq 50, RNN, Tanh, SGD) - RNN/Tanh/SGD are prone to instability
    {'arch': 'RNN', 'act': 'tanh', 'opt': 'SGD', 'seq_len': 50, 'clip': None}, # No clipping (Already in Optimizer Var, but used as baseline here)
    {'arch': 'RNN', 'act': 'tanh', 'opt': 'SGD', 'seq_len': 50, 'clip': 1.0}, # Clipping by Value=1.0
    
    # Add a few more unique ones to reach the 15-20 range if possible and time permits:
    {'arch': 'Bidirectional LSTM', 'act': 'tanh', 'opt': 'RMSProp', 'seq_len': 100, 'clip': None},
    {'arch': 'LSTM', 'act': 'relu', 'opt': 'Adam', 'seq_len': 50, 'clip': 1.0}, # Best arch + clip
    {'arch': 'RNN', 'act': 'sigmoid', 'opt': 'Adam', 'seq_len': 25, 'clip': None},
]

def run_experiments(experiments):
    """
    Runs all experiments and collects metrics.
    """
    all_results = []
    
    # Create results directory
    os.makedirs('results', exist_ok=True)
    
    for i, params in enumerate(experiments):
        print(f"\n--- Running Experiment {i+1}/{len(experiments)} ---")
        
        # A. Data Preprocessing
        # Load and pad data for the current sequence length
        X_train, y_train, X_test, y_test = load_and_preprocess_data(
            max_len=params['seq_len'], 
            vocab_size=VOCAB_SIZE
        )

        # B. Model Building
        model = build_model(
            architecture=params['arch'],
            activation=params['act'],
            optimizer=params['opt'],
            seq_len=params['seq_len'],
            clip_value=params['clip']
        )
        
        # C. Define Custom Metrics Callback
        metrics_callback = Metrics(validation_data=(X_test, y_test))

        # D. Training
        print(f"Training Model: {params}")
        
        history = model.fit(
            X_train, y_train,
            epochs=N_EPOCHS,
            batch_size=BATCH_SIZE,
            validation_data=(X_test, y_test),
            callbacks=[metrics_callback],
            verbose=2 # Set to 2 to only show one line per epoch
        )
        
        # E. Record Results
        # Store all epoch results for later analysis of 'Loss vs Epochs' plot
        for epoch_data in metrics_callback.epoch_metrics:
            result = {
                'Model': params['arch'],
                'Activation': params['act'],
                'Optimizer': params['opt'],
                'Seq_Length': params['seq_len'],
                'Grad_Clipping': 'Yes' if params['clip'] else 'No',
                'Epoch': epoch_data['epoch'],
                'Accuracy': epoch_data['accuracy'],
                'F1': epoch_data['f1_macro'],
                'Epoch_Time_s': epoch_data['epoch_time_s'],
                'Loss': epoch_data['val_loss']
            }
            all_results.append(result)

    # Save all results to CSV
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('results/metrics.csv', index=False)
    print("\nAll experiments complete. Results saved to results/metrics.csv")

if __name__ == '__main__':
    set_seeds(42) # Ensure full script reproducibility
    # run_experiments(EXPERIMENTS)
    print("--- Resuming Experiments from #5 (Index 4) ---")
    run_experiments(EXPERIMENTS[4:])