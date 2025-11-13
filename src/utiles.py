import time
import numpy as np
from tensorflow.keras.callbacks import Callback
from sklearn.metrics import f1_score, precision_score, recall_score

class Metrics(Callback):
    """
    A Keras callback to compute Macro F1-score at the end of each epoch 
    and record the epoch training time.
    """
    def __init__(self, validation_data):
        super().__init__()
        self.validation_data = validation_data
        self.epoch_metrics = []

    def on_train_begin(self, logs=None):
        self.val_data = self.validation_data[0]
        self.val_targ = self.validation_data[1]

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start_time = time.time()

    def on_epoch_end(self, epoch, logs=None):
        epoch_time = time.time() - self.epoch_start_time
        
        # Get predictions (round to 0 or 1 for classification metrics)
        val_predict_probs = self.model.predict(self.val_data, verbose=0)
        val_predict = (val_predict_probs > 0.5).astype(int)
        
        # Calculate F1-score (Macro)
        _val_f1 = f1_score(self.val_targ, val_predict, average='macro', zero_division=0)
        _val_accuracy = logs.get('val_accuracy') # Accuracy is computed by Keras internally
        
        logs['val_f1_macro'] = _val_f1
        logs['epoch_time_s'] = epoch_time
        
        self.epoch_metrics.append({
            'epoch': epoch + 1,
            'accuracy': _val_accuracy,
            'f1_macro': _val_f1,
            'epoch_time_s': epoch_time,
            'val_loss': logs.get('val_loss')
        })

        print(f" — val_f1_macro: {_val_f1:.4f} — epoch_time: {epoch_time:.2f}s")
        return

def get_optimizer_name(optimizer):
    """Helper to get a clean name for the optimizer object."""
    config = optimizer.get_config()
    name = config['name']
    
    # Check for gradient clipping
    clip_value = config.get('clipvalue')
    clip_norm = config.get('clipnorm')
    
    if clip_value is not None and clip_value > 0:
        return f"{name}_ClipV{clip_value}"
    elif clip_norm is not None and clip_norm > 0:
        return f"{name}_ClipN{clip_norm}"
    return name