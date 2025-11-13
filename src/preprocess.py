import numpy as np
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing.sequence import pad_sequences
import random, torch # For reproducibility setup

# --- Setup for Reproducibility ---
def set_seeds(seed=42):
    """Fixes random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

# --- Data Preparation ---
def load_and_preprocess_data(max_len=50, vocab_size=10000):
    """
    Loads IMDB dataset and performs required preprocessing:
    - 50/50 split (handled by load_data)
    - Keep top 'vocab_size' words
    - Pad/truncate to 'max_len'
    """
    set_seeds(42)
    
    # Keras IMDb dataset is already preprocessed (tokenized, integer-encoded, top-k filtering)
    # The default split is 25k train, 25k test
    (x_train, y_train), (x_test, y_test) = imdb.load_data(
        num_words=vocab_size,
        skip_top=0, # Do not skip top words
        oov_char=2, # Out-of-vocabulary token
        index_from=3 # Start index for actual words
    )
    
    # Pad/truncate sequences to a fixed length
    x_train_padded = pad_sequences(x_train, maxlen=max_len, padding='post', truncating='post')
    x_test_padded = pad_sequences(x_test, maxlen=max_len, padding='post', truncating='post')
    
    # Convert labels to float for binary cross-entropy compatibility
    y_train = np.asarray(y_train).astype('float32')
    y_test = np.asarray(y_test).astype('float32')

    print(f"Loaded and preprocessed data with max_len={max_len} and vocab_size={vocab_size}")
    return x_train_padded, y_train, x_test_padded, y_test

if __name__ == '__main__':
    # Example usage (for testing/reporting: Dataset Summary)
    X_train, y_train, X_test, y_test = load_and_preprocess_data(max_len=50)
    print(f"\nTraining set shape: {X_train.shape}")
    print(f"Vocabulary size: {imdb.load_data(num_words=10000)[0][0].max() + 1}")
    # To get statistics like average review length, you'd need to manually decode the reviews
    # or skip the pre-tokenized data, but the assignment allows Keras Tokenizer.