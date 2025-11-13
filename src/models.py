from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, LSTM, Bidirectional, Embedding, Dense, Dropout
from tensorflow.keras.optimizers import Adam, SGD, RMSprop

VOCAB_SIZE = 10000
EMBEDDING_DIM = 100
HIDDEN_SIZE = 64
DROPOUT_RATE = 0.4 # Fixed between 0.3-0.5
NUM_RNN_LAYERS = 2 # 2 Hidden layers

def build_model(architecture='LSTM', activation='relu', optimizer='Adam', seq_len=50, clip_value=None):
    """
    Builds and compiles the specified RNN model.
    clip_value: None for no clipping, or a float for clipvalue.
    """
    
    # 1. Select the base RNN layer
    # src/models.py (Corrected Logic)

# 1. Select the base RNN layer
    if architecture == 'RNN':
        RNN_Layer = SimpleRNN
    elif architecture == 'LSTM':
        RNN_Layer = LSTM
    elif architecture == 'Bidirectional LSTM':
        # A Bi-LSTM uses an LSTM cell as its inner layer
        RNN_Layer = LSTM  # <--- ADD THIS LINE
    else:
        raise ValueError("Invalid architecture")
    
    # 2. Select the optimizer with optional gradient clipping (Clip by Value used for simplicity)
    if clip_value is not None:
        clip_args = {'clipvalue': clip_value}
    else:
        clip_args = {}
        
    if optimizer == 'Adam':
        opt = Adam(**clip_args)
    elif optimizer == 'SGD':
        opt = SGD(learning_rate=0.01, **clip_args) # Using a default LR
    elif optimizer == 'RMSProp':
        opt = RMSprop(**clip_args)
    else:
        raise ValueError("Invalid optimizer")

    # 3. Build the Sequential Model
    model = Sequential()
    
    # Embedding Layer (Input Layer)
    model.add(Embedding(
        input_dim=VOCAB_SIZE, 
        output_dim=EMBEDDING_DIM, 
        input_length=seq_len
    ))
    
    # 2 Hidden Layers (RNN/LSTM/Bi-LSTM)
    for i in range(NUM_RNN_LAYERS):
        is_last_layer = (i == NUM_RNN_LAYERS - 1)
        
        # RNN/LSTM/GRU layers should return sequences for intermediate layers
        # but only return the final output for the last layer before Dense
        return_seq = not is_last_layer 
        
        rnn_layer = RNN_Layer(
            units=HIDDEN_SIZE, 
            activation=activation, 
            return_sequences=return_seq
        )
        
        if architecture == 'Bidirectional LSTM':
             # Apply Bidirectional wrapper
             model.add(Bidirectional(rnn_layer))
        else:
             model.add(rnn_layer)
        
        # Dropout
        if not is_last_layer:
            model.add(Dropout(DROPOUT_RATE))

    # Output Layer (Fully Connected)
    model.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    return model