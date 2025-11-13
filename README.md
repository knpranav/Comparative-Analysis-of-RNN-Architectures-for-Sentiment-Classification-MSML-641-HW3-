Comparative Analysis of RNN Architectures for Sentiment Classification
======================================================================

Project Overview
----------------

This project implements and evaluates multiple Recurrent Neural Network (RNN) architectures (Simple RNN, LSTM, Bidirectional LSTM) for binary sentiment classification using the IMDb Movie Review Dataset. The primary goal is to systematically compare the effects of varying architectural components, optimizers, activation functions, and sequence lengths on model performance (Accuracy, F1-score) and efficiency (Epoch Time) under CPU-constrained hardware.

  

Setup Instructions
------------------

 1\. Python Environment

This project was developed using Python 3.9 (Anaconda distribution) within an isolated virtual environment (venvhw3).

To set up the environment:

1.  Navigate to the root directory of the project.
2.  Create and activate the virtual environment using your Python 3.9 executable (Adjust the path if necessary):

    C:\Users\pranav\anaconda3\python.exe -m venv venvhw3
     Activate on Windows:
    .\venvhw3\Scripts\activate
    
    2. Dependencies
    
    Install all required packages using the provided dependency list:
    
    pip install -r requirements.txt
    
    
    ▶️ How to Run the Project
    
    The project is executed in two main stages: Training and Evaluation.
    
    1. Training and Data Collection
    
    The train.py script executes all 14 experiments, training each configuration for 3 epochs and recording metrics.
    The necessary data (IMDb) is downloaded automatically by the Keras utility.
    
    Command:
    
    python src/train.py
    
    
    Expected Runtime:
    Given the CPU-only hardware (Intel Core i5, 16GB RAM), the total runtime for all 14 experiments is approximately 30–45 minutes.
    Individual epoch times vary significantly, from ≈ 23 seconds (Simple RNN, Seq 25) to ≈ 177 seconds (Bi-LSTM, Seq 100).
    
    2. Evaluation and Plot Generation
    
    The evaluate.py script processes the collected data (results/metrics.csv), generates the final summary table, and creates the required plots.
    
    Command:
    
    python src/evaluate.py
    
    📂 Deliverables and Expected Output
    
    All generated metrics and plots are stored within the results/ directory.
    
    File/Directory	Description
    data/	Empty directory reserved for potential external datasets (required submission component)
    
    src/	Contains all project logic
    
    results/metrics.csv	Raw Output File: Full performance history (Loss, F1, Accuracy, Time) for every epoch of all 14 experiments
    
    results/summarymetrics.csv	Final Table: Contains the single best (last epoch) metrics for each of the 14 unique configurations
    
    results/plots/accf1vsseqlen.png	Plot comparing Accuracy/F1 across the three Sequence Lengths
    
    results/plots/lossvsepochs.png	Plot comparing Validation Loss vs. Epochs for the best and worst performing models
    
    msml641hw3report.pdf	The final project report containing analysis, discussion, and conclusions
    

