import torch
import torch.nn as nn
import pandas as pd
import numpy as np

# Modify these variables according to your setup
MODEL_PATH = 'path/to/your_model.pt'  # Replace with the path to your .pt file
DATA_PATH = 'path/to/test_data.csv'   # Replace with your test data's path

# Load your saved model
def load_model(model_path):
    model = torch.load(model_path, map_location=torch.device('cpu'))  # Load to CPU by default
    model.eval()  # Set the model to evaluation mode
    return model

# Preprocess your data (adapt this as needed)
def preprocess_data(data):
    # Assuming input is in a CSV format; modify if it's different
    data = pd.read_csv(data)
    # Example: Convert DataFrame to tensor
    input_tensor = torch.tensor(data.values, dtype=torch.float32)
    return input_tensor

# Test the model with the provided data
def test_model(model, input_tensor):
    with torch.no_grad():  # Disable gradient calculation
        output = model(input_tensor)
    print("Model Output:", output)

def main():
    # Load model and data
    model = load_model(MODEL_PATH)
    input_tensor = preprocess_data(DATA_PATH)
    
    # Ensure the input tensor has the correct shape for the model
    if len(input_tensor.shape) == 1:
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension if necessary

    # Run the model
    test_model(model, input_tensor)

if __name__ == '__main__':
    main()