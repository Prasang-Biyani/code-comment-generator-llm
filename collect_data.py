from datasets import load_dataset
import pandas as pd
import os

# Load the Python subset of CodeSearchNet (training split)
dataset = load_dataset("code_search_net", "python", split="train")

# Extract relevant fields: code and comments
data = {
    "code": dataset["func_code_string"],           # The code snippet
    "comment": dataset["func_documentation_string"] # The associated comment/docstring
}

# Create a pandas DataFrame
df = pd.DataFrame(data)

# Ensure the 'data' directory exists (as per your project structure)
os.makedirs("data", exist_ok=True)

# Save to CSV
csv_path = "data/codesearchnet_python_train.csv"
df.to_csv(csv_path, index=False)

print(f"Dataset saved to {csv_path}")
print(f"Number of code-comment pairs: {len(df)}")

