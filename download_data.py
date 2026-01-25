"""
Download TextbookReasoning dataset from HuggingFace.

This script downloads the train split of the TextbookReasoning dataset
and saves it as a parquet file for use in the environment.
"""

import os
import pandas as pd
from datasets import load_dataset


def download_textbook_reasoning():
    """Download and convert TextbookReasoning dataset to parquet."""

    print("Loading TextbookReasoning dataset from HuggingFace...")
    print("This may take a few minutes (dataset is ~532 MB)...")

    # Load train split only
    dataset = load_dataset("MegaScience/TextbookReasoning", split="train")

    print(f"Loaded {len(dataset)} examples")

    # Convert to pandas DataFrame
    df = pd.DataFrame(dataset)

    print(f"\nDataset info:")
    print(f"- Records: {len(df)}")
    print(f"- Columns: {list(df.columns)}")
    print(f"- Subjects: {df['subject'].unique().tolist()}")

    # Create output directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Save as parquet
    output_path = "data/textbookreasoning_train.parquet"
    df.to_parquet(output_path, index=False)

    file_size_mb = os.path.getsize(output_path) / 1024 / 1024
    print(f"\nSaved to: {output_path}")
    print(f"File size: {file_size_mb:.2f} MB")

    # Show sample
    print(f"\nSample question:")
    print(f"Subject: {df.iloc[0]['subject']}")
    print(f"Question: {df.iloc[0]['question'][:200]}...")
    print(f"Reference answer: {df.iloc[0].get('reference_answer', 'N/A')[:100]}...")


if __name__ == "__main__":
    download_textbook_reasoning()
