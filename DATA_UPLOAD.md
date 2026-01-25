# Data Upload Requirements for TextbookReasoning

## Overview

This environment requires the TextbookReasoning dataset to be uploaded to OpenReward cloud storage. The dataset contains 651,840 academic reasoning questions across 7 STEM subjects.

## Directory Structure

Upload the following data structure to your OpenReward namespace at https://openreward.ai:

```
/orwd_data/textbookreasoning/
└── textbookreasoning_train.parquet
```

## Files Required

### textbookreasoning_train.parquet
- **Description**: Complete TextbookReasoning training dataset
- **Size**: ~532 MB
- **Records**: 651,840 examples
- **Subjects**: medicine, math, biology, cs, physics, chemistry, economics
- **Schema**:
  - `question` (string): The question/problem statement
  - `answer` (string): Detailed answer explanation
  - `subject` (string): Subject category
  - `reference_answer` (string): Concise reference answer for grading

## How to Download Data

### Step 1: Install Required Libraries

```bash
pip install datasets pandas pyarrow
```

### Step 2: Download Dataset

Create a download script `download_data.py`:

```python
import os
import pandas as pd
from datasets import load_dataset

# Load train split only (651,840 examples)
dataset = load_dataset("MegaScience/TextbookReasoning", split="train")

# Convert to pandas DataFrame
df = pd.DataFrame(dataset)

# Save as parquet
df.to_parquet("textbookreasoning_train.parquet", index=False)

print(f"Downloaded {len(df)} examples")
print(f"Subjects: {df['subject'].unique()}")
print(f"File size: {os.path.getsize('textbookreasoning_train.parquet') / 1024 / 1024:.2f} MB")
```

Run the script:

```bash
python download_data.py
```

This will create `textbookreasoning_train.parquet` in your current directory.

### Step 3: Upload to OpenReward

1. Go to https://openreward.ai and navigate to your namespace settings
2. Create a directory: `/orwd_data/textbookreasoning/`
3. Upload `textbookreasoning_train.parquet` to this directory
4. Verify the file path matches exactly: `/orwd_data/textbookreasoning/textbookreasoning_train.parquet`

## Dataset Schema Details

### Required Columns

| Column | Type | Description | Notes |
|--------|------|-------------|-------|
| `question` | string | The question/problem text | May contain LaTeX notation |
| `answer` | string | Detailed answer with explanation | Used as fallback if reference_answer is empty |
| `subject` | string | Subject category | One of 7 categories |
| `reference_answer` | string | Concise reference answer | Primary grading reference; may be empty |

### Subject Distribution

The dataset covers 7 subjects:
- Medicine
- Math
- Biology
- Computer Science (cs)
- Physics
- Chemistry
- Economics

## Data Size Summary

| Component | Size | Count |
|-----------|------|-------|
| Total parquet file | ~532 MB | 651,840 examples |
| Train split | 100% | 651,840 examples |

## Important Notes

1. **Large Dataset**: The 532 MB file should NOT be included in the Docker image. It must be mounted at `/orwd_data/`.

2. **Reference Answer Handling**: Some examples may have empty `reference_answer` fields. The environment falls back to the `answer` field for grading in these cases.

3. **LaTeX Notation**: Questions and answers may contain LaTeX math notation (e.g., `\boxed{}`). The LLM grader is designed to handle this.

4. **Mount Path**: The environment expects data at `/orwd_data/textbookreasoning/`. Do not modify this path.

5. **Parquet Format**: Keep the file in parquet format for efficient loading (pandas + pyarrow).

## Verification

After uploading, verify the data is accessible:

1. Deploy the environment to OpenReward
2. Check that `list_splits()` returns `["train"]`
3. Verify task count: `list_tasks("train")` should return 651,840 tasks
4. Test a few examples to ensure questions and grading work correctly

## Troubleshooting

### Issue: FileNotFoundError when running environment

**Solution**: Verify:
1. Data is uploaded to `/orwd_data/textbookreasoning/`
2. File is named exactly: `textbookreasoning_train.parquet`
3. Path is case-sensitive

### Issue: KeyError for columns

**Solution**: Ensure the parquet file has all required columns:
- `question`
- `answer`
- `subject`
- `reference_answer`

### Issue: Out of memory when loading

**Solution**: The environment loads the full dataset at startup. Ensure your deployment has sufficient RAM (recommend at least 2GB).

### Issue: Download fails or times out

**Solution**:
- Check internet connection
- HuggingFace may be rate-limiting; wait and retry
- Use the `datasets` library cache to resume partial downloads

## License and Usage

Please refer to the original dataset's license terms on HuggingFace before using this data:
https://huggingface.co/datasets/MegaScience/TextbookReasoning

The dataset is licensed under CC-BY-NC-SA-4.0 (non-commercial use).

## Support

For issues with:
- **Dataset access**: Contact the dataset authors on HuggingFace
- **OpenReward uploads**: See https://docs.openreward.org/
- **Environment setup**: Open an issue on the GitHub repository
