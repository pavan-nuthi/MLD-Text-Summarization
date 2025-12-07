# Text Summarization Project

This project compares **BERTSum** (Extractive) and **BART-base** (Abstractive) summarization models on the BBC News Summary Dataset.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### Training

Train BERTSum:
```bash
python3 main.py train --model_type bertsum --epochs 3 --batch_size 4
```

Train BART:
```bash
python3 main.py train --model_type bart --epochs 3 --batch_size 4
```

### Evaluation

Evaluate BERTSum:
```bash
python3 main.py evaluate --model_type bertsum --model_path ./saved_models/bertsum
```

Evaluate BART:
```bash
python3 main.py evaluate --model_type bart --model_path ./saved_models/bart
```

### Testing (Test Split Only)

To evaluate specifically on the test set:

```bash
python3 main.py evaluate --model_type bertsum --model_path ./saved_models/bertsum --split test
```

```bash
python3 main.py evaluate --model_type bart --model_path ./saved_models/bart --split test
```

### Troubleshooting: Memory Issues (MPS/Mac)

If you encounter `RuntimeError: MPS backend out of memory`, try reducing the batch size and increasing gradient accumulation steps:

```bash
python3 main.py train --model_type bart --epochs 3 --batch_size 1 --gradient_accumulation_steps 4
```

## Project Structure

- `models/`: Model definitions.
- `utils/`: Helper scripts for data loading and preprocessing.
- `train.py`: Training logic.
- `evaluation.py`: Evaluation logic.
