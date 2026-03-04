Link to saved best model: https://drive.google.com/drive/folders/1zmsIBOOMN_DMvc20EHCPCLv2IO3kBP0y?usp=sharing


## Model architecture

This project uses a Transformer text classifier based on **RoBERTa-base** (`AutoModelForSequenceClassification`) with **2 labels**:

- `0` = Non-PCL
- `1` = PCL

Training variants in the repo include class-weighted loss, data augmentation ( backtranslation), and R-Drop regularization.

## Run best configuration

Install dependencies (one of):

- `pip install -r requirements_enhanced.txt`

Then run:

- `python run_best_config.py`

This script launches `task3_roberta_enhanced.py` with the best hyperparameters found in the search.


