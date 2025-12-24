import torch
from torch.utils.data import Dataset
import os
import csv

class TwitterDataSet(Dataset):
    def __init__(self, root_folder, file_name, classes, tokenizer, header=True, min_length=40, max_length=128, verbose=False):
        super(TwitterDataSet, self).__init__()

        self.root_folder = root_folder
        self.file_name = file_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.classes = classes
        self.class_converter = {c: i for i, c in enumerate(self.classes)}
        self.min_length = min_length
        self.header = header
        self.verbose = verbose

        self.texts = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        if self.verbose:
            print(f"Loading data from {self.file_name}...")
        data_file = os.path.join(self.root_folder, self.file_name)

        with open(data_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            if self.header:
                if self.verbose:
                    print("Skipping header row")
                next(reader)

            if self.verbose:
                print("Processing rows")
            for row in reader:
                if len(row) < 4:
                    continue
                text, label = row[3], row[2]
                if len(text.split()) < self.min_length:
                    continue
                if label not in self.class_converter:
                    continue
                self.texts.append(text)

                self.labels.append(self.class_converter[label])
        if self.verbose:
            print(f"Loaded {len(self.texts)} samples.")

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }
    
if __name__ == "__main__":
    from transformers import BertTokenizer

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = TwitterDataSet(
        root_folder='dataset/twitter',
        file_name='twitter_sentiment_analysis.csv',
        classes=['Negative', 'Neutral', 'Positive', "Irrelevant"],
        tokenizer=tokenizer,
        min_length=40,
        max_length=200
    )

    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample text: {sample['text']}")
    print(f"Reconstructed Text: {tokenizer.decode(sample['input_ids'], skip_special_tokens=True)}")
    print(f"Input IDs: {sample['input_ids']}")
    print(f"Attention Mask: {sample['attention_mask']}")
    print(f"Label: {sample['label']}")

    for sample in dataset:
        assert sample['input_ids'].shape[0] == 200
        assert sample['attention_mask'].shape[0] == 200
        assert sample['label'].item() in range(len(dataset.classes))
        assert 40 <= len(sample['text'].split()) <= 200

    import pandas as pd
    df = pd.DataFrame(dataset)