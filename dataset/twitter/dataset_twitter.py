import torch
from torch.utils.data import Dataset
import os
import csv

class TwitterDataSet(Dataset):
    def __init__(self, root_folder, file_name, classes, tokenizer, header=True, min_length=40, max_length=200, **kwargs):
        super(TwitterDataSet, self).__init__()

        self.root_folder = root_folder
        self.file_name = file_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_length = min_length
        
        self.classes = classes
        self.class_converter_sentiment = {c: i for i, c in enumerate(self.classes['sentiment'])}
        self.class_weights_sentiment = {c: 0 for c in self.class_converter_sentiment.keys()}

        self.header = header

        self.texts = []
        self.labels = []
        self._load_data()

    def _load_data(self):
        data_file = os.path.join(self.root_folder, self.file_name)

        with open(data_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            if self.header:
                next(reader)

            # columns: cod,category,sentiment,text
            for row in reader:
                if len(row) < 4:
                    continue
                cod, category, label, text = row

                if len(text.split()) < self.min_length:
                    continue
                if label not in self.class_converter_sentiment:
                    continue
                
                self.texts.append(text)
                self.labels.append((self.class_converter_sentiment[label],))
                self.class_weights_sentiment[label] += 1

    def __len__(self):
        return len(self.texts)

    def get_label_count(self):
        return {
            'sentiment': self.class_weights_sentiment,
            'sarcasm': {}
        }
    
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
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

    max_length = 200
    min_length = 40
    CFG = {
        'root_folder': './dataset/besstie',
        'file_name': 'train_SS_with_nan.csv',
        'classes': {
            'sentiment': ['Negative', 'Positive'],
            'sarcasm': [],
        }
    }
    dataset = TwitterDataSet(
        **CFG,
        tokenizer=tokenizer,
        min_length=min_length,
        max_length=max_length
    )

    print(f"Dataset size: {len(dataset)}")
    sample = dataset[0]
    print(f"Sample text: {sample['text']}")
    print(f"Reconstructed Text: {tokenizer.decode(sample['input_ids'], skip_special_tokens=True)}")
    print(f"Input IDs: {sample['input_ids']}")
    print(f"Attention Mask: {sample['attention_mask']}")
    print(f"Label: {sample['label']}")
    print(f"Class weights sentiment: {dataset.class_weights_sentiment}")

    for sample in dataset:
        assert sample['input_ids'].shape[0] == max_length
        assert sample['attention_mask'].shape[0] == max_length
        assert sample['label'][0].item() in range(len(dataset.classes['sentiment']))
        assert len(sample['label']) == 1
        assert min_length <= (sample["input_ids"] != 0).sum().item() <= max_length
    print("All samples passed the checks.")