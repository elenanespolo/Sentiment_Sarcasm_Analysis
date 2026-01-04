import torch
from torch.utils.data import Dataset
import os
import csv

class BicodemixDataSet(Dataset):
    def __init__(self, root_folder, file_name, classes, tokenizer, header=True, min_length=10, max_length=200, **kwargs):
        super(BicodemixDataSet, self).__init__()

        self.root_folder = root_folder
        self.file_name = file_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.classes = classes
        self.sarc_class_converter = {'sarc': 1, 'notsarc': 0}
        self.sent_class_converter = {'positive': 1, 'negative': 0, 'neutral': 2}
        self.min_length = min_length
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

            for row in reader:
                if len(row) < 5:
                    continue
                text, sarcasm_label, sentiment_label = row

                if len(text.split()) < self.min_length:
                    continue

                #TODO: check if label is in classes of interest
                # if label not in self.class_converter.values():
                #     continue

                self.texts.append(text)
                self.labels.append((self.sarc_class_converter[sarcasm_label], self.sent_class_converter[sentiment_label]))

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
    
    def get_type(self):
        return {
            'variety': self.variety,
            'source': self.source,
            'task': self.task
        }

if __name__ == "__main__":
    from transformers import BertTokenizer

    max_length = 200
    min_length = 1
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    dataset = BicodemixDataSet(
        root_folder='dataset/bicodemix',
        file_name='train.csv',
        classes=['0', '1'],
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

    for sample in dataset:
        assert sample['input_ids'].shape[0] == max_length
        assert sample['attention_mask'].shape[0] == max_length
        assert sample['label'].item() in range(len(dataset.classes))
        # print(f"Text length (in words): {len(sample['text'].split())}")
        # print((sample["input_ids"] != 0).sum().item())
        assert min_length <= (sample["input_ids"] != 0).sum().item() <= max_length
    print("All samples passed the checks.")