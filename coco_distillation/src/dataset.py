import numpy as np
from PIL import Image

class Dataset:
    def __init__(self, df, tfms, tokenizer):
        self.df = df
        self.tfms = tfms
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        sample = self.df.iloc[idx,:]
        image = sample['image']
        caption = sample['caption']
        image = Image.open(image).convert('RGB')
        image = np.array(image)
        augs = self.tfms(image=image)
        image = augs['image']
        caption = f"{caption}<|endoftext|>"
        input_ids = self.tokenizer(
            caption,
            truncation=True)['input_ids']
        labels = input_ids.copy()
        labels[:-1] = input_ids[1:]
        return image, input_ids, labels