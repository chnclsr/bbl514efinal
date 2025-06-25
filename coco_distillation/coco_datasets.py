import os
from PIL import Image
from torch.utils.data import Dataset
import json

class CocoCaptionsDataset(Dataset):
    def __init__(self, annotation_file, image_dir, tokenizer, transform=None, max_len=64):
        with open(annotation_file, 'r') as f:
            self.annotations_data = json.load(f)

        self.image_dir = image_dir
        self.tokenizer = tokenizer
        self.transform = transform
        self.max_len = max_len

        self.img_id_to_captions = {}
        for ann in self.annotations_data['annotations']:
            img_id = ann['image_id']
            caption = ann['caption']
            if img_id not in self.img_id_to_captions:
                self.img_id_to_captions[img_id] = []
            self.img_id_to_captions[img_id].append(caption)

        self.img_data = {ann['id']: ann for ann in self.annotations_data['images']}

        self.annotations = self.annotations_data['annotations']

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        ann = self.annotations[idx]
        image_id = ann['image_id']
        caption = ann['caption']

        img_info = self.img_data[image_id]
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        encoding = self.tokenizer.encode_plus(
            caption,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids = encoding['input_ids'].flatten()
        attention_mask = encoding['attention_mask'].flatten()

        return image, input_ids, attention_mask, caption, image_id


class CocoTestDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(image_dir) if os.path.isfile(os.path.join(image_dir, f))]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, self.image_files[idx]