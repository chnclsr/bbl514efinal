import os
import random
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, Dataset
from transformers import GPT2TokenizerFast
import time
import nltk
from coco_datasets import CocoCaptionsDataset
from multi_modal import create_model, create_image_processor
from trainers import Trainer
from distiller import DatasetDistiller
from constants import *


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class CollateFn:
    def __init__(self, image_processor, tokenizer):
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def __call__(self, batch):
        images_pil, _, _, captions, img_ids = zip(*batch)
        pixel_values = self.image_processor(images=list(images_pil), return_tensors="pt").pixel_values
        labels = self.tokenizer(
            list(captions), return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_TEXT_LEN
        ).input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100
        return pixel_values, labels, list(captions), list(img_ids)


class SyntheticLabelledDataset(Dataset):
    def __init__(self, syn_image_dataset, real_label_dataset):
        self.syn_images = syn_image_dataset
        if len(syn_image_dataset) > len(real_label_dataset):
            raise ValueError("Not enough real labels for the number of synthetic images.")
        self.real_labels = Subset(real_label_dataset, list(range(len(syn_image_dataset))))

    def __len__(self):
        return len(self.syn_images)

    def __getitem__(self, idx):
        syn_pixel_values = self.syn_images[idx][0]
        _, _, _, real_caption, real_id = self.real_labels[idx]
        return syn_pixel_values, None, real_caption, real_id


class StudentCollateFn:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        pixel_values, _, captions, img_ids = zip(*batch)
        pixel_values_tensor = torch.stack(pixel_values, 0)

        labels = self.tokenizer(
            list(captions), return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_TEXT_LEN
        ).input_ids
        labels[labels == self.tokenizer.pad_token_id] = -100

        return pixel_values_tensor, labels, list(captions), list(img_ids)


if __name__ == "__main__":
    set_seed(42)
    nltk.download('punkt', quiet=True)

    TEACHER_ENCODER = 'google/vit-base-patch16-224'
    STUDENT_ENCODER = 'google/vit-base-patch16-224'
    DECODER = 'gpt2'

    tokenizer = GPT2TokenizerFast.from_pretrained(DECODER)
    tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.bos_token is None:
        tokenizer.add_special_tokens({'bos_token': '[BOS]'})

    teacher_processor = create_image_processor(TEACHER_ENCODER)

    print("Loading datasets...")
    train_dataset = CocoCaptionsDataset(
        annotation_file=os.path.join(COCO_ROOT_PATH, "annotations/captions_train2017.json"),
        image_dir=os.path.join(COCO_ROOT_PATH, "train2017"), tokenizer=tokenizer
    )
    val_dataset = CocoCaptionsDataset(
        annotation_file=os.path.join(COCO_ROOT_PATH, "annotations/captions_val2017.json"),
        image_dir=os.path.join(COCO_ROOT_PATH, "val2017"), tokenizer=tokenizer
    )
    print("Dataset loading complete.")

    train_subset = Subset(train_dataset, list(range(int(K * len(train_dataset)))))
    val_subset = Subset(val_dataset, list(range(int(K * len(val_dataset)))))

    collate_fn = CollateFn(teacher_processor, tokenizer)
    train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, collate_fn=collate_fn)

    print(f"\nUsing {len(train_subset)} training samples and {len(val_subset)} validation samples.")
    print(f"Device Used: {DEVICE}\n")


    print("--- Step 1: Training Teacher Model on Real Data ---")
    teacher_model = create_model(TEACHER_ENCODER, DECODER, tokenizer).to(DEVICE)
    optimizer_teacher = optim.Adam(teacher_model.parameters(), lr=TEACHER_LR)
    teacher_trainer = Trainer(teacher_model, optimizer_teacher, tokenizer, DEVICE, name="Teacher")

    start_time_teacher = time.time()
    for epoch in range(1, TEACHER_TRAINING_EPOCHS + 1):
        teacher_trainer.train_epoch(train_loader, epoch)
    time_teacher = time.time() - start_time_teacher

    print("\nEvaluating Teacher Model...")
    teacher_bleu, teacher_cider = teacher_trainer.evaluate(val_loader, "Teacher on Real Val Set")
    torch.save(teacher_model.state_dict(), "teacher_model.pth")

    print("\n--- Step 2: Distilling Dataset using the Teacher Model ---")
    distiller = DatasetDistiller(
        teacher_model=teacher_model,
        real_data_loader=train_loader,
        tokenizer=tokenizer,
        device=DEVICE,
        num_synthetic_samples=NUM_SYNTHETIC_IMAGES,
        student_encoder_name=STUDENT_ENCODER,
        student_decoder_name=DECODER
    )
    synthetic_image_dataset = distiller.synthesize_data(OUTER_LOOP_EPOCHS)


    print("\n--- Step 3: Training Student Model on Synthetic Data ---")
    student_model = create_model(STUDENT_ENCODER, DECODER, tokenizer).to(DEVICE)
    optimizer_student = optim.Adam(student_model.parameters(), lr=STUDENT_LR)
    student_trainer = Trainer(student_model, optimizer_student, tokenizer, DEVICE, name="Student")

    final_train_dataset = SyntheticLabelledDataset(synthetic_image_dataset, train_dataset)
    student_collate_fn = StudentCollateFn(tokenizer)
    synthetic_loader = DataLoader(final_train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                                  collate_fn=student_collate_fn, num_workers=4)

    start_time_student = time.time()
    for epoch in range(1, STUDENT_TRAINING_EPOCHS + 1):
        student_trainer.train_epoch(synthetic_loader, epoch)
    time_student = time.time() - start_time_student

    print("\nEvaluating Student Model...")
    student_bleu, student_cider = student_trainer.evaluate(val_loader, "Student on Real Val Set")


    print("\n\n" + "=" * 20 + " FINAL SUMMARY " + "=" * 20)
    print(f"Real Training Set Size: {len(train_subset)}")
    print(f"Distilled (Synthetic) Set Size: {NUM_SYNTHETIC_IMAGES}")
    if len(train_subset) > 0:
        print(f"Compression Ratio: {(NUM_SYNTHETIC_IMAGES / len(train_subset)) * 100:.2f}%\n")

    print("--- Performance Comparison on Real Validation Set ---")
    print(
        f"Teacher (trained on {len(train_subset)} real samples)      - BLEU: {teacher_bleu:.4f}, CIDEr: {teacher_cider:.4f}")
    print(
        f"Student (trained on {NUM_SYNTHETIC_IMAGES} synthetic samples) - BLEU: {student_bleu:.4f}, CIDEr: {student_cider:.4f}")
    if teacher_cider > 0:
        print(f"  -> Student performance is {(student_cider / teacher_cider) * 100:.2f}% of the Teacher's performance.")

    print("\n--- Training Times (seconds) ---")
    print(f"Teacher Training Time:   {time_teacher:.2f}s")
    print(f"Student Training Time:   {time_student:.2f}s (on synthetic data)")

    print("\n" + "=" * 65)