import torch
# --- SETTINGS ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 224
EMBED_DIM = 512
BATCH_SIZE = 16
MAX_TEXT_LEN = 16
COCO_ROOT_PATH = "../coco_distillation"
TEACHER_LR = 1e-3
STUDENT_LR = 1e-3
SYNTHETIC_LR = 1e-3
CONTRASTIVE_TEMP = 0.07
TEACHER_TRAINING_EPOCHS = 1
STUDENT_TRAINING_EPOCHS = 1
OUTER_LOOP_EPOCHS = 1
INNER_LOOP_STEPS = 1
# Calculate the number f samples for (K * 100)%
K = 0.01
NUM_SYNTHETIC_IMAGES = 250

