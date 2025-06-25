import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset
from tqdm import tqdm
from torch.backends.cuda import sdp_kernel
from multi_modal import create_model
from constants import *


class DatasetDistiller:
    def __init__(self, teacher_model, real_data_loader, tokenizer, device,
                 num_synthetic_samples, student_encoder_name, student_decoder_name):

        self.teacher_model = teacher_model.to(device)
        self.teacher_model.eval()
        self.real_data_loader = real_data_loader
        self.tokenizer = tokenizer
        self.device = device
        self.student_encoder_name = student_encoder_name
        self.student_decoder_name = student_decoder_name

        self.synthetic_images = nn.Parameter(
            torch.randn(num_synthetic_samples, 3, 224, 224, device=device)
        )
        self.optimizer_synthetic = optim.Adam([self.synthetic_images], lr=SYNTHETIC_LR)
        self.distillation_criterion = nn.L1Loss()

        print(f"[Distiller Init] Ready to generate {num_synthetic_samples} synthetic images.")

    def synthesize_data(self, outer_epochs):
        print("\n[Dataset Distillation Start]")
        real_data_iter = iter(self.real_data_loader)

        for i_outer in tqdm(range(outer_epochs), desc="Distilling Dataset"):
            student_model = create_model(self.student_encoder_name, self.student_decoder_name, self.tokenizer).to(
                self.device)

            try:
                pixel_values_real, labels_real, _, _ = next(real_data_iter)
            except StopIteration:
                real_data_iter = iter(self.real_data_loader)
                pixel_values_real, labels_real, _, _ = next(real_data_iter)

            pixel_values_real = pixel_values_real.to(self.device)
            labels_real = labels_real.to(self.device)

            # --- Öğretmenin Gradyanlarını Hesapla ---
            self.teacher_model.zero_grad()
            loss_teacher = self.teacher_model(pixel_values=pixel_values_real, labels=labels_real).loss
            gw_teacher = torch.autograd.grad(loss_teacher, self.teacher_model.parameters(), allow_unused=True)

            gw_teacher_filtered = {}
            for (name, _), grad in zip(self.teacher_model.named_parameters(), gw_teacher):
                if grad is not None:
                    gw_teacher_filtered[name] = grad.detach()

            # --- Öğrencinin Gradyanlarını Hesapla ve Karşılaştır ---
            batch_size_real = pixel_values_real.shape[0]
            syn_indices = torch.randperm(len(self.synthetic_images))[:batch_size_real]
            pixel_values_syn = self.synthetic_images[syn_indices]

            student_model.zero_grad()

            with sdp_kernel(enable_flash=False, enable_math=True, enable_mem_efficient=False):
                loss_student = student_model(pixel_values=pixel_values_syn, labels=labels_real).loss
                gw_student = torch.autograd.grad(loss_student, student_model.parameters(), create_graph=True,
                                                 allow_unused=True)

            distill_loss = 0
            for (name, param_student), grad_student in zip(student_model.named_parameters(), gw_student):
                if grad_student is not None and name in gw_teacher_filtered:
                    distill_loss += self.distillation_criterion(grad_student, gw_teacher_filtered[name])

            self.optimizer_synthetic.zero_grad()
            if isinstance(distill_loss, torch.Tensor):
                distill_loss.backward()
                self.optimizer_synthetic.step()

        print("[Dataset Distillation Complete]")

        return TensorDataset(self.synthetic_images.detach().cpu())