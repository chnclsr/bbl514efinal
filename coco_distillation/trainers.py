import torch
from collections import defaultdict
from tqdm import tqdm
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from pycocoevalcap.cider.cider import Cider


class Trainer:
    def __init__(self, model, optimizer, tokenizer, device, name="Model"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.device = device
        self.name = name

    def train_epoch(self, data_loader, epoch):
        self.model.train()
        total_loss = 0.0

        for pixel_values, labels, _, _ in tqdm(data_loader, desc=f"{self.name} Training Epoch {epoch}"):
            pixel_values = pixel_values.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(pixel_values=pixel_values, labels=labels)
            loss = outputs.loss

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        print(f"Epoch {epoch} Average Training Loss: {avg_loss:.4f}")
        return avg_loss

    def evaluate(self, data_loader, epoch_str="Final"):
        self.model.eval()
        all_refs = defaultdict(list)
        all_hyps = {}

        # Değerlendirme için tek ve temiz bir döngü
        for pixel_values, _, captions, img_ids in tqdm(data_loader, desc=f"{self.name} Evaluation"):
            pixel_values = pixel_values.to(self.device)

            # --- KALICI ÇÖZÜM: `generate` çağrısını hatasız parametrelerle yap ---
            # `num_beams` parametresi tamamen kaldırıldı. Bu, `NotImplementedError` hatasını çözer.
            # Bu çağrı artık basit ve hızlı "greedy search" yöntemini kullanır.
            output_ids = self.model.generate(
                pixel_values,
                max_length=50,
            )

            batch_hyps = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

            # Referansları ve Hipotezleri topla
            for i, img_id in enumerate(img_ids):
                all_refs[img_id].append(captions[i])
                if img_id not in all_hyps:
                    all_hyps[img_id] = [batch_hyps[i].strip()]

        # Skorlama için anahtarları eşleştir
        filtered_refs = {k: v for k, v in all_refs.items() if k in all_hyps}

        if not filtered_refs:
            print("Evaluation failed: No matching references found for generated hypotheses.")
            return 0.0, 0.0

        cider_scorer = Cider()
        cider_score, _ = cider_scorer.compute_score(filtered_refs, all_hyps)

        bleu_refs, bleu_hyps = [], []
        for img_id in all_hyps.keys():
            bleu_refs.append([ref.split() for ref in filtered_refs[img_id]])
            bleu_hyps.append(all_hyps[img_id][0].split())

        chencherry = SmoothingFunction()
        bleu = corpus_bleu(bleu_refs, bleu_hyps, smoothing_function=chencherry.method1)

        print(f"{self.name} {epoch_str} Evaluation: BLEU: {bleu:.4f}, CIDEr: {cider_score:.4f}")
        return bleu, cider_score