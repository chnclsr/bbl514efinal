from transformers import VisionEncoderDecoderModel, ViTImageProcessor, GPT2TokenizerFast


def create_model(encoder_model_name, decoder_model_name, tokenizer):
    """
    Hugging Face'in standart VisionEncoderDecoderModel'ini oluşturan yardımcı fonksiyon.
    Bu, görüntü kodlayıcıyı (ViT/ResNet) ve metin üreticiyi (GPT-2) birleştirir.
    """
    model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(
        encoder_model_name, decoder_model_name
    )

    # --- Tokenizer ve Model Konfigürasyonunu Eşleştir ---

    # Modelin, tokenizer'ın özel token'larını tanımasını sağla
    model.config.decoder.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id  # .generate() için ana config'e de ekle
    model.config.pad_token_id = tokenizer.pad_token_id  # .generate() için ana config'e de ekle

    # --- ANA DÜZELTME: Decoder için başlangıç token'ını ayarla ---
    # Bu satır, "ValueError: Make sure to set the decoder_start_token_id..." hatasını çözer.
    model.config.decoder_start_token_id = tokenizer.bos_token_id

    # Decoder'ın kelime dağarcığı boyutunu, bizim tokenizer'ımızla aynı yap
    model.decoder.resize_token_embeddings(len(tokenizer))

    # Cross-attention katmanlarını etkinleştir (Bu zaten varsayılan olarak yapılır ama emin olalım)
    model.config.decoder.add_cross_attention = True

    return model


def create_image_processor(encoder_model_name):
    """
    Modele uygun görüntü işlemcisini oluşturur.
    """
    return ViTImageProcessor.from_pretrained(encoder_model_name)