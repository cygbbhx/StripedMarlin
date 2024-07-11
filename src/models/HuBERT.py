from transformers import AutoModelForAudioClassification

def get_model():
    model = AutoModelForAudioClassification.from_pretrained("superb/hubert-base-superb-ks",
                                                            num_labels = 2,
                                                            ignore_mismatched_sizes=True
                                                        )
    return model