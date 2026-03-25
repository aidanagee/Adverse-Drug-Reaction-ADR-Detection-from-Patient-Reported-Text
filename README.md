# Adverse Drug Reaction (ADR) Detection from Patient-Reported Text
### NLP with DistilBERT — Medical Text Classification

Fine-tunes a **DistilBERT** transformer model to automatically detect adverse drug reaction (ADR) mentions in patient-reported social media text. Part of a medical AI project series.

---

## Overview

Adverse Drug Reactions account for over **1 million hospitalizations per year** in the US and are severely underreported through official channels. Patients increasingly describe medication experiences on social media — this model enables large-scale automated pharmacovigilance (drug safety monitoring) that would be impossible to do manually.

This is an active area of real NLP research used by pharmaceutical companies and regulatory agencies including the FDA.

---

## Dataset

**SMM4H 2019 Task 1** — Social Media Mining for Health  
Annotated tweets mentioning medications, labeled for ADR presence.

- ~25,000 annotated tweets
- Binary classification: ADR mentioned vs not mentioned
- ~85% non-ADR / ~15% ADR (severe class imbalance addressed with weighted loss)

| Example Text | Label |
|---|---|
| "Been on Zoloft 3 weeks, insomnia is unbearable" | ADR ✓ |
| "just picked up my prescription" | Not ADR ✗ |
| "metformin making me so nauseous I can barely eat" | ADR ✓ |
| "my doctor prescribed me metformin for diabetes" | Not ADR ✗ |

---

## Model Architecture

- **Base model:** `distilbert-base-uncased` (HuggingFace)
- **Task head:** Linear classification layer on [CLS] token output
- **Parameters:** ~66M (40% smaller than BERT-base, ~97% performance)
- **Fine-tuning:** Full model fine-tuned end-to-end on labeled ADR data

### Why DistilBERT?
Standard keyword matching and bag-of-words approaches fail to capture context and negation critical in medical text. DistilBERT's bidirectional attention correctly handles cases like:
- *"The drug didn't cause nausea"* vs *"The drug caused nausea"*
- *"started feeling sick after my new prescription"* (implicit ADR)

---

## Key Implementation Details

| Component | Choice | Reason |
|---|---|---|
| Loss function | Weighted Cross Entropy | Handles class imbalance — ADR class penalized ~5x more |
| Primary metric | F1 Score | Accuracy misleading with 85/15 split |
| LR Scheduler | Linear warmup + decay | Prevents destabilizing pretrained weights early in training |
| Optimizer | AdamW (lr=2e-5) | Standard for transformer fine-tuning |
| Gradient clipping | max_norm=1.0 | Prevents exploding gradients |

---

## Results

| Metric | Score |
|---|---|
| Accuracy | ~88% |
| F1 Score | ~0.74 |
| Precision | ~0.76 |
| Recall | ~0.72 |

*Results on synthetic demonstration data. Real SMM4H dataset expected to achieve F1 ~0.50-0.55 (competitive with shared task baseline).*

---

## Project Structure

```
adr-detection-distilbert/
├── adr_detection_distilbert.ipynb   # Full notebook with all code and explanations
├── README.md
└── adr_results.png                  # Training curves and confusion matrix (generated)
```

---

## How to Run

```bash
# Install dependencies
pip install torch transformers scikit-learn pandas matplotlib

# Run the notebook
jupyter notebook adr_detection_distilbert.ipynb
```

To use the real SMM4H dataset:
1. Register and download from https://healthlanguageprocessing.org/smm4h-2019/
2. Replace the `get_synthetic_dataset()` function with your data loader
3. Retrain — no other changes needed

---

## Inference

```python
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load trained model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertForSequenceClassification.from_pretrained('./saved_model')

result = predict_adr("horrible rash since starting the new antibiotic", model, tokenizer, device)
# {'label': 'ADR', 'confidence': 0.91, 'adr_probability': 0.91}
```

---

## Medical AI Project Series

This project is part of a broader focus on applying ML to medical problems:

| Project | Task | Architecture |
|---|---|---|
| [Intracranial Aneurysm Detection](https://github.com/aidanagee/aneurysm-detection-rsna) | 3D CT segmentation | 3D U-Net (PyTorch) |
| [Lung Nodule Detection](https://github.com/aidanagee/lung-nodule-detection-3d-unet) | 3D CT segmentation | 3D U-Net (PyTorch) |
| **ADR Detection (this project)** | Medical NLP classification | DistilBERT (HuggingFace) |

---

## References

1. Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers. *NAACL 2019.*
2. Sanh et al. (2019). DistilBERT, a distilled version of BERT. *NeurIPS Workshop 2019.*
3. Weissenbacher et al. (2019). Overview of SMM4H Shared Tasks at ACL 2019.
4. Nikfarjam et al. (2015). Pharmacovigilance from social media. *JAMIA.*

---

## Author

**Aidan Agee** — [GitHub](https://github.com/aidanagee) | [LinkedIn](https://linkedin.com/in/adaadaada)
