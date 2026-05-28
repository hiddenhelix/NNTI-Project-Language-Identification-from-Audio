#!/usr/bin/env python3


# convert it to ipynb to run on colab

# !pip install evaluate               # uncomment it to run code on colab

#%%
import os
import sys
import random
from datetime import datetime
from typing import Any, Dict, List
from collections import Counter, defaultdict

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
os.makedirs("./indic-SLID", exist_ok=True)
os.makedirs("./indic-SLID_new", exist_ok=True)

#%%
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Function
import wandb
import torchaudio

from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from datasets import load_dataset, Audio
from transformers import (
    AutoModelForAudioClassification,
    AutoFeatureExtractor,
    AutoConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    set_seed,
)
from huggingface_hub import login
import evaluate

set_seed(42)

#%%
current_time_str = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"Current time: {current_time_str}")


print("Check if GPU available:")
print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
print(f"torch.cuda.get_device_name(): {torch.cuda.get_device_name()}")



#%%
HF_TOKEN = os.environ.get("HF_TOKEN", "xxxxxxxxxxxxxxxxx")    # add huggingface token
WANDB_KEY = os.environ.get(
    "WANDB_API_KEY",
    "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"    # add wandb key  to record your stats

)
login(token=HF_TOKEN)
wandb.login(key=WANDB_KEY)



#%%
# Gradient Reversal Layer 
class GradientReversalFunction(Function):
    @staticmethod    
    def forward(ctx, x, lambda_):
        ctx.save_for_backward(torch.tensor(lambda_)) 
        return x.clone()       

    @staticmethod
    def backward(ctx, grad_output):
        lambda_ = ctx.saved_tensors[0].item()
        return -lambda_ * grad_output, None


def grad_reverse(x, lambda_=1.0):
    return GradientReversalFunction.apply(x, lambda_)



#%%
# Wrapper model that adds a speaker classification head on top of the base audio classifier
class DANNModel(nn.Module):
    def __init__(self, base_model, num_speakers, dann_lambda=0.0):
        super().__init__()
        hidden_size = base_model.config.hidden_size
        self.base_model = base_model
        self.dann_lambda = dann_lambda
        self.config = base_model.config


        # small MLP to predict speaker from reversed embeddings
        self.speaker_head = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_speakers),
        )

    def forward(
        self,
        input_values=None,
        attention_mask=None,
        labels=None,
        speaker_label=None,
        **kwargs,
    ):
        # only compute hidden states if we have speaker labels
        need_hidden = speaker_label is not None

        outputs = self.base_model(
            input_values=input_values,
            attention_mask=attention_mask,
            output_hidden_states=need_hidden,
            **kwargs,
        )

        logits = outputs.logits  

        loss = None
        if labels is not None:
            lang_loss = nn.CrossEntropyLoss()(logits, labels)

            if need_hidden and outputs.hidden_states is not None:
                # grab the last transformer layer output and mean-pool across time
                hidden = outputs.hidden_states[-1]      
                pooled = hidden.mean(dim=1)             

                # reverse the gradient 
                reversed_pooled = grad_reverse(pooled, self.dann_lambda)
                speaker_logits = self.speaker_head(reversed_pooled)
                spk_loss = nn.CrossEntropyLoss()(speaker_logits, speaker_label)

                # total loss = language classification + adversarial speaker loss
                loss = lang_loss + spk_loss
            else:
                loss = lang_loss

        return type(outputs)(
            loss=loss,
            logits=logits,
            hidden_states=None,
        )

#%%
# Redefine label mappings and model config
class DANNTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        if self.state.max_steps > 0:
            p = self.state.global_step / self.state.max_steps
        else:
            p = 0.0

        # sigmoid schedule for lambda
        lambda_ = float(2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0)

        if hasattr(model, "dann_lambda"):
            model.dann_lambda = lambda_

        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


#%%
# Audio augmentation functions 
def pitch_shift(waveform: np.ndarray, sample_rate=16000) -> np.ndarray:
    # Randomly shifts pitch up or down by a few semitones.
    semitones = random.uniform(-4, 4)
    wav_tensor = torch.from_numpy(waveform).float()
    if wav_tensor.dim() == 1:
        wav_tensor = wav_tensor.unsqueeze(0)
    shifted = torchaudio.functional.pitch_shift(wav_tensor, sample_rate, n_steps=semitones)
    return shifted.squeeze().numpy()


def speed_perturbation(waveform: np.ndarray, sample_rate=16000) -> np.ndarray:
    # Speed up or slow down speech (0.85x to 1.15x) without changing pitch.
    factor = random.uniform(0.85, 1.15)
    wav_tensor = torch.from_numpy(waveform).float()
    if wav_tensor.dim() == 1:
        wav_tensor = wav_tensor.unsqueeze(0)
    new_sr = int(sample_rate * factor)
    r1 = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sr)
    r2 = torchaudio.transforms.Resample(orig_freq=new_sr, new_freq=sample_rate)
    return r2(r1(wav_tensor)).squeeze().numpy()


def add_noise(waveform: np.ndarray) -> np.ndarray:
    # Adds gaussian noise at a random SNR between 15-30 dB.
    snr_db = random.uniform(15, 30)
    signal_power = np.mean(waveform ** 2) + 1e-10
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, np.sqrt(noise_power), waveform.shape)
    return (waveform + noise).astype(np.float32)


def time_masking(waveform: np.ndarray) -> np.ndarray:
    # Zeros out a random chunk of audio (SpecAugment-style, but in time domain).
    length = len(waveform)
    mask_length = int(length * random.uniform(0.02, 0.1))
    start = random.randint(0, max(0, length - mask_length))
    augmented = waveform.copy()
    augmented[start:start + mask_length] = 0.0
    return augmented


def gain_perturbation(waveform: np.ndarray) -> np.ndarray:
    # Random volume boost or cut, -6 to +6 dB.
    gain_db = random.uniform(-6, 6)
    return (waveform * (10 ** (gain_db / 20))).astype(np.float32)


def augment_audio(waveform: np.ndarray, sample_rate=16000) -> np.ndarray:
    # randomly apply one of the speaker-altering augmentations
    speaker_altering = [pitch_shift, speed_perturbation]
    waveform = random.choice(speaker_altering)(waveform, sample_rate)

    # layer on additional augmentations probabilistically
    if random.random() < 0.5:
        waveform = add_noise(waveform)
    if random.random() < 0.3:
        waveform = time_masking(waveform)
    if random.random() < 0.3:
        waveform = gain_perturbation(waveform)
    return waveform


# Probability of applying augmentation to a given sample during training. 
aug_prob = 0.7



#%%
batch_size = 8
gradient_accumulation_steps = 4  
num_train_epochs = 15
lr = 0.00005



#%%
# Model selection
model_id = "facebook/mms-300m"
#model_id = "facebook/w2v-bert-2.0"
#model_id = "utter-project/mHuBERT-147"
#model_id = "facebook/wav2vec2-xls-r-300m"


#%%
feature_extractor = AutoFeatureExtractor.from_pretrained(
    model_id,
    do_normalize=True,
    return_attention_mask=True,
)

#%%
# Load dataset and check structure
dataset = load_dataset("badrex/nnti-dataset-full")
print(f"dataset['train']: {dataset['train']}")



#%%
print("Data Constraints Check:")
language_counts = Counter(dataset['train']['language'])
print(f"\nTotal training samples: {len(dataset['train'])}")
print(f"Total languages: {len(language_counts)}")
print(f"\nSamples per language:")
for lang, count in sorted(language_counts.items()):
    print(f"  {lang}: {count}")

speakers_per_lang = defaultdict(set)
for lang, sid in zip(dataset['train']['language'], dataset['train']['speaker_id']):
    speakers_per_lang[lang].add(sid)
print(f"\nSpeakers per language:")
for lang, speakers in sorted(speakers_per_lang.items()):
    print(f"  {lang}: {len(speakers)} speakers")
print(f"\nTotal unique speakers: {len(set(dataset['train']['speaker_id']))}")


#%%
# check the strucutre of one training sample (before decoding)
print(f"dataset['train'][0]: {dataset['train'][0]}")


#%%
# shuffle the dataset 
train_ds = dataset['train'].shuffle(seed=42)
valid_ds = dataset['validation'].shuffle(seed=42)

#%%
# resample to 16kHz
train_ds = train_ds.cast_column("audio_filepath", Audio(sampling_rate=16000))
valid_ds = valid_ds.cast_column("audio_filepath", Audio(sampling_rate=16000))

#%%
# based on the model type, set input features key
if model_id == "facebook/w2v-bert-2.0":
    input_features_key = "input_features"
else:
    input_features_key = "input_values"

#%%
max_duration = 7


#%%
# get the set of languages
LABELS = train_ds.unique('language')
sorted_labels = sorted(l.upper() for l in LABELS)
print(f"Languages: {sorted_labels}")

str_to_int = {s: i for i, s in enumerate(LABELS)}
int_to_str = {i: s for s, i in str_to_int.items()}
num_labels = len(str_to_int)


#%%
# get the set of speakers across both train and validation to build speaker classification head
all_speakers = sorted(set(train_ds["speaker_id"]) | set(valid_ds["speaker_id"]))
spk_to_int = {s: i for i, s in enumerate(all_speakers)}
num_speakers = len(spk_to_int)
print(f"Total unique speakers (train + val): {num_speakers}")


#%%
def preprocess_function(examples, augment=False):
    audio_arrays = [x["array"] for x in examples["audio_filepath"]]

    if augment:
        augmented = []
        for arr in audio_arrays:
            if random.random() < aug_prob:
                try:
                    arr = augment_audio(arr, sample_rate=feature_extractor.sampling_rate)
                except Exception:
                    # if augmentation fails for some reason, just use the original
                    pass
            augmented.append(arr)
        audio_arrays = augmented

    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        truncation=True,
        max_length=int(feature_extractor.sampling_rate * max_duration),
        return_attention_mask=True,
    )

    # encode language labels and speaker labels as integers
    inputs["label"] = [str_to_int[x] for x in examples["language"]]
    inputs["speaker_label"] = [spk_to_int[x] for x in examples["speaker_id"]]
    inputs[input_features_key] = [np.array(x) for x in inputs[input_features_key]]
    inputs["length"] = [len(f) for f in inputs[input_features_key]]

    return inputs

#%%
# encode both splits — this takes a while because of audio loading + augmentation
keep_cols = ['speaker_id', 'language']


# %% [markdown]
# ## encode the train and valid splits 


#%%

train_ds_encoded = train_ds.map(
    lambda examples: preprocess_function(examples, augment=True),
    remove_columns=[c for c in train_ds.column_names if c not in keep_cols],
    batched=True,
    batch_size=32,
)


#%%
valid_ds_encoded = valid_ds.map(
    lambda examples: preprocess_function(examples, augment=False),
    remove_columns=[c for c in valid_ds.column_names if c not in keep_cols],
    batched=True,
    batch_size=32,
)

# save to disk so we don't have to redo this if training crashes
SAVE_DIR = os.environ.get("SAVE_DIR", "./indic-SLID_new")
os.makedirs(SAVE_DIR, exist_ok=True)
train_ds_encoded.save_to_disk(os.path.join(SAVE_DIR, "nnti_dann_train_encoded"))
valid_ds_encoded.save_to_disk(os.path.join(SAVE_DIR, "nnti_dann_valid_encoded"))
print("Datasets saved to", SAVE_DIR)


#%%

config = AutoConfig.from_pretrained(model_id)
config.num_labels = num_labels
config.label2id = str_to_int
config.id2label = int_to_str


#%%
# apply dropout to various parts of the model to help DANN generalize better
config.hidden_dropout = 0.1
config.attention_dropout = 0.1
config.activation_dropout = 0.1
config.feat_proj_dropout = 0.1


#%%
# load pretrained model and wrap with DANNModel to add speaker classification head
base_model = AutoModelForAudioClassification.from_pretrained(model_id, config=config)
slid_model = DANNModel(base_model, num_speakers=num_speakers)


#%%
# check total parameters
total = sum(p.numel() for p in slid_model.parameters())
print(f"Total parameters: {total:,}")
print(f"Under 600M limit: {total < 600_000_000}")

#%%
# custom data collator to handle the extra speaker labels and padding
class AudioDataCollatorWithSpeaker:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        batch = {
            input_features_key: [f[input_features_key] for f in features],
            "attention_mask": [f["attention_mask"] for f in features],
        }
        batch = self.feature_extractor.pad(batch, padding=True, return_tensors="pt")

        batch["labels"] = torch.tensor(
            [f["label"] for f in features], dtype=torch.long
        )
        batch["speaker_label"] = torch.tensor(
            [f["speaker_label"] for f in features], dtype=torch.long
        )
        return batch



#%%
data_collator = AudioDataCollatorWithSpeaker(feature_extractor)

#%%
# initialize wandb run for logging
wandb.init(project="Indic-SLID", name=f"DANN_{model_id}_{lr}_{current_time_str}")


#%%
training_args = TrainingArguments(
    output_dir=f"./indic-SLID_new/checkpoints_{model_id.replace('/', '_')}_dann",
    report_to="wandb",
    logging_steps=1,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    eval_strategy="steps",
    eval_steps=100,
    save_strategy="steps",
    save_steps=100,
    learning_rate=lr,
    gradient_accumulation_steps=gradient_accumulation_steps,
    num_train_epochs=num_train_epochs,
    weight_decay=0.01,
    warmup_ratio=0.08,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    greater_is_better=True,
    save_total_limit=2,
    fp16=True,
    lr_scheduler_type="cosine",
    remove_unused_columns=False,
    label_names=["labels"],
    eval_accumulation_steps=4,  
    push_to_hub=False,
)


#%%
# load accuracy metric for evaluation
accuracy_metric = evaluate.load("accuracy")


#%%
def compute_metrics(eval_pred):
    """Computes accuracy for language classification"""
    predictions = np.argmax(eval_pred.predictions, axis=1)
    label_ids = eval_pred.label_ids
    if isinstance(label_ids, (tuple, list)):
        label_ids = label_ids[0]
    return accuracy_metric.compute(predictions=predictions, references=label_ids)


#%%
trainer = DANNTrainer(
    slid_model,
    training_args,
    train_dataset=train_ds_encoded,
    eval_dataset=valid_ds_encoded,
    processing_class=feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=7)],
)

#%%
print("Train loop starting...")
trainer.train()

# %%
# push model to hub 
# slid_model.push_to_hub(
#     "your-hf-account/indic-language-identification"
# )

#%%
print("Final evaluation starting...")
final_metrics = trainer.evaluate()
print("Final metrics:", final_metrics)
print("Best checkpoint:", trainer.state.best_model_checkpoint)


#%%
predictions = trainer.predict(valid_ds_encoded)
preds = np.argmax(predictions.predictions, axis=1)
# guard against tuple/list format for label_ids
label_ids = predictions.label_ids
if isinstance(label_ids, (tuple, list)):
    label_ids = label_ids[0]


#%%
pred_names = [int_to_str[p] for p in preds]
label_names = [int_to_str[l] for l in label_ids]


#%%
report = classification_report(label_names, pred_names, digits=4)

print("Per-Language Classification Report (DANN):")
print(report)


#%%
with open("./indic-SLID_new/classification_report_dann.txt", "w") as f:
    f.write(report)

#%%
# confusion matrix plotting
sorted_label_names = sorted(int_to_str.values())
cm = confusion_matrix(label_names, pred_names, labels=sorted_label_names)

fig, ax = plt.subplots(figsize=(16, 14))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=sorted_label_names)
disp.plot(ax=ax, cmap="Blues", xticks_rotation=45, values_format="d")
ax.set_title(
    f"Language ID Confusion Matrix (DANN) | "
    f"Acc: {final_metrics.get('eval_accuracy', 'N/A'):.4f}"
)
plt.tight_layout()
plt.savefig("./indic-SLID_new/confusion_matrix_dann.png", dpi=150)
plt.close()
print("Saved confusion matrix to ./indic-SLID_new/confusion_matrix_dann.png")


#%%
# save just the base model (without the speaker head) for downstream use
save_dir = "./indic-SLID_new/dann_model"
slid_model.base_model.save_pretrained(save_dir)
print(f"Model saved to {save_dir}")


#%%
# push results to wandb
artifact = wandb.Artifact("dann-results", type="results")
artifact.add_file("./indic-SLID_new/classification_report_dann.txt")
artifact.add_file("./indic-SLID_new/confusion_matrix_dann.png")
wandb.log_artifact(artifact)
wandb.log({
    "confusion_matrix": wandb.Image("./indic-SLID_new/confusion_matrix_dann.png"),
    "final_accuracy": final_metrics["eval_accuracy"],
})
wandb.finish()




#%%
TASK3_DIR = "./indic-SLID_new/task3_plots_dann"
os.makedirs(TASK3_DIR, exist_ok=True)


print("TASK 3: Extracting embeddings...")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# load the saved model fresh 
analysis_model = AutoModelForAudioClassification.from_pretrained(save_dir)
analysis_model = analysis_model.to(DEVICE)
analysis_model.eval()


#%%
embeddings_list = []
def hook_fn(module, input, output):
    hidden = output[0] if isinstance(output, tuple) else output
    embeddings_list.append(hidden.mean(dim=1).detach().cpu().float().numpy())

# figure out the right layer depending on model architecture
if hasattr(analysis_model, 'wav2vec2'):
    last_layer = analysis_model.wav2vec2.encoder.layers[-1]
elif hasattr(analysis_model, 'wav2vec2_bert'):
    last_layer = analysis_model.wav2vec2_bert.encoder.layers[-1]
else:
    last_layer = list(analysis_model.modules())[-3]

hook_handle = last_layer.register_forward_hook(hook_fn)
print(f"Hook registered on: {last_layer.__class__.__name__}")


#%%
# run inference on the entire validation set in batches of 8
all_preds2 = []
all_labels2 = []
all_speaker_ids = []
all_languages2 = []

for i in range(0, len(valid_ds), 8):
    batch_samples = valid_ds.select(range(i, min(i + 8, len(valid_ds))))
    audio_arrays = [x["array"] for x in batch_samples["audio_filepath"]]
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        truncation=True,
        max_length=int(feature_extractor.sampling_rate * max_duration),
        return_attention_mask=True,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        outputs = analysis_model(**inputs)
    preds2 = torch.argmax(outputs.logits, dim=-1).cpu().numpy()
    all_preds2.extend(preds2.tolist())
    all_labels2.extend([str_to_int[l] for l in batch_samples["language"]])
    all_speaker_ids.extend(batch_samples["speaker_id"])
    all_languages2.extend(batch_samples["language"])

    # progress reporting every 20 batches
    if (i // 8) % 20 == 0:
        print(f"  {min(i + 8, len(valid_ds))}/{len(valid_ds)} samples")

hook_handle.remove()
embeddings = np.vstack(embeddings_list)
all_labels2 = np.array(all_labels2)
print(f"Embeddings shape: {embeddings.shape}")



#%%
# t-SNE visualization of embeddings colored by language and speaker
print("Running t-SNE (~5 mins)...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)
embeddings_2d = tsne.fit_transform(embeddings)
print("t-SNE done")

# colored by language
cmap = plt.cm.get_cmap('tab20', 22)
fig, ax = plt.subplots(figsize=(16, 12))
for idx, lang in enumerate(sorted_label_names):
    mask = np.array(all_languages2) == lang
    ax.scatter(
        embeddings_2d[mask, 0], embeddings_2d[mask, 1],
        label=lang, color=cmap(idx), alpha=0.6, s=20
    )
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
ax.set_title("t-SNE — Colored by Language (DANN)\n(Good: 22 distinct clusters)")
plt.tight_layout()
plt.savefig(f"{TASK3_DIR}/tsne_by_language.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved t-SNE by language")


# colored by speaker
unique_speakers = sorted(set(all_speaker_ids))
spk_to_idx = {s: i for i, s in enumerate(unique_speakers)}
cmap_spk = plt.cm.get_cmap('tab20', min(len(unique_speakers), 20))
fig, ax = plt.subplots(figsize=(16, 12))
for spk in unique_speakers[:20]:
    mask = np.array(all_speaker_ids) == spk
    ax.scatter(
        embeddings_2d[mask, 0], embeddings_2d[mask, 1],
        label=spk[:8], color=cmap_spk(spk_to_idx[spk] % 20), alpha=0.6, s=20
    )
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7)
ax.set_title("t-SNE — Colored by Speaker (DANN)\n(Good: random scatter | Bad: clusters = bias)")
plt.tight_layout()
plt.savefig(f"{TASK3_DIR}/tsne_by_speaker.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved t-SNE by speaker")



#%%
# Probing classifiers for speaker and language information in the embeddings
le = LabelEncoder()
speaker_ints = le.fit_transform(all_speaker_ids)
probe = LogisticRegression(max_iter=1000, random_state=42)
spk_scores = cross_val_score(probe, embeddings, speaker_ints, cv=5, scoring='accuracy')

lang_probe = LogisticRegression(max_iter=1000, random_state=42)
lang_scores = cross_val_score(lang_probe, embeddings, all_labels2, cv=5, scoring='accuracy')

print(f"Speaker probe:  {spk_scores.mean():.4f} +/- {spk_scores.std():.4f}")
print(f"Language probe: {lang_scores.mean():.4f} +/- {lang_scores.std():.4f}")

if spk_scores.mean() > 0.5:
    print("HIGH speaker probe — bias still present")
else:
    print("LOW speaker probe — successfully removed speaker bias!")

probe_text = (
    f"Speaker probe (5-fold CV):  {spk_scores.mean():.4f} +/- {spk_scores.std():.4f}\n"
    f"Language probe (5-fold CV): {lang_scores.mean():.4f} +/- {lang_scores.std():.4f}\n\n"
    f"Interpretation:\n"
    f"  Speaker HIGH (>0.5) = bias present\n"
    f"  Speaker LOW  (<0.5) = DANN worked!\n"
    f"  Language should always be HIGH\n"
)
with open(f"{TASK3_DIR}/speaker_probe_results.txt", "w") as f:
    f.write(probe_text)
print("Saved speaker probe results")

#%%
# push results to wandb
wandb.init(project="Indic-SLID", name=f"DANN_Task3_{current_time_str}", resume="allow")
artifact = wandb.Artifact("dann-task3-results", type="results")
artifact.add_file(f"{TASK3_DIR}/tsne_by_language.png")
artifact.add_file(f"{TASK3_DIR}/tsne_by_speaker.png")
artifact.add_file(f"{TASK3_DIR}/speaker_probe_results.txt")
wandb.log_artifact(artifact)
wandb.log({
    "tsne_language": wandb.Image(f"{TASK3_DIR}/tsne_by_language.png"),
    "tsne_speaker": wandb.Image(f"{TASK3_DIR}/tsne_by_speaker.png"),
    "speaker_probe_acc": float(spk_scores.mean()),
    "language_probe_acc": float(lang_scores.mean()),
})
wandb.finish()
print("Results uploaded to WandB")


#%%
# final summary
print(f"  Final accuracy:      {final_metrics['eval_accuracy']:.4f}")
print(f"  Speaker probe acc:   {spk_scores.mean():.4f}  "
      f"({'BAD' if spk_scores.mean() > 0.5 else 'GOOD'})")
print(f"  Language probe acc:  {lang_scores.mean():.4f}")
print(f"  Results saved to:    {TASK3_DIR}/")



# %%
exit(0)