# Subject Prediction with STW Filtering

This project focuses on running a subject prediction model using a filtered dataset based on the **STW (Standard Thesaurus for Economics)** subject taxonomy. The process includes preparing the dataset, filtering out irrelevant subjects, and experimenting with prompt-based and model-based approaches for subject classification.

---

## 📁 Project Setup on KDSRV03

Follow these steps to run the project on the university server `KDSRV03`.

### 🔐 Connect to Server

* If off-campus, connect via **FortiClient VPN**.
* SSH into the server using your university credentials ([connect via SSH](https://www.hiperf.rz.uni-kiel.de/caucluster/access/#user-account)).

### 📂 Navigate to Project Directory

```bash
cd /data2/z2/stu213218/subject-prediction
```

### 🐍 Activate Virtual Environment

```bash
source modelenv/bin/activate
```

### 📦 Install Dependencies

```bash
pip3 install -r requirements.txt
```

---

## 📊 Dataset Preparation

### 1. Check Dataset Content

Check whether abstracts contain any valid STW subject terms.

```bash
python3 src/subject_match.py
```

### 2. Filter STW-Only Subjects

Filter out data samples that contain only valid STW terms.

```bash
python3 filter_subject_matching.py
```

### 3. Reformat Validation Dataset

Convert the `val.json` format to a JSONL format (`val_fixed.json`) for evaluation:

```bash
python3 src/reformat_val_data.py
```

---

## 🤖 Inference

Run inference on a sample abstract using the trained model:

```bash
python3 src/inference/inference.py
```

---

## 🧪 Evaluation

Evaluate the model performance using precision, recall, F1 score, and hamming loss:

```bash
python3 src/evaluation/evaluate_model.py
```

Results are saved to:

```
data/processed/evaluation_results.json
```

---

## 📂 File Structure Overview

* `data/processed/val.json` – Raw validation input with abstract & labels.
* `data/processed/val_fixed.json` – Reformatted NDJSON version for evaluation.
* `data/processed/trained_model/` – Directory with tokenizer and model.
* `data/processed/label_encoder.pkl` – MultiLabelBinarizer encoder used for label transformation.

---

## 🔁 Git Workflow

```bash
git add .
git commit -m "Update README with full pipeline documentation"
git push
```

---

For further information or debugging, please check relevant scripts in:

* `src/inference/`
* `src/evaluation/`
* `src/data_preparation/`
