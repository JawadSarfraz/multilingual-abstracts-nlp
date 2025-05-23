# Subject Prediction with STW Filtering

This project focuses on running a subject prediction model using a filtered dataset based on the **STW (Standard Thesaurus for Economics)** subject taxonomy. The process includes preparing the dataset, filtering out irrelevant subjects, and experimenting with prompt-based and model-based approaches for subject classification.

---

## 📁 Project Setup on KDSRV03

Follow these steps to run the project on the university server `KDSRV03`.

### 🔐 Connect to Server
- If off-campus, connect via **FortiClient VPN**.
- SSH into the server using your university credentials ([connect via SSH](https://www.hiperf.rz.uni-kiel.de/caucluster/access/#user-account)).

### 📂 Navigate to Project Directory
```bash
cd /data2/z2/stu213218/subject-prediction
```

# Activate Virtual Environment
```bash
source modelenv/bin/activate
```

# Install Dependencies

```bash
pip3 install -r requirements.txt
```

# check the dataset

Check current dataset, first to create sample dataset and then run script to find whether objects contain stw dataset or not. If reture false it means object hasnot stw data in their subjects.


```bash
python3 src/subject_match.py
```

Filter the sample dataset which contains only those subjects present inside stw dataset

```bash
python3 filter_subject_matching.py
```



