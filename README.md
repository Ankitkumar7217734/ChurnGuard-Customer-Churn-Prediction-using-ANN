# ChurnGuard — Customer Churn Prediction using ANN

A deep learning project that predicts whether a bank customer will churn (leave the bank) using an Artificial Neural Network (ANN) built with TensorFlow/Keras. The trained model is served through an interactive **Streamlit** web application.

**Live Demo:** [https://churnguard-customer-churn-prediction-using-ann-34hhjk7cdn7q4cg.streamlit.app/](https://churnguard-customer-churn-prediction-using-ann-34hhjk7cdn7q4cg.streamlit.app/)

---

## Project Structure

```
├── app.py                          # Streamlit web application
├── experiments.ipynb               # Model training & experimentation notebook
├── Churn_Modelling.csv             # Dataset (10,000 bank customer records)
├── model.h5                        # Saved trained ANN model
├── label_encoder_gender.pkl        # Saved LabelEncoder for Gender
├── one_hot_encoder_geography.pkl   # Saved OneHotEncoder for Geography
├── scaler.pkl                      # Saved StandardScaler
├── requirements.txt                # Python dependencies
└── logs/
    └── fit/                        # TensorBoard training logs
```

---

## Dataset

**File:** `Churn_Modelling.csv`

The dataset contains 10,000 bank customer records with the following features:

| Feature         | Description                            |
| --------------- | -------------------------------------- |
| CreditScore     | Customer's credit score                |
| Geography       | Country (France, Germany, Spain)       |
| Gender          | Male / Female                          |
| Age             | Customer's age                         |
| Tenure          | Years as a bank customer (0–10)        |
| Balance         | Account balance                        |
| NumOfProducts   | Number of bank products used (1–4)     |
| HasCrCard       | Has a credit card (1 = Yes, 0 = No)    |
| IsActiveMember  | Active member status (1 = Yes, 0 = No) |
| EstimatedSalary | Estimated annual salary                |
| Exited          | **Target** — churned (1) or not (0)    |

Columns dropped during preprocessing: `RowNumber`, `CustomerId`, `Surname`

---

## Model Architecture

An ANN built using TensorFlow/Keras `Sequential` API:

```
Input Layer  →  Dense(64, activation='relu')
             →  Dense(32, activation='relu')
             →  Dense(1,  activation='sigmoid')
```

- **Optimizer:** Adam (learning rate = 0.001)
- **Loss:** Binary Crossentropy
- **Metric:** Accuracy
- **Epochs:** Up to 50 (with early stopping)
- **Batch Size:** 32
- **Train/Test Split:** 80% / 20%

### Callbacks

- **EarlyStopping** — monitors `val_loss`, patience = 5, restores best weights
- **TensorBoard** — logs saved to `logs/fit/`

---

## Data Preprocessing

1. **Drop** irrelevant columns: `RowNumber`, `CustomerId`, `Surname`
2. **Label Encode** `Gender` → `LabelEncoder` (saved as `label_encoder_gender.pkl`)
3. **One-Hot Encode** `Geography` → `OneHotEncoder` (saved as `one_hot_encoder_geography.pkl`)
4. **Scale** all features → `StandardScaler` (saved as `scaler.pkl`)

---

## Installation

1. **Clone the repository** or navigate to the project folder.

2. **Create and activate a virtual environment:**

   ```bash
   python -m venv .venv
   source .venv/bin/activate        # macOS/Linux
   .venv\Scripts\activate           # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Streamlit App

```bash
streamlit run app.py
```

The app will open at **http://localhost:8501**

### Input Fields

| Field              | Type     | Range / Options        |
| ------------------ | -------- | ---------------------- |
| Gender             | Dropdown | Male, Female           |
| Geography          | Dropdown | France, Germany, Spain |
| Age                | Slider   | 18 – 92                |
| Credit Score       | Slider   | 350 – 850              |
| Tenure             | Slider   | 0 – 10                 |
| Number of Products | Slider   | 1 – 4                  |
| Balance            | Number   | ≥ 0.0                  |
| Estimated Salary   | Number   | ≥ 0.0                  |
| Has Credit Card    | Dropdown | Yes, No                |
| Is Active Member   | Dropdown | Yes, No                |

The app outputs the **churn probability** and a prediction label:

- Probability > 0.5 → _"The customer is likely to churn."_
- Probability ≤ 0.5 → _"The customer is unlikely to churn."_

---

## Training the Model

Open and run `experiments.ipynb` to:

- Preprocess the dataset
- Train the ANN model
- Save the model (`model.h5`) and preprocessing artifacts (`.pkl` files)
- Evaluate predictions on sample input

> The notebook was originally developed in **Google Colab**.

---

## Monitoring with TensorBoard

```bash
tensorboard --logdir logs/fit
```

Then open **http://localhost:6006** to visualise training/validation loss and accuracy curves.

---

## Dependencies

```
tensorflow-cpu
pandas
numpy
scikit-learn
streamlit
scikeras
```

Install via:

```bash
pip install -r requirements.txt
```

---

## Tech Stack

- **Deep Learning:** TensorFlow / Keras
- **Data Processing:** Pandas, NumPy, scikit-learn
- **Web App:** Streamlit
- **Monitoring:** TensorBoard
- **Language:** Python 3.12
