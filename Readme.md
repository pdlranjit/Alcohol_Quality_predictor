# 🍷 Wine Quality Prediction — KNN vs Random Forest

## 📌 What This Project Solves
This project predicts the **quality of wine** based on its chemical properties
(alcohol, pH, acidity, citric acid, etc.) using two machine learning models:
- **KNN (K-Nearest Neighbors)**
- **Random Forest Classifier**

Both models are trained, evaluated, and compared to find which one performs better.

---

## 📂 Dataset
- **File:** `WineQT.csv`
- **Input Features:** alcohol, pH, citric acid, volatile acidity, and more
- **Target Column:** `quality` (wine quality score)

---

## 🔄 How the Pipeline Works (Step by Step)

| Step | What Happens |
|------|-------------|
| 1️⃣ Load Data | Reads `WineQT.csv` using pandas |
| 2️⃣ Explore Data | Checks shape, columns, missing values, info |
| 3️⃣ Visualize | Quality distribution, correlation heatmap, alcohol vs quality, outlier detection |
| 4️⃣ Handle Imbalance | Uses **SMOTE** to balance the training classes |
| 5️⃣ Scale Features | Applies **StandardScaler** to normalize data |
| 6️⃣ Train KNN | Finds best K, trains `KNeighborsClassifier` |
| 7️⃣ Train Random Forest | Trains `RandomForestClassifier` (200 trees, balanced weights) |
| 8️⃣ Evaluate Both | Accuracy score + classification report for each model |
| 9️⃣ Compare | Bar chart comparing KNN vs Random Forest accuracy |
| 🔟 Save Models | Saves both models using `joblib` (`.pkl` files) |

---

## 📊 Models Used

### K-Nearest Neighbors (KNN)
- Finds the best value of K automatically
- Trained on SMOTE-balanced data
- Saved as: `my_trained_model.pkl`

### Random Forest Classifier
- 200 decision trees (`n_estimators=200`)
- Class weight set to `balanced`
- Random state: `42`
- Saved as: `rf_model.pkl`

---

## 🗂️ Project Files

| File | Purpose |
|------|---------|
| `knn_model.ipynb` | Main Jupyter notebook with full pipeline |
| `main.py` | Script version to run the model |
| `app.py` | App interface |
| `WineQT.csv` | Wine quality dataset |
| `my_trained_model.pkl` | Saved KNN model |
| `rf_model.pkl` | Saved Random Forest model |
| `scaler.pkl` | Saved StandardScaler |
| `requirements.txt` | All required Python libraries |

---

## ⚙️ How to Run

### 1. Install Requirements
```bash
pip install -r requirements.txt
```

### 2. Run the Notebook
Open `knn_model.ipynb` in Jupyter or VS Code and run all cells.

### 3. Or Run as Script
```bash
python main.py
```

---

## 📦 Libraries Used

```
pandas
numpy
scikit-learn
imbalanced-learn   (SMOTE)
matplotlib
seaborn
joblib
```

---

## 📈 Output You Will See
- Wine Quality Distribution chart
- Feature Correlation Heatmap
- Alcohol vs Quality boxplot
- Outliers Check plot
- KNN Classification Report
- Random Forest Classification Report
- **KNN vs Random Forest Accuracy** bar chart (side by side comparison)

---

## 👤 Author
Ranjit Damase Phone NO.9864583081

---

## 📝 Notes
- SMOTE is applied **only on training data** to avoid data leakage
- Both models are saved as `.pkl` files and can be loaded for future predictions
- Random Forest generally handles imbalanced datasets better than KNN