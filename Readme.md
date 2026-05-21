# 🍷 Wine Quality Prediction — KNN vs Random Forest

## 📌 What This Project Solves
This project predicts the **quality of wine** based on its chemical properties
(alcohol, pH, acidity, citric acid, etc.) using two machine learning models:
- **KNN (K-Nearest Neighbors)**
- **Random Forest Classifier**
- **Streamlit Web App** for interactive wine quality prediction in the browser

Both models are trained, evaluated, and compared to find which one performs better.
The best model is then deployed via a **Streamlit app** where anyone can test it
by entering wine properties and getting an instant quality prediction.

---

## 📂 Dataset
- **File:** `WineQT.csv`
- **Input Features:** alcohol, pH, citric acid, volatile acidity, residual sugar,
  chlorides, free sulfur dioxide, sulphates, and more
- **Target Column:** `quality` (wine quality score out of 10)

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
| 1️⃣1️⃣ Deploy | Streamlit app lets users predict wine quality interactively |

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

## 🌐 Streamlit Web App

The project includes an interactive web app built with **Streamlit** (`app.py`).

### What the App Does:
- Input wine chemical properties using sliders (citric acid, pH, alcohol, sulphates, etc.)
- Click the **"Predict Quality"** button
- Instantly see the **Predicted Quality score** out of 10
- View **Confidence Scores** — how sure the model is for each quality class

### Example Output:
```
Predicted Quality: 7 / 10

Confidence Scores:
  Quality 5: 38.5%
  Quality 6: 53.8%
```

> The confidence score shows the probability the model assigns to each quality class.
> The class with the highest % becomes the final prediction.

---

## 🗂️ Project Files

| File | Purpose |
|------|---------|
| `knn_model.ipynb` | Main Jupyter notebook with full pipeline |
| `main.py` | Script version to run the model |
| `app.py` | ⭐ Streamlit web app for interactive wine quality prediction |
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

### 4. ⭐ Run the Streamlit App
```bash
streamlit run app.py
```
Then open your browser at: **http://localhost:8501**

You can now enter wine properties and predict quality interactively — no coding needed!

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
streamlit
```

---

## 📈 Output You Will See

### From the Notebook:
- Wine Quality Distribution chart
- Feature Correlation Heatmap
- Alcohol vs Quality boxplot
- Outliers Check plot
- KNN Classification Report
- Random Forest Classification Report
- **KNN vs Random Forest Accuracy** bar chart (side by side comparison)

### From the Streamlit App:
- Interactive sliders for all wine features
- Predicted quality score out of 10
- Confidence scores per quality class

---

## 📝 Notes
- SMOTE is applied **only on training data** to avoid data leakage
- Both models are saved as `.pkl` files and loaded by the Streamlit app
- Random Forest generally handles imbalanced datasets better than KNN
- The Streamlit app makes the model usable by **anyone without coding knowledge**

---

## 👤 Author
Author Name: Ranjit Damase  Contact Info:9864583081