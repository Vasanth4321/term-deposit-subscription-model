# 📊 Term Deposit Subscription Model

A machine learning project to predict whether a customer will subscribe to a **bank term deposit** based on demographic and campaign-related features.  
This project demonstrates **end-to-end data science workflow**: preprocessing, model building, evaluation, and deployment using **Streamlit**.


## 🚀 Live Demo
Try the interactive app here:  
👉 [Bank Term Deposit Predictor](https://term-deposit-subscription-model-gnfxjggcxqnbvyh9okvcgf.streamlit.app/)


## 📂 Project Structure
```
term-deposit-subscription-model/
│
├── datasets/
│   ├── raw/                # Original dataset
│   └── preprocessed/       # Cleaned & transformed dataset
│
├── notebooks/              # Exploratory Data Analysis & model building
├── app/                    # Streamlit application
├── requirements.txt        # Dependencies
├── .python-version         # Python version to run the application
└── README.md               # Project documentation
```


## 🛠️ Tech Stack
- **Python** (pandas, numpy, scikit-learn, matplotlib, seaborn)
- **Streamlit** for deployment
- **EDA & Feature Engineering** for preprocessing
- **Classification Models**: Decision Trees, Random Forest, Gradient Boosting


## 📑 Dataset
- Source: [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/bank+marketing)
- Features include:
  - Demographics: age, job, marital status, education
  - Campaign details: contact type, duration, previous outcomes
  - Target: `y` (whether the client subscribed to a term deposit)


## 📈 Workflow
1. **Data Preprocessing**
   - Handling missing values
   - Encoding categorical variables
   - Scaling numerical features
2. **Exploratory Data Analysis (EDA)**
   - Visualizing distributions and correlations
   - Identifying key predictors
3. **Model Training**
   - Logistic Regression, Random Forest, XGBoost
   - Hyperparameter tuning
4. **Evaluation**
   - Accuracy, Precision, Recall, F1-score, ROC-AUC
5. **Deployment**
   - Streamlit app with user-friendly interface


## 🎯 Key Features
- Upload or input customer data
- Predict likelihood of subscription
- Visualize model performance metrics
- Interactive and recruiter-ready dashboard


## 📦 Installation
Clone the repo and install dependencies:
```bash
git clone https://github.com/Vasanth4321/term-deposit-subscription-model.git
cd term-deposit-subscription-model
pip install -r requirements.txt
```

## 🔮 Future Improvements

### 📌 Advanced Modeling
- Add more advanced models such as **LightGBM** and **CatBoost**

### 🧠 Feature Engineering
- Improve feature selection using **SHAP values** for interpretability

### ☁️ Deployment Enhancements
- Deploy the app on cloud platforms like **Azure** or **AWS**


## 👨‍💻 Author
#### *Vasanth N.V.S*
Passionate about EDA, ML deployment, and recruiter-ready projects.


## ⭐ Contribute
Pull requests are welcome! For major changes, please open an issue first to discuss what you’d like to change.

