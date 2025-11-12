---

# car_price_guess_model

---

# 🚗 Car Price Prediction Project with Artificial Intelligence

> This project was developed as part of an Artificial Intelligence course. It predicts the **estimated market price of a car** based on the features entered by the user using an AI model.

---

## 🎯 Project Goal

The main goal of the project is to predict **car prices** using a machine learning model based on car details provided by the user, such as **brand, model, age, mileage, engine power**, etc.

---

## 👥 Team Members

* **Muhammet Burak Akkaş** – Team Leader, AI Model Development & GUI
* **Doğanay Yıldız** – Graph Creation
* **Gürkan Özdemir** – Error Analysis
* **Berkay Berber** – Data Analysis

---

## 📈 Project Development Process

### 1. Starting Point

The project started with the idea of creating a **simple linear regression model**. The goal was to predict car prices based on basic information provided by the user. During the initial data analysis, we noticed that the dataset contained missing and inconsistent information.

### 2. Challenges and Solutions

During the process, we encountered some challenges and solved them as a team:

* **Low Model Performance:** In the first attempts, our R² score was quite low. Thanks to Berkay’s detailed data analysis, we decided to **include categorical data in the model using One-Hot Encoding**.
* **Error Analysis:** Gürkan identified where the model made the most errors, helping us understand the weak points of the model.
* **Visualization:** Doğanay visualized the model results and errors **with graphs**, making performance evaluation more understandable.
* **Model Improvement:** Muhammet Burak updated the model with **a more powerful algorithm like RandomForest** based on the collected data and analysis, and created a user-friendly GUI.

### 3. What We Learned

Through this project, we gained experience in:

* The critical importance of **data cleaning and analysis** for model success
* Evaluating model performance with the **right metrics** (RMSE, R²)
* Making results understandable through **visualization**
* Combining all components to create a working application with a robust AI model

---

## 💻 Technologies Used

The project was developed using the Python ecosystem and the following libraries:

* **contourpy** – 1.3.3
* **cycler** – 0.12.1
* **fonttools** – 4.60.1
* **joblib** – 1.5.2
* **kiwisolver** – 1.4.9
* **matplotlib** – 3.10.7
* **numpy** – 2.3.4
* **packaging** – 25.0
* **pandas** – 2.3.3
* **pillow** – 12.0.0
* **pip** – 24.0
* **pyparsing** – 3.2.5
* **python-dateutil** – 2.9.0.post0
* **pytz** – 2025.2
* **scikit-learn** – 1.7.2
* **scipy** – 1.16.3
* **six** – 1.17.0
* **threadpoolctl** – 3.6.0
* **tzdata** – 2025.2

---

## 🚀 Running the Project

To run the project on your computer, follow these steps:

1. Clone the repository and navigate to the project folder:

```bash
git clone [REPO_URL]
cd [PROJECT_FOLDER_NAME]
```

2. Create and activate a virtual environment:

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate
```

3. Install the required libraries:

```bash
pip install -r requirements.txt
```

4. Run the application:

```bash
python car_price_guess_model[MODEL_NUMBER].py
```

---

If you want, I can also **make the Markdown fully polished and natural in English**, so it reads like a professional GitHub README.

Do you want me to do that?
