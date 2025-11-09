## 🚀 DoorDash Delivery Time Predictor

An end-to-end machine learning application that predicts food delivery durations using real-world DoorDash order data.
This project covers data cleaning, feature engineering, model training with XGBoost, and deployment as an interactive web app on Hugging Face Spaces.

📌 Project Overview

This project builds a predictive model to estimate how long a DoorDash order will take from pickup to drop-off.

Key highlights:

* Cleaned, preprocessed, and transformed real DoorDash delivery data.
* Engineered high-signal features such as busy_dashers_ratio.
* Reduced multicollinearity using a correlation matrix.
* Trained an optimized XGBoost regression model for accurate ETA prediction.
* Deployed an easy-to-use web app for real-time delivery time estimates.

🔧 Tech Stack

* Python
* XGBoost
* Pandas, NumPy, Scikit-learn
* Matplotlib / Seaborn
* Gradio (for the web UI)
* Hugging Face Spaces (for deployment)

## 🌐 Deployment

The model is deployed as an interactive web application using Gradio and Hugging Face Spaces.

👉 **Live Demo:**  
https://huggingface.co/spaces/Mash37/DoorDash_Delivery_time_predictor

Or run it locally:

```bash
python app.py
```

Author
Mashroor Newaz
Feel free to reach out for questions, suggestions, or collaboration.
