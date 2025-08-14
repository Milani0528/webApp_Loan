from flask import Flask, render_template, request
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

# Load model and label encoders
model = joblib.load("catboost_churn_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")  # Assumes saved during training
feature_list = joblib.load("feature_list.pkl")      # Ensures column order

@app.route("/", methods=["GET", "POST"])
def predict():
    if request.method == "POST":
        # Get input values
        input_data = {
            "Customer_Age": int(request.form["Customer_Age"]),
            "Gender": request.form["Gender"],
            "Dependent_count": int(request.form["Dependent_count"]),
            "Education_Level": request.form["Education_Level"],
            "Marital_Status": request.form["Marital_Status"],
            "Income_Category": request.form["Income_Category"],
            "Card_Category": request.form["Card_Category"],
            "Months_on_book": int(request.form["Months_on_book"]),
            "Total_Relationship_Count": int(request.form["Total_Relationship_Count"]),
            "Months_Inactive_12_mon": int(request.form["Months_Inactive_12_mon"]),
            "Contacts_Count_12_mon": int(request.form["Contacts_Count_12_mon"]),
            "Credit_Limit": float(request.form["Credit_Limit"]),
            "Total_Revolving_Bal": float(request.form["Total_Revolving_Bal"]),
            "Avg_Open_To_Buy": float(request.form["Avg_Open_To_Buy"]),
            "Total_Amt_Chng_Q4_Q1": float(request.form["Total_Amt_Chng_Q4_Q1"]),
            "Total_Trans_Amt": float(request.form["Total_Trans_Amt"]),
            "Total_Trans_Ct": int(request.form["Total_Trans_Ct"]),
            "Total_Ct_Chng_Q4_Q1": float(request.form["Total_Ct_Chng_Q4_Q1"]),
            "Avg_Utilization_Ratio": float(request.form["Avg_Utilization_Ratio"]),
        }

        # Encode categorical variables
        for col, encoder in label_encoders.items():
            if col in input_data:
                input_data[col] = encoder.transform([input_data[col]])[0]

        # Create DataFrame in correct column order
        input_df = pd.DataFrame([input_data], columns=feature_list)

        # Predict
        prediction = model.predict(input_df)[0]
        result = "⚠️ Customer Likely to Churn" if prediction == 1 else "✅ Customer Likely to Stay"

        return render_template("form.html", result=result, user_input=input_data)

    return render_template("form.html", result=None)

if __name__ == "__main__":
    app.run(debug=True)
