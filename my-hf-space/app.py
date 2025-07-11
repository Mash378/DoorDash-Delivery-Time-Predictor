import gradio as gr
import joblib
import numpy as np
import pandas as pd

DoorDash_data = pd.read_csv("../datasets/historical_data.csv")
market_ids = DoorDash_data["market_id"].unique().tolist()
order_protocols = DoorDash_data["order_protocol"].unique().tolist()
store_categories = DoorDash_data["store_primary_category"].unique().tolist()

#Load the model
model = joblib.load("model.joblib")
model_columns = joblib.load("model_columns.pkl")

def predict(store_category, market_id, order_protocol,total_busy_dashers,
            total_onshift_dashers, total_outstanding_orders,avg_price_per_item, estimated_drive_duration):
    
    busy_dashers_ratio = total_busy_dashers/total_onshift_dashers
    store_category_col = "category" + store_category
    market_id_col = f"market_id_{market_id}.0"
    order_protocol_col = f"order_protocol_{order_protocol}.0"

    X_input = pd.DataFrame([[0]*len(model_columns)], columns=model_columns)

    # Fill in selected fields
    if store_category_col in X_input.columns:
        X_input.at[0, store_category_col] = 1
    if market_id_col in X_input.columns:
        X_input.at[0, market_id_col] = 1
    if order_protocol_col in X_input.columns:
        X_input.at[0, order_protocol_col] = 1

    X_input.at[0, "busy_dashers_ratio"] = busy_dashers_ratio
    X_input.at[0, "total_outstanding_orders"] = total_outstanding_orders
    X_input.at[0, "avg_price_per_item"] = avg_price_per_item
    X_input.at[0, "estimated_store_to_consumer_driving_duration"] = estimated_drive_duration


    prediction = model.predict(X_input)
    return str(prediction[0])

demo = gr.Interface(
    fn= predict,
    inputs=[
        gr.Dropdown(store_categories, label="Store Category"),
        gr.Dropdown(market_ids, label="Market ID"),
        gr.Dropdown(order_protocols, label="Order Protocol"),
        gr.Number(label="Number of busy dashers"),
        gr.Number(label="Number of total onshift dashers"),
        gr.Number(label="Total Outstanding Orders"),
        gr.Number(label="Average Price per Item ($)"),
        gr.Number(label="Estimated Drive Duration (sec)")
    ],
    outputs=gr.Textbox(label="Predicted Delivery Time"),
    title="DoorDash Delivery Time Estimator",
    description="Estimate delivery time using top features from the model."
)

demo.launch(share=True)