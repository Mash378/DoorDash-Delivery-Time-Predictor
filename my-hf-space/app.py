import gradio as gr
import joblib
import numpy as np
import pandas as pd

model = joblib.load("model.joblib")

def predict(categoryindian, market_id_1, order_protocol_3,total_busy_dashers,
            total_onshift_dashers, total_outstanding_orders,avg_price_per_item, estimated_drive_duration):
    
    
    busy_dashers_ratio = total_busy_dashers/total_onshift_dashers

    X_input = pd.DataFrame([[
        categoryindian,
        market_id_1,
        order_protocol_3,
        busy_dashers_ratio,
        total_outstanding_orders,
        avg_price_per_item,
        estimated_drive_duration
    ]], columns=[
        "categoryindian",
        "market_id_1.0",
        "order_protocol_3.0",
        "busy_dashers_ratio",
        "total_outstanding_orders",
        "avg_price_per_item",
        "estimated_store_to_consumer_driving_duration"
    ])
    

    prediction = model.predict(X_input)
    return str(prediction[0])

demo = gr.Interface(
    fn= predict,
    inputs=[
        gr.Radio([0,1],label="Is Indian category"),
        gr.Radio([0,1],label="Is market id 1.0"),
        gr.Radio([0,1],label="Is order protocol 3.0"),
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