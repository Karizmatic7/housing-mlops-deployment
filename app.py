import gradio as gr
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model.pkl")

# Define prediction function
def predict_price(area, bedrooms, bathrooms):
    input_df = pd.DataFrame({
        "area": [area],
        "bedrooms": [bedrooms],
        "bathrooms": [bathrooms]
    })
    prediction = model.predict(input_df)
    return f"Predicted price: ${prediction[0]:,.2f}"

# Create Gradio interface
iface = gr.Interface(
    fn=predict_price,
    inputs=[
        gr.Number(label="Area (sq ft)"),
        gr.Number(label="Bedrooms"),
        gr.Number(label="Bathrooms")
    ],
    outputs="text",
    title="Housing Price Predictor",
    description="Enter the area, number of bedrooms, and bathrooms to predict housing price."
)

# Launch the app
iface.launch()
