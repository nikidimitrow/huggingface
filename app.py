from transformers import pipeline
import gradio as gr

model = pipeline("text-generation", model="gpt2")

def predict(text):
    return model(text, max_length=100)[0]['generated_text']

with gr.Interface(fn=predict, inputs="text", outputs="text") as iface:
    iface.launch()

iface.launch()