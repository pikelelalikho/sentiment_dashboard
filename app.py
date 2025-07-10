import gradio as gr
from sentiment_model import vader_sentiment, hf_sentiment

def analyze_sentiment(text):
    vader_result = vader_sentiment(text)
    hf_result = hf_sentiment(text)
    return vader_result, hf_result

with gr.Blocks(title="Sentiment Analysis Dashboard") as demo:
    gr.Markdown("""
    # ♻️ Sentiment Analysis Dashboard
    Compare **VaderSentiment (Rule-Based)** with **Hugging Face Transformers (Model-Based)** on your text.

    Enter your text below to see live sentiment predictions for your bootcamp project.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            input_text = gr.Textbox(
                placeholder="Type your sentence here...",
                label="Input Text",
                lines=3
            )
            submit_btn = gr.Button("Analyze Sentiment", variant="primary")
        
        with gr.Column(scale=1):
            vader_output = gr.Textbox(label="🟢 Vader Sentiment Output")
            hf_output = gr.Textbox(label="🔵 Hugging Face Sentiment Output")

    submit_btn.click(analyze_sentiment, inputs=input_text, outputs=[vader_output, hf_output])

    gr.Markdown("""
    ---
    ### 📊 About
    - Built with Gradio, Transformers, and VaderSentiment 🎭
    """)

if __name__ == "__main__":
    demo.launch()
