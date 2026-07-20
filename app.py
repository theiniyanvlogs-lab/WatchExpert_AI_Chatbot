import gradio as gr
from chatbot import chatbot

demo = gr.ChatInterface(
    fn=chatbot,
    title="⌚ WatchExpert AI Chatbot",
    description="Ask questions about watches, warranty, returns, offers and shop details.",
    examples=[
        "Do you sell Rolex?",
        "What is the warranty?",
        "Do you provide EMI?",
        "Can I exchange my old watch?",
        "What are your shop timings?"
    ]
)

if __name__ == "__main__":
    demo.launch(
        share=True,
        debug=True
    )
