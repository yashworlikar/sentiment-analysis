import gradio as gr
from transformers import pipeline
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import numpy as np

# Download required NLTK data
nltk.download('punkt', quiet=True)

class ProductReviewAnalyzer:
    def __init__(self):
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        self.sentiment_emojis = {
            'POSITIVE': '👍',
            'NEGATIVE': '👎',
            'NEUTRAL': '😐'
        }

    def analyze_text_style(self, text: str) -> dict:
        words = word_tokenize(text)
        word_count = len(words)
        avg_word_length = sum(len(word) for word in words) / word_count if word_count else 0
        sentence_count = max(1, text.count('.') + text.count('!') + text.count('?'))
        
        return {
            'word_count': word_count,
            'avg_word_length': round(avg_word_length, 2),
            'words_per_sentence': round(word_count / sentence_count, 2)
        }

    def get_emotional_tone(self, score: float) -> str:
        if score >= 0.75:
            return 'Highly Positive'
        elif score >= 0.5:
            return 'Generally Positive'
        elif score >= 0:
            return 'Somewhat Positive'
        elif score >= -0.5:
            return 'Somewhat Negative'
        else:
            return 'Highly Negative'

    def analyze_sentiment(self, text: str):
        sentiment_result = self.sentiment_pipeline(text)[0]
        label = sentiment_result['label']
        score = sentiment_result['score']
        
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        style_metrics = self.analyze_text_style(text)
        
        analysis_results = {
            'sentiment': label,
            'confidence': f"{score:.2%}",
            'emotional_tone': self.get_emotional_tone(polarity),
            'subjectivity': f"{subjectivity:.2%}",
            'text_metrics': style_metrics
        }
        
        chart_data = {
            'positivity': max(0, polarity),
            'negativity': abs(min(0, polarity)),
            'neutrality': 1 - abs(polarity)
        }
        
        return analysis_results, self.sentiment_emojis[label], chart_data

def create_interface():
    analyzer = ProductReviewAnalyzer()
    
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="green"), css="#analyze-button {margin:auto}") as demo:
        gr.Markdown("# 🛒 Product Review Analyzer\n## Gain insights from customer feedback")
        
        with gr.Row():
            with gr.Column():
                input_text = gr.Textbox(
                    label="Review Text",
                    placeholder="Paste a product review here...",
                    lines=5,
                    interactive=True
                )
                analyze_btn = gr.Button("Analyze Review", variant="primary", elem_id="analyze-button")
            
        with gr.Row():
            with gr.Column():
                emoji_output = gr.Textbox(label="Overall Sentiment", lines=1)
                results_output = gr.JSON(label="Analysis Results")
            with gr.Column():
                plot_output = gr.Plot(label="Sentiment Distribution")
        with gr.Row():
                examples = [
    "Absolutely thrilled with this purchase! The quality is top-notch, and it exceeded my expectations in every way. The design is sleek, and it works perfectly. I highly recommend this to anyone looking for a reliable product.",
    "Unfortunately, this product didn't meet my needs. It stopped working after a week, and I had to go through the hassle of returning it. The customer service was not very helpful either, which added to my frustration.",
    "Decent product, but the shipping was a nightmare. It took forever to get here, and when it finally arrived, the packaging was damaged. The product itself works fine, but the delivery experience was disappointing.",
    "Fantastic quality! Worth every penny. The materials used are durable, and it feels very premium. I've been using it daily, and it hasn't shown any signs of wear and tear. Will definitely buy again.",
    "Not impressed. The item arrived damaged, and when I contacted customer service, they were unhelpful and rude. I expected better for the price I paid. I won't be purchasing from this brand again.",
    "Good value for the price, but the instructions were confusing. It took me a while to figure out how to use it properly. Once I got the hang of it, it worked well, but the initial setup was frustrating."
]
   
        def process_review(text):
            if not text.strip():
                return (
                    {"error": "Please enter a review to analyze"},
                    "🤷",
                    None
                )
            
            # Assuming `analyzer.analyze_sentiment` returns analysis results
            results, emoji, chart_data = analyzer.analyze_sentiment(text)
            
            # Generate sentiment distribution plot
            fig, ax = plt.subplots(figsize=(6, 4))
            categories = ['Positive', 'Negative', 'Neutral']
            values = [
                chart_data['positivity'],
                chart_data['negativity'],
                chart_data['neutrality']
            ]
            
            colors = ['#27ae60', '#c0392b', '#7f8c8d']
            bars = ax.bar(categories, values, color=colors)
            
            ax.set_ylim(0, 1)
            ax.set_title('Sentiment Distribution')
            
            # Display values above bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{height:.2f}',
                    ha='center',
                    va='bottom'
                )
            
            plt.close(fig)
            return results, emoji, fig

        gr.Examples(
                examples=examples,
                inputs=[input_text],
                outputs=[results_output, emoji_output, plot_output],
                fn=process_review,
                run_on_click=  True
    )       

        analyze_btn.click(
            process_review,
            inputs=[input_text],
            outputs=[results_output, emoji_output, plot_output]
        )
        
        gr.Markdown(""" 
        ### 🤔 How to Interpret the Results
        - **Sentiment**: Positive or Negative sentiment of the review.
        - **Confidence**: The model's confidence in the sentiment classification.
        - **Emotional Tone**: Deeper understanding of the mood in the review.
        - **Subjectivity**: Whether the review is opinion-based or factual.
        - **Text Metrics**: Breakdown of review structure and readability.
        
        [Check out our other AI demos](https://systenics.ai/demo)
        """)
    
    return demo
if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
