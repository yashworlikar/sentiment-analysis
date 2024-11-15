import gradio as gr
from transformers import pipeline
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import numpy as np

# Download required NLTK data
nltk.download('punkt_tab', quiet=True)

class EnhancedSentimentAnalyzer:
    def __init__(self):
        # Initialize sentiment pipeline without top_k parameter
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
        
        # Emoji mappings
        self.sentiment_emojis = {
            'POSITIVE': '😊',
            'NEGATIVE': '😔',
            'NEUTRAL': '😐'
        }

    def analyze_text_style(self, text: str) -> dict:
        """Analyze writing style and patterns"""
        words = word_tokenize(text)
        
        # Calculate metrics
        word_count = len(words)
        avg_word_length = sum(len(word) for word in words) / word_count if word_count else 0
        sentence_count = text.count('.') + text.count('!') + text.count('?')
        sentence_count = max(1, sentence_count)  # Ensure no division by zero
        
        return {
            'word_count': word_count,
            'avg_word_length': round(avg_word_length, 2),
            'words_per_sentence': round(word_count / sentence_count, 2)
        }

    def get_emotional_tone(self, score: float) -> str:
        """Convert sentiment score to emotional tone"""
        if score >= 0.75:
            return 'Very Positive'
        elif score >= 0.5:
            return 'Moderately Positive'
        elif score >= 0:
            return 'Slightly Positive'
        elif score >= -0.5:
            return 'Slightly Negative'
        else:
            return 'Very Negative'

    def analyze_sentiment(self, text: str):
        # Get base sentiment - now correctly handling the pipeline output
        sentiment_result = self.sentiment_pipeline(text)[0]  # Get first result
        label = sentiment_result['label']
        score = sentiment_result['score']
        
        # Get additional analysis
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Get text style analysis
        style_metrics = self.analyze_text_style(text)
        
        # Prepare results
        analysis_results = {
            'sentiment': label,
            'confidence': f"{score:.2%}",
            'emotional_tone': self.get_emotional_tone(polarity),
            'subjectivity': f"{subjectivity:.2%}",
            'text_metrics': style_metrics
        }
        
        # Prepare visualization data
        chart_data = {
            'positivity': max(0, polarity),
            'negativity': abs(min(0, polarity)),
            'neutrality': 1 - abs(polarity)
        }
        
        return analysis_results, self.sentiment_emojis[label], chart_data

def create_interface():
    analyzer = EnhancedSentimentAnalyzer()
    
    with gr.Blocks(theme=gr.themes.Soft(primary_hue="blue")) as demo:
        gr.Markdown("""
        # 🎭 Sentiment Analysis Plus
        ## Discover the emotional depth of your text
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                input_text = gr.Textbox(
                    label="Input Text",
                    placeholder="Type or paste your text here...",
                    lines=5
                )
                analyze_btn = gr.Button("Analyze", variant="primary")
            
            with gr.Column(scale=1):
                emoji_output = gr.Textbox(
                    label="Mood",
                    lines=1,
                    show_label=True
                )
        
        with gr.Row():
            with gr.Column():
                results_output = gr.JSON(
                    label="Analysis Results"
                )
            with gr.Column():
                plot_output = gr.Plot(label="Sentiment Distribution")

        def process_text(text):
            if not text.strip():
                return (
                    {"error": "Please enter some text to analyze"},
                    "😶",
                    None
                )
            
            results, emoji, chart_data = analyzer.analyze_sentiment(text)
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(6, 4))
            categories = ['Positive', 'Negative', 'Neutral']
            values = [
                chart_data['positivity'],
                chart_data['negativity'],
                chart_data['neutrality']
            ]
            
            colors = ['#2ecc71', '#e74c3c', '#95a5a6']
            bars = ax.bar(categories, values, color=colors)
            
            ax.set_ylim(0, 1)
            ax.set_title('Sentiment Distribution')
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{height:.2f}',
                    ha='center',
                    va='bottom'
                )
            
            plt.close(fig)  # Prevent display in notebook if running there
            return results, emoji, fig
        
        analyze_btn.click(
            process_text,
            inputs=[input_text],
            outputs=[results_output, emoji_output, plot_output]
        )
        
        gr.Markdown("""
        ### 📊 Understanding Your Results
        - **Sentiment**: Overall emotional direction of the text
        - **Confidence**: How certain the model is about its analysis
        - **Emotional Tone**: Detailed breakdown of the emotional content
        - **Subjectivity**: How opinionated vs. factual the text is
        - **Text Metrics**: Statistical analysis of your writing style
        """)
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()