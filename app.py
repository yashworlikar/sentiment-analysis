import gradio as gr
import torch
from transformers import pipeline
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
from collections import Counter
import emoji
import re
from typing import Dict, Tuple, List
import numpy as np

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

class EnhancedSentimentAnalyzer:
    def __init__(self):
        # Initialize sentiment pipeline
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            top_k=None
        )
        
        # Emoji mappings for sentiment visualization
        self.sentiment_emojis = {
            'POSITIVE': '😊',
            'NEGATIVE': '😔',
            'NEUTRAL': '😐'
        }
        
        # Create emotion color mappings
        self.emotion_colors = {
            'joy': '#FFD700',      # Gold
            'sadness': '#4169E1',  # Royal Blue
            'anger': '#FF4500',    # Red Orange
            'fear': '#800080',     # Purple
            'surprise': '#32CD32', # Lime Green
            'neutral': '#808080'   # Gray
        }

    def analyze_text_style(self, text: str) -> Dict:
        """Analyze writing style and patterns"""
        words = word_tokenize(text)
        
        # Basic text statistics
        avg_word_length = sum(len(word) for word in words) / len(words) if words else 0
        
        # Count punctuation
        punctuation_count = sum(1 for char in text if char in '.,!?;:')
        
        # Count emojis
        emoji_count = sum(char in emoji.EMOJI_DATA for char in text)
        
        return {
            'avg_word_length': round(avg_word_length, 2),
            'punctuation_density': round(punctuation_count / len(text) * 100, 2),
            'emoji_density': round(emoji_count / len(text) * 100, 2)
        }

    def get_emotional_tone(self, sentiment_score: float) -> str:
        """Convert sentiment score to emotional tone"""
        if sentiment_score >= 0.75:
            return 'Very Positive'
        elif sentiment_score >= 0.5:
            return 'Moderately Positive'
        elif sentiment_score > 0.3:
            return 'Slightly Positive'
        elif sentiment_score > -0.3:
            return 'Neutral'
        elif sentiment_score > -0.5:
            return 'Slightly Negative'
        elif sentiment_score > -0.75:
            return 'Moderately Negative'
        else:
            return 'Very Negative'

    def analyze_sentiment(self, text: str) -> Tuple[Dict, str, Dict]:
        # Get base sentiment
        sentiment = self.sentiment_pipeline(text)[0]
        score = sentiment['score']
        label = sentiment['label']
        
        # Get detailed analysis
        blob = TextBlob(text)
        detailed_score = blob.sentiment.polarity
        
        # Get text style analysis
        style_metrics = self.analyze_text_style(text)
        
        # Prepare the visualization data
        emotional_tone = self.get_emotional_tone(detailed_score)
        
        # Create response data
        analysis_results = {
            'primary_sentiment': label,
            'confidence': f"{score:.2%}",
            'emotional_tone': emotional_tone,
            'subjectivity': f"{blob.sentiment.subjectivity:.2%}",
            'style_metrics': style_metrics
        }
        
        # Create visual feedback
        emoji_feedback = self.sentiment_emojis.get(label, '😐')
        
        # Prepare chart data
        chart_data = {
            'labels': ['Negativity', 'Neutrality', 'Positivity'],
            'values': [
                max(0, -detailed_score),
                1 - abs(detailed_score),
                max(0, detailed_score)
            ]
        }
        
        return analysis_results, emoji_feedback, chart_data

def create_interface():
    analyzer = EnhancedSentimentAnalyzer()
    
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🎭 Enhanced Sentiment Explorer
            Discover the emotional depth of your text with advanced analysis
            """
        )
        
        with gr.Row():
            with gr.Column(scale=2):
                input_text = gr.Textbox(
                    label="Enter your text",
                    placeholder="Type something to analyze...",
                    lines=5
                )
                analyze_btn = gr.Button("Analyze Sentiment", variant="primary")
            
            with gr.Column(scale=1):
                emoji_output = gr.Textbox(
                    label="Quick Sentiment",
                    lines=1,
                    show_label=False
                )
        
        with gr.Row():
            with gr.Column():
                sentiment_output = gr.JSON(
                    label="Detailed Analysis",
                )
            
            with gr.Column():
                chart = gr.Plot(label="Sentiment Distribution")
        
        def process_text(text):
            if not text.strip():
                return (
                    {"error": "Please enter some text to analyze"},
                    "😶",
                    None
                )
            
            results, emoji, chart_data = analyzer.analyze_sentiment(text)
            
            # Create visualization
            import matplotlib.pyplot as plt
            
            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(
                chart_data['labels'],
                chart_data['values'],
                color=['#FF6B6B', '#4ECDC4', '#45B7D1']
            )
            
            ax.set_ylim(0, 1)
            ax.set_title('Sentiment Distribution')
            
            # Add value labels on top of bars
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width()/2.,
                    height,
                    f'{height:.2f}',
                    ha='center',
                    va='bottom'
                )
            
            return results, emoji, fig
        
        analyze_btn.click(
            process_text,
            inputs=[input_text],
            outputs=[sentiment_output, emoji_output, chart]
        )
        
        gr.Markdown(
            """
            ### 📊 Understanding the Analysis
            - **Primary Sentiment**: Overall emotional direction
            - **Confidence**: How certain the model is about its prediction
            - **Emotional Tone**: Detailed breakdown of the emotional content
            - **Subjectivity**: How opinionated the text is
            - **Style Metrics**: Writing style analysis
            """
        )
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()