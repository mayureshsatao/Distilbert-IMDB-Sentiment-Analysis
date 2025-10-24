import streamlit as st
import torch
from transformers import BertForSequenceClassification, BertTokenizer
import time
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="AI Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for beautiful styling
st.markdown("""
    <style>
    /* Main background with animated gradient */
    .stApp {
        background: linear-gradient(-45deg, #667eea, #764ba2, #f093fb, #4facfe);
        background-size: 400% 400%;
        animation: gradientShift 15s ease infinite;
    }

    @keyframes gradientShift {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }

    /* Glass morphism container */
    .main-container {
        background: transparent;
        padding: 2rem 3rem;
        margin: 0 auto;
        max-width: 1000px;
    }

    /* Remove the content-box class as we're not using it anymore */

    /* Title with glow effect */
    .title {
        font-size: 4rem;
        font-weight: 900;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
        font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        text-shadow: 0 0 40px rgba(102, 126, 234, 0.5);
        letter-spacing: -2px;
    }

    .subtitle {
        text-align: center;
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.3rem;
        margin-bottom: 2.5rem;
        font-weight: 400;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }

    /* Enhanced text area styling */
    .stTextArea textarea {
        border-radius: 15px;
        border: 2px solid rgba(102, 126, 234, 0.3);
        padding: 1.2rem;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        background: white !important;
        color: #333 !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }

    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.15);
        transform: translateY(-2px);
        background: white !important;
        color: #333 !important;
    }

    .stTextArea textarea::placeholder {
        color: #999 !important;
        opacity: 1;
    }

    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 50px;
        padding: 1rem 3rem;
        font-size: 1.2rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.4);
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 30px rgba(102, 126, 234, 0.6);
    }

    /* Result cards with enhanced design */
    .result-card {
        padding: 2.5rem;
        border-radius: 20px;
        margin-top: 2rem;
        text-align: center;
        animation: slideUp 0.6s cubic-bezier(0.68, -0.55, 0.265, 1.55);
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.2);
        position: relative;
        overflow: hidden;
    }

    .result-card::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: radial-gradient(circle, rgba(255, 255, 255, 0.1) 0%, transparent 70%);
        animation: shimmer 3s infinite;
    }

    @keyframes shimmer {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }

    .positive-card {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }

    .negative-card {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        color: white;
    }

    .sentiment-text {
        font-size: 3rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        position: relative;
        z-index: 1;
        text-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }

    .sentiment-emoji {
        font-size: 5rem;
        margin-bottom: 1rem;
        position: relative;
        z-index: 1;
        animation: bounce 1s ease;
    }

    @keyframes bounce {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-20px); }
    }

    .confidence-bar {
        width: 100%;
        height: 12px;
        background: rgba(255, 255, 255, 0.3);
        border-radius: 10px;
        margin-top: 1.5rem;
        overflow: hidden;
        position: relative;
        z-index: 1;
    }

    .confidence-fill {
        height: 100%;
        background: rgba(255, 255, 255, 0.8);
        border-radius: 10px;
        animation: fillBar 1s ease-out;
    }

    @keyframes fillBar {
        from { width: 0; }
    }

    /* Animation */
    @keyframes slideUp {
        from {
            opacity: 0;
            transform: translateY(40px) scale(0.95);
        }
        to {
            opacity: 1;
            transform: translateY(0) scale(1);
        }
    }

    /* Info box with gradient border */
    .info-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 20px;
        margin-top: 2.5rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        position: relative;
    }

    .info-box::before {
        content: '✨';
        font-size: 2rem;
        position: absolute;
        top: -15px;
        left: 50%;
        transform: translateX(-50%);
        background: white;
        width: 50px;
        height: 50px;
        border-radius: 50%;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }

    /* Stats cards */
    .stat-card {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.1) 0%, rgba(118, 75, 162, 0.1) 100%);
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        border: 2px solid rgba(102, 126, 234, 0.2);
        transition: transform 0.3s ease;
    }

    .stat-card:hover {
        transform: translateY(-5px);
    }

    .stat-number {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .stat-label {
        color: #666;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    /* Hide Streamlit branding and default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stDeployButton {display: none;}

    /* Hide default Streamlit header */
    .block-container {
        padding-top: 1rem;
    }

    /* Remove top padding */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Hide any empty divs at top */
    div[data-testid="stToolbar"] {
        display: none;
    }

    div[data-testid="stDecoration"] {
        display: none;
    }

    div[data-testid="stStatusWidget"] {
        display: none;
    }

    /* Loading animation */
    .loading {
        text-align: center;
        color: white;
        font-size: 1.3rem;
        margin-top: 1rem;
        font-weight: 600;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
    }

    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: #333;
        margin-bottom: 1.5rem;
        text-align: center;
        position: relative;
    }

    .section-header::after {
        content: '';
        display: block;
        width: 60px;
        height: 4px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        margin: 0.5rem auto 0;
        border-radius: 2px;
    }

    /* Expander styling */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
        border-radius: 10px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)


# Load model with caching
@st.cache_resource
def load_model():
    try:
        model = BertForSequenceClassification.from_pretrained('./fine_tuned_bert_imdb_model')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        return model, tokenizer
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None


# Function to predict sentiment
def predict_sentiment(text, model, tokenizer):
    inputs = tokenizer.encode_plus(
        text,
        add_special_tokens=True,
        return_tensors='pt',
        max_length=512,
        truncation=True,
        padding=True
    )

    with torch.no_grad():
        outputs = model(
            inputs['input_ids'],
            token_type_ids=None,
            attention_mask=inputs['attention_mask']
        )

    logits = outputs[0]
    predicted_class = torch.argmax(logits).item()

    # Get probabilities
    probs = torch.nn.functional.softmax(logits, dim=1)[0]
    confidence = probs[predicted_class].item()

    labels = ['Negative', 'Positive']
    predicted_sentiment = labels[predicted_class]

    return predicted_sentiment, confidence, probs.cpu().numpy()


# Create confidence gauge chart
def create_confidence_gauge(confidence, sentiment):
    color = "#38ef7d" if sentiment == "Positive" else "#ff6a00"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Level", 'font': {'size': 24, 'color': '#333'}},
        number={'suffix': "%", 'font': {'size': 48, 'color': color}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': color},
            'bar': {'color': color},
            'bgcolor': "rgba(0,0,0,0.1)",
            'borderwidth': 2,
            'bordercolor': "white",
            'steps': [
                {'range': [0, 50], 'color': 'rgba(200, 200, 200, 0.3)'},
                {'range': [50, 75], 'color': 'rgba(150, 150, 150, 0.3)'},
                {'range': [75, 100], 'color': 'rgba(100, 100, 100, 0.3)'}
            ],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': confidence * 100
            }
        }
    ))

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={'color': "#333", 'family': "Arial"},
        height=300,
        margin=dict(l=20, r=20, t=50, b=20)
    )

    return fig


# Main app
def main():
    # Load model
    model, tokenizer = load_model()

    if model is None or tokenizer is None:
        st.error("⚠️ Failed to load the model. Please check the model path.")
        return

    # Header with animated gradient background
    st.markdown('<h1 class="title">🎬 AI Sentiment Analyzer</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">✨ Powered by BERT | Analyze movie review sentiments with cutting-edge AI ✨</p>',
                unsafe_allow_html=True)

    # Model info stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
            <div class="stat-card">
                <div class="stat-number">BERT</div>
                <div class="stat-label">Model Type</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="stat-card">
                <div class="stat-number">IMDb</div>
                <div class="stat-label">Training Data</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="stat-card">
                <div class="stat-number">2</div>
                <div class="stat-label">Classes</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Input section
    st.markdown('<div class="section-header">📝 Enter Your Movie Review</div>', unsafe_allow_html=True)
    user_input = st.text_area(
        "",
        placeholder="Type or paste a movie review here...\n\nFor example: 'This movie was absolutely fantastic! The acting was superb and the cinematography was breathtaking. I was on the edge of my seat the entire time.'",
        height=180,
        label_visibility="collapsed"
    )

    # Analyze button
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("🔍 Analyze Sentiment", use_container_width=True, type="primary")

    # Prediction
    if analyze_button and user_input:
        st.markdown('<p class="loading">🤖 Analyzing sentiment with AI...</p>', unsafe_allow_html=True)
        time.sleep(0.3)

        prediction, confidence, probs = predict_sentiment(user_input, model, tokenizer)

        # Display result with enhanced visuals
        if prediction == 'Positive':
            st.markdown(f"""
                <div class="result-card positive-card">
                    <div class="sentiment-emoji">😊 🎉</div>
                    <div class="sentiment-text">{prediction} Sentiment</div>
                    <p style="font-size: 1.2rem; margin-top: 1rem; opacity: 0.95;">
                        This review expresses positive feelings!
                    </p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence * 100}%;"></div>
                    </div>
                    <p style="font-size: 1.1rem; margin-top: 1rem; opacity: 0.9;">
                        Confidence: {confidence * 100:.1f}%
                    </p>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="result-card negative-card">
                    <div class="sentiment-emoji">😞 💔</div>
                    <div class="sentiment-text">{prediction} Sentiment</div>
                    <p style="font-size: 1.2rem; margin-top: 1rem; opacity: 0.95;">
                        This review expresses negative feelings.
                    </p>
                    <div class="confidence-bar">
                        <div class="confidence-fill" style="width: {confidence * 100}%;"></div>
                    </div>
                    <p style="font-size: 1.1rem; margin-top: 1rem; opacity: 0.9;">
                        Confidence: {confidence * 100:.1f}%
                    </p>
                </div>
            """, unsafe_allow_html=True)

        # Show confidence gauge
        st.markdown("<br>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            fig = create_confidence_gauge(confidence, prediction)
            st.plotly_chart(fig, use_container_width=True)

        # Show probability breakdown
        st.markdown('<div class="section-header">📊 Probability Breakdown</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)

        with col1:
            negative_prob = probs[0] * 100
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%); 
                            padding: 1.5rem; border-radius: 15px; color: white; text-align: center;">
                    <div style="font-size: 2rem; font-weight: 700;">{negative_prob:.1f}%</div>
                    <div style="font-size: 1rem; margin-top: 0.5rem; opacity: 0.9;">Negative</div>
                </div>
            """, unsafe_allow_html=True)

        with col2:
            positive_prob = probs[1] * 100
            st.markdown(f"""
                <div style="background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); 
                            padding: 1.5rem; border-radius: 15px; color: white; text-align: center;">
                    <div style="font-size: 2rem; font-weight: 700;">{positive_prob:.1f}%</div>
                    <div style="font-size: 1rem; margin-top: 0.5rem; opacity: 0.9;">Positive</div>
                </div>
            """, unsafe_allow_html=True)

    elif analyze_button and not user_input:
        st.warning("⚠️ Please enter a movie review to analyze.")

    # Info section
    st.markdown("""
        <div class="info-box">
            <strong style="font-size: 1.3rem;">💡 How it works</strong><br><br>
            This app uses a fine-tuned BERT (Bidirectional Encoder Representations from Transformers) 
            model trained on IMDb movie reviews to classify sentiments as positive or negative. 
            The model analyzes the semantic meaning and context of your text to make accurate predictions.
        </div>
    """, unsafe_allow_html=True)

    # Examples section
    with st.expander("📚 See example reviews and tips"):
        st.markdown("""
        ### ✅ Positive Example
        *"This movie was absolutely fantastic! The acting was superb and the plot kept me engaged throughout. 
        The cinematography was breathtaking, and the soundtrack perfectly complemented every scene. 
        I would highly recommend this to anyone who loves great storytelling."*

        ### ❌ Negative Example
        *"I was really disappointed with this film. The pacing was terrible and the story made no sense. 
        The characters were poorly developed, and the dialogue felt forced. I wouldn't recommend wasting 
        your time on this one."*

        ### 💭 Tips for Best Results
        - Write detailed reviews (longer text gives better accuracy)
        - Use natural language and complete sentences
        - Include specific opinions about aspects like acting, plot, cinematography, etc.
        - The model works best with English movie reviews
        """)


if __name__ == '__main__':
    main()