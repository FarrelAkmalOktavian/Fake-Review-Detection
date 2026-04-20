import streamlit as st
import numpy as np
import pickle
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import warnings
warnings.filterwarnings("ignore")

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fake Review Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
}

.main-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    font-weight: 400;
    letter-spacing: -0.5px;
    line-height: 1.2;
    margin-bottom: 0.25rem;
}

.sub-title {
    font-size: 1rem;
    color: #6b7280;
    font-weight: 300;
    margin-bottom: 2rem;
}

.metric-card {
    background: #f8f9fb;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}

.metric-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #9ca3af;
    margin-bottom: 0.3rem;
}

.metric-value {
    font-size: 1.8rem;
    font-weight: 500;
    color: #111827;
}

.result-box-fake {
    background: #fff1f2;
    border-left: 4px solid #e11d48;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
}

.result-box-real {
    background: #f0fdf4;
    border-left: 4px solid #16a34a;
    border-radius: 8px;
    padding: 1.2rem 1.5rem;
    margin: 1rem 0;
}

.result-label {
    font-family: 'DM Serif Display', serif;
    font-size: 1.4rem;
    margin-bottom: 0.3rem;
}

.result-desc {
    font-size: 0.9rem;
    color: #4b5563;
}

.confidence-bar-wrap {
    background: #e5e7eb;
    border-radius: 99px;
    height: 8px;
    width: 100%;
    margin: 0.6rem 0;
}

.token-pill {
    display: inline-block;
    padding: 4px 10px;
    border-radius: 20px;
    margin: 3px 3px;
    font-size: 0.82rem;
    font-weight: 500;
    color: white;
}

.sidebar-section {
    background: #f8f9fb;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
    border: 1px solid #e5e7eb;
}

.step-indicator {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 0.5rem;
}
            
.stTextArea textarea {
    border-radius: 10px !important;
    border: 1.5px solid #d1d5db !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.95rem !important;
}
</style>
""", unsafe_allow_html=True)

# ── NLTK Downloads ─────────────────────────────────────────────────────────────
@st.cache_resource
def download_nltk():
    for pkg in ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']:
        try:
            nltk.download(pkg, quiet=True)
        except:
            pass

download_nltk()

# ── Attention Layer (must match training) ──────────────────────────────────────
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(name="att_weight", shape=(input_shape[-1], 1), initializer="normal")
        self.b = self.add_weight(name="att_bias", shape=(input_shape[1], 1), initializer="zeros")
        super(AttentionLayer, self).build(input_shape)

    def call(self, x):
        et = tf.keras.backend.squeeze(tf.keras.backend.tanh(tf.keras.backend.dot(x, self.W) + self.b), axis=-1)
        at = tf.keras.backend.softmax(et)
        at = tf.keras.backend.expand_dims(at, axis=-1)
        output = x * at
        return tf.keras.backend.sum(output, axis=1)

# ── Load Resources ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    import os
    # Cek file ada atau tidak
    #st.write("📁 Working directory:", os.getcwd())
    #st.write("📄 Files found:", os.listdir('.'))
    
    model = tf.keras.models.load_model(
        'lstm_attention_s4.keras',
        custom_objects={'AttentionLayer': AttentionLayer}
    )
    #st.write("✅ Model loaded")
    
    with open('tokenizer.pkl', 'rb') as f:
        tokenizer = pickle.load(f)
    # st.write("✅ Tokenizer loaded")
    
    with open('label_encoder.pkl', 'rb') as f:
        le = pickle.load(f)
    # st.write("✅ Label encoder loaded")
    
    return model, tokenizer, le

# ── Attention Model (for extracting weights) ───────────────────────────────────
@st.cache_resource
def build_attention_extractor(_model):
    """Build a sub-model that outputs LSTM sequences before attention pooling."""
    lstm_output = None
    for layer in _model.layers:
        if isinstance(layer, tf.keras.layers.LSTM):
            lstm_output = layer.output
            break
    if lstm_output is None:
        return None
    attention_extractor = tf.keras.Model(inputs=_model.input, outputs=lstm_output)
    return attention_extractor

# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess_text(text):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    text = str(text).lower()
    text = re.sub(r'[^\x00-\x7f]', r'', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    tokens = word_tokenize(text)
    cleaned_tokens = [lemmatizer.lemmatize(w) for w in tokens]
    return " ".join(cleaned_tokens), cleaned_tokens

# ── Attention Weights (manual computation matching AttentionLayer) ─────────────
def get_attention_weights(attention_extractor, sequence, model):
    lstm_out = attention_extractor.predict(sequence, verbose=0)  # (1, maxlen, 128)
    att_layer = None
    for layer in model.layers:
        if isinstance(layer, AttentionLayer):
            att_layer = layer
            break
    if att_layer is None:
        return None
    W = att_layer.get_weights()[0]  # (128, 1)
    b = att_layer.get_weights()[1]  # (maxlen, 1)
    x = lstm_out[0]                  # (maxlen, 128)
    et = np.tanh(x @ W + b)         # (maxlen, 1)
    et = et.squeeze(-1)              # (maxlen,)
    at = np.exp(et) / np.sum(np.exp(et))  # softmax
    return at                        # (maxlen,)

# ── Predict ────────────────────────────────────────────────────────────────────
def predict(text, model, tokenizer, le, attention_extractor, max_len=150):
    clean, tokens = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([clean])
    padded = pad_sequences(seq, maxlen=max_len)
    prob = float(model.predict(padded, verbose=0)[0][0])

    classes = le.classes_  # e.g. ['CG', 'OR']
    # label=1 → le.classes_[1]; label=0 → le.classes_[0]
    label_idx = int(prob > 0.5)
    label = classes[label_idx]
    confidence = prob if label_idx == 1 else 1 - prob

    att_weights = get_attention_weights(attention_extractor, padded, model)

    # Map weights back to original tokens (after padding, align to end)
    token_weights = {}
    if att_weights is not None and tokens:
        # The sequence is right-padded; valid tokens start at index (max_len - len(seq[0]))
        offset = max_len - len(seq[0])
        for i, tok in enumerate(tokens):
            if offset + i < max_len:
                token_weights[tok] = float(att_weights[offset + i])

    return label, confidence, prob, token_weights, clean

# ── Attention Heatmap ──────────────────────────────────────────────────────────
def render_attention_heatmap(token_weights, is_fake):
    if not token_weights:
        st.info("No attention weights available for this input.")
        return

    sorted_tokens = sorted(token_weights.items(), key=lambda x: x[1], reverse=True)[:20]
    if not sorted_tokens:
        return

    tokens_list = [t[0] for t in sorted_tokens]
    weights_list = [t[1] for t in sorted_tokens]

    base_color = "#e11d48" if is_fake else "#16a34a"
    fig, ax = plt.subplots(figsize=(8, max(3, len(tokens_list) * 0.35)))
    fig.patch.set_alpha(0)
    ax.set_facecolor('none')

    norm_weights = np.array(weights_list)
    norm_weights = (norm_weights - norm_weights.min()) / (norm_weights.max() - norm_weights.min() + 1e-8)

    bars = ax.barh(tokens_list[::-1], norm_weights[::-1], color=base_color, alpha=0.85)
    for bar, w in zip(bars, norm_weights[::-1]):
        bar.set_alpha(0.3 + 0.7 * w)

    ax.set_xlabel("Normalized Attention Weight", fontsize=10, color="#6b7280")
    ax.tick_params(colors='#374151', labelsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#e5e7eb')
    ax.spines['bottom'].set_color('#e5e7eb')
    ax.xaxis.label.set_color('#6b7280')
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# ── Token Highlight HTML ───────────────────────────────────────────────────────
def render_token_pills(token_weights, is_fake):
    if not token_weights:
        return
    base_rgb = (225, 29, 72) if is_fake else (22, 163, 74)
    max_w = max(token_weights.values()) if token_weights else 1
    pills_html = ""
    for tok, w in sorted(token_weights.items(), key=lambda x: x[1], reverse=True)[:30]:
        alpha = int(80 + 175 * (w / max_w))
        color = f"rgba({base_rgb[0]},{base_rgb[1]},{base_rgb[2]},{alpha/255:.2f})"
        pills_html += f'<span class="token-pill" style="background:{color};">{tok}</span>'
    st.markdown(pills_html, unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## About")
    st.markdown("""
<div class="sidebar-section">
    <p style="font-size:0.88rem; color:#4b5563; margin:0;">
    This tool uses an <strong>LSTM + Attention</strong> model trained to classify 
    e-commerce reviews as <em>genuine (OR)</em> or <em>computer-generated fake (CG)</em>.
    </p>
</div>
""", unsafe_allow_html=True)

    st.markdown("### Model Performance")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
<div class="metric-card">
    <div class="metric-label">Accuracy</div>
    <div class="metric-value" style="font-size:1.3rem;">93.6%</div>
</div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""
<div class="metric-card">
    <div class="metric-label">Precision</div>
    <div class="metric-value" style="font-size:1.3rem;">93.1%</div>
</div>""", unsafe_allow_html=True)
    col3, col4 = st.columns(2)
    with col3:
        st.markdown("""
<div class="metric-card" style="margin-top:8px;">
    <div class="metric-label">Recall</div>
    <div class="metric-value" style="font-size:1.3rem;">94.1%</div>
</div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""
<div class="metric-card" style="margin-top:8px;">
    <div class="metric-label">F1-Score</div>
    <div class="metric-value" style="font-size:1.3rem;">93.6%</div>
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Architecture")
    st.markdown("""
<div class="sidebar-section">
<p style="font-size:0.82rem; color:#4b5563; margin:0; line-height:1.8;">
• Embedding (10,000 vocab, 300-dim)<br>
• SpatialDropout1D (0.3)<br>
• LSTM (64 units)<br>
• <strong>Single Attention Layer</strong><br>
• Dense (64, ReLU) + Dropout<br>
• Dense (1, Sigmoid)
</p>
</div>
""", unsafe_allow_html=True)

    st.markdown("### Labels")
    st.markdown("""
<div class="sidebar-section">
<p style="font-size:0.82rem; color:#4b5563; margin:0; line-height:1.8;">
🟢 <strong>OR</strong> — Original (human-written)<br>
🔴 <strong>CG</strong> — Computer-Generated (fake)
</p>
</div>
""", unsafe_allow_html=True)

# ── Main ───────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-title">Fake Review Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">LSTM + Attention Mechanism · E-commerce Review Analysis</p>', unsafe_allow_html=True)

# Load model
try:
    model, tokenizer, le = load_resources()
    attention_extractor = build_attention_extractor(model)
    model_loaded = True
except Exception as e:
    st.error(f"**Could not load model files.** Make sure `lstm_attention_s4.keras`, `tokenizer.pkl`, and `label_encoder.pkl` are in the same folder as `app.py`.\n\n`{e}`")
    model_loaded = False

# ── Input Section ──────────────────────────────────────────────────────────────
st.markdown("#### Paste a review below")

example_reviews = {
    "— select an example —": "",
    "Example: Genuine review": "I bought this blender three months ago and use it almost daily. It handles frozen fruit perfectly and cleans up easily. The motor is still running strong. Highly recommend for anyone who makes smoothies regularly.",
    "Example: Suspicious review": "I bought this product and this product is good. The product works well and the product quality is good. I use this product every day and the product does not have problems. This product is recommended for people who need a product like this product.",
}

example_choice = st.selectbox("Or load an example:", list(example_reviews.keys()))
default_text = example_reviews[example_choice]

review_input = st.text_area(
    "Review text",
    value=default_text,
    height=140,
    placeholder="Type or paste an e-commerce review here...",
    label_visibility="collapsed"
)

col_btn, col_clear = st.columns([1, 5])
with col_btn:
    analyze_btn = st.button("Analyze →", type="primary", use_container_width=True)

# ── Result ─────────────────────────────────────────────────────────────────────
if analyze_btn:
    if not review_input.strip():
        st.warning("Please enter a review before analyzing.")
    elif not model_loaded:
        st.error("Model is not loaded. Check the file paths.")
    else:
        with st.spinner("Analyzing..."):
            label, confidence, raw_prob, token_weights, clean_text = predict(
                review_input, model, tokenizer, le, attention_extractor
            )

        is_fake = label == "CG"

        # Result box
        if is_fake:
            st.markdown(f"""
<div class="result-box-fake">
    <div class="result-label" style="color:#be123c;">🔴 Fake Review Detected (CG)</div>
    <div class="result-desc">This review is likely <strong>computer-generated</strong>. 
    It may have been produced automatically to manipulate product ratings.</div>
</div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
<div class="result-box-real">
    <div class="result-label" style="color:#15803d;">🟢 Genuine Review (OR)</div>
    <div class="result-desc">This review appears to be <strong>written by a real person</strong>. 
    The language patterns are consistent with authentic user experiences.</div>
</div>""", unsafe_allow_html=True)

        # Confidence metrics
        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric("Prediction", "FAKE (CG)" if is_fake else "GENUINE (OR)")
        with m2:
            st.metric("Confidence", f"{confidence * 100:.1f}%")
        with m3:
            st.metric("Raw Score (sigmoid)", f"{raw_prob:.4f}")

        # Confidence bar
        bar_color = "#e11d48" if is_fake else "#16a34a"
        st.markdown(f"""
<div style="margin: 0.5rem 0 1.5rem 0;">
    <div style="font-size:0.8rem; color:#9ca3af; margin-bottom:4px;">Confidence</div>
    <div class="confidence-bar-wrap">
        <div style="height:8px; width:{confidence*100:.1f}%; background:{bar_color}; 
             border-radius:99px; transition: width 0.4s ease;"></div>
    </div>
    <div style="font-size:0.75rem; color:{bar_color}; text-align:right;">{confidence*100:.1f}%</div>
</div>
""", unsafe_allow_html=True)

        # Attention visualization
        st.markdown("---")
        st.markdown("#### Attention Analysis")
        st.markdown("Words the model focused on most when making its decision:")

        tab1, tab2 = st.tabs(["Word Highlights", "Attention Chart"])

        with tab1:
            render_token_pills(token_weights, is_fake)
            st.caption("Darker color = higher attention weight")

        with tab2:
            render_attention_heatmap(token_weights, is_fake)

        # Preprocessed text
        with st.expander("View preprocessed text"):
            st.code(clean_text, language=None)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center; color:#9ca3af; font-size:0.78rem; border-top: 1px solid #e5e7eb; padding-top: 1rem;">
Fake Review Detection · LSTM + Attention · S4 Model · Accuracy 93.76% · F1-Score 93.78%
</div>
""", unsafe_allow_html=True)
