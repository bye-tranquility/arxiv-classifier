import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import streamlit as st
import torch
import numpy as np

# Global constants
MODEL_PATH = "./model_final"
MAX_LENGTH = 256

CATEGORIES = {
    0: {"code": "cs",      "name": "Computer Science"},
    1: {"code": "econ",    "name": "Economics"},
    2: {"code": "eess",    "name": "Electrical Engineering and Systems Science"},
    3: {"code": "math",    "name": "Mathematics"},
    4: {"code": "physics", "name": "Physics"},
    5: {"code": "q-bio",   "name": "Quantitative Biology"},
    6: {"code": "q-fin",   "name": "Quantitative Finance"},
    7: {"code": "stat",    "name": "Statistics"},
}

EXAMPLES = [
    {
        "title": "Attention Is All You Need",
        "abstract": "The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder. The best performing models also connect the encoder and decoder through an attention mechanism. We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely. Experiments on two machine translation tasks show these models to be superior in quality while being more parallelizable and requiring significantly less time to train. Our model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU. On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.8 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature. We show that the Transformer generalizes well to other tasks by applying it successfully to English constituency parsing both with large and limited training data.",
    },
    {
        "title": "Shor's algorithm is possible with as few as 10,000 reconfigurable atomic qubits",
        "abstract": "Quantum computers have the potential to perform computational tasks beyond the reach of classical machines. A prominent example is Shor's algorithm for integer factorization and discrete logarithms, which is of both fundamental importance and practical relevance to cryptography. However, due to the high overhead of quantum error correction, optimized resource estimates for cryptographically relevant instances of Shor's algorithm require millions of physical qubits. Here, by leveraging advances in high-rate quantum error-correcting codes, efficient logical instruction sets, and circuit design, we show that Shor's algorithm can be executed at cryptographically relevant scales with as few as 10,000 reconfigurable atomic qubits. Increasing the number of physical qubits improves time efficiency by enabling greater parallelism; under plausible assumptions, the runtime for discrete logarithms on the P-256 elliptic curve could be just a few days for a system with 26,000 physical qubits, while the runtime for factoring RSA-2048 integers is one to two orders of magnitude longer. Recent neutral-atom experiments have demonstrated universal fault-tolerant operations below the error-correction threshold, computation on arrays of hundreds of qubits, and trapping arrays with more than 6,000 highly coherent qubits. Although substantial engineering challenges remain, our theoretical analysis indicates that an appropriately designed neutral-atom architecture could support quantum computation at cryptographically relevant scales. More broadly, these results highlight the capability of neutral atoms for fault-tolerant quantum computing with wide-ranging scientific and technological applications.",
    },
    {
        "title": "Optimal Portfolio Selection with Transaction Costs",
        "abstract": "Consider an investor who has the following instruments available to him: a bank account paying a fixed rate of interest r and n risky assets (“stocks”) whose prices are modeled as geometric Brownian motions. The investor is allowed to consume at a rate c(t) from the bank account and is subject to the constraint that he remain solvent at all times. Any trading in the stocks must be self-financing, and incurs a transaction cost which is proportional to the amount being traded. The investor’s objective is to maximize his expected discounted utility of lifetime consumption.",
    },
]

# Loading model
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model


def classify(text: str, tokenizer, model, threshold: float = 0.95) -> list[dict]:
    inputs = tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(**inputs).logits
        raw_probs = torch.sigmoid(logits).squeeze(0).numpy()

    total = raw_probs.sum()
    probs = raw_probs / total if total > 0 else raw_probs

    sorted_idx = np.argsort(probs)[::-1]

    results = []
    cumulative = 0.0
    for idx in sorted_idx:
        idx = int(idx)
        p = float(probs[idx])
        cumulative += p
        results.append({
            "code": CATEGORIES[idx]["code"],
            "name": CATEGORIES[idx]["name"],
            "prob": p,
            "cumulative": cumulative,
        })
        if cumulative >= threshold:
            break

    return results


def build_input_text(title: str, abstract: str) -> str:
    result = title.strip()
    if abstract.strip():
        result += " [SEP] " + abstract.strip()
    return result


# Session state
if "history" not in st.session_state:
    st.session_state.history = []

if "title_input" not in st.session_state:
    st.session_state.title_input = ""

if "abstract_input" not in st.session_state:
    st.session_state.abstract_input = ""

if "last_results" not in st.session_state:
    st.session_state.last_results = None


def clear_inputs():
    st.session_state.title_input = ""
    st.session_state.abstract_input = ""

def clear_history():
    st.session_state.history = []
    st.session_state.last_results = None

def set_example(title, abstract):
    st.session_state.title_input = title
    st.session_state.abstract_input = abstract

# Page
st.set_page_config(page_title="arXiv Category Classifier", layout="centered")

st.title("ARXIV CATEGORY CLASSIFIER")

with st.expander("Project Description"):
    st.subheader("Lightweight arXiv (or any other scientific) paper classifier based on SciBERT")
    st.write(
        "The classifier is built on SciBERT embeddings with a fine-tuned classification head, trained on the Kaggle arXiv dataset. You can predict article categories using either its title or abstract. Predictions are more accurate when both are provided."
    )
    st.write(
        "The model outputs normalized probabilities across 8 top-level arXiv categories and returns the most likely ones until cumulative confidence reaches 95%."
    )

# Input part
col_input, col_results = st.columns([1, 1], gap="large")

with col_input:
    title_value = st.text_input(
        "Title",
        key="title_input",
        placeholder="e.g. Attention Is All You Need",
    )

    abstract_value = st.text_area(
        "Abstract",
        key="abstract_input",
        height=150,
        placeholder="It is an optional field. Paste the abstract of an article here for better predictions.",
    )

    classify_btn = st.button("Classify", type="primary", use_container_width=True)
    
    col_btn2, col_btn3 = st.columns(2)
    with col_btn2:
        st.button("Clear input", on_click=clear_inputs, use_container_width=True)
    with col_btn3:
        st.button("Clear results", on_click=clear_history, use_container_width=True)

    with st.expander("QUICK EXAMPLES", expanded=True):
        for i, ex in enumerate(EXAMPLES):
            with st.container(border=True):
                st.write(f"**{ex['title']}**")
                st.caption(ex["abstract"][:120] + "...")
                st.button(
                    "Use this example", 
                    key=f"ex_{i}", 
                    on_click=set_example, 
                    args=(ex["title"], ex["abstract"])
                )

# Run classification
results = st.session_state.last_results

if classify_btn and title_value.strip():
    text = build_input_text(title_value, abstract_value)

    with st.spinner("Classifying..."):
        tokenizer, model = load_model()
        results = classify(text, tokenizer, model)
        st.session_state.last_results = results

    cats_summary = ", ".join(f"{r['code']} {r['prob']*100:.1f}%" for r in results)
    st.session_state.history.insert(0, {
        "title": title_value.strip()[:80] + ("..." if len(title_value.strip()) > 80 else ""),
        "has_abstract": bool(abstract_value.strip()),
        "categories": cats_summary,
        "full_title": title_value.strip(),
        "full_abstract": abstract_value.strip(),
    })

elif classify_btn:
    st.warning("Please enter a paper title.")

# Results panel
with col_results:
    if results:
        top = results[0]

        st.caption("TOP MATCHES")
        col_code, col_pct = st.columns([2, 1])
        with col_code:
            st.subheader(top["code"])
            st.write(top["name"])
        with col_pct:
            st.metric(label="Confidence", value=f"{top['prob']*100:.1f}%")

        st.divider()

        for r in results:
            col_name, col_prob = st.columns([3, 1])
            with col_name:
                st.write(f"**{r['code']}**  {r['name']}")
            with col_prob:
                st.write(f"**{r['prob']*100:.1f}%**")
            st.progress(r["prob"])

        st.divider()
        st.metric("Predictions coverage", f"{results[-1]['cumulative']*100:.1f}%")

    else:
        st.info("Results will appear here after classification.")
