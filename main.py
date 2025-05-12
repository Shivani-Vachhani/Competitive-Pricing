import streamlit as st
import pandas as pd
import json
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# === 🔐 Set your Groq API key ===
GROQ_API_KEY = "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"
GROQ_MODEL = "llama-3.3-70b-versatile"

# === Streamlit UI ===
st.title("🔍 Competitive Price Analyzer with Visual Insights")
st.write("Upload your pricing CSV to analyze item competitiveness using Groq LLM.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

def compare(uploaded_file):

    # Load the data
    df = uploaded_file
    rest_name=df["restaurant_name"][0]
    # Define your restaurant ID
    my_restaurant_id = 1

    # Split menus
    my_menu = df[df["restaurant_id"] == my_restaurant_id][["restaurant_item", "item_price"]].drop_duplicates()
    competitor_menu = df[df["restaurant_id"] != my_restaurant_id][["restaurant_id", "restaurant_name", "restaurant_item", "item_price"]].drop_duplicates()

    # Load model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Encode items
    my_embeddings = model.encode(my_menu["restaurant_item"].tolist(), convert_to_tensor=True)
    comp_embeddings = model.encode(competitor_menu["restaurant_item"].tolist(), convert_to_tensor=True)

    # Cosine similarity
    similarity_matrix = cosine_similarity(my_embeddings.cpu().numpy(), comp_embeddings.cpu().numpy())

    SIMILARITY_THRESHOLD = 0.7

    # Preprocessing function
    def preprocess(text):
        words = re.findall(r'\b\w+\b', text.lower())
        return set(w for w in words if w not in ENGLISH_STOP_WORDS)

    # Match logic
    matches = []

    for i, my_item in enumerate(my_menu.itertuples()):
        my_words = preprocess(my_item.restaurant_item)
        price_matches = []

        for j, comp_item in enumerate(competitor_menu.itertuples()):
            comp_words = preprocess(comp_item.restaurant_item)
            shared_words = my_words.intersection(comp_words)
            score = similarity_matrix[i, j]

            if score > SIMILARITY_THRESHOLD and len(shared_words) >= 2:
                price_matches.append({
                    "My Item": my_item.restaurant_item,
                    "My Price": my_item.item_price,
                    "Competitor": comp_item.restaurant_name,
                    "Comp Restaurant ID": comp_item.restaurant_id,
                    "Comp Item": comp_item.restaurant_item,
                    "Comp Price": comp_item.item_price,
                    "Similarity": round(score, 3),
                    "Price Diff": round(my_item.item_price - comp_item.item_price, 2),
                    "Shared Words": ", ".join(shared_words)
                })

        if price_matches:
            comp_prices = [m["Comp Price"] for m in price_matches]
            avg_price = round(sum(comp_prices) / len(comp_prices), 2)
            min_price = round(min(comp_prices), 2)
            max_price = round(max(comp_prices), 2)

            for match in price_matches:
                match["Avg Comp Price"] = avg_price
                match["Min Comp Price"] = min_price
                match["Max Comp Price"] = max_price
                match["Overpriced"] = match["My Price"] > max_price
                match["Underpriced"] = match["My Price"] < min_price

            matches.extend(price_matches)

    # Convert to DataFrame
    if matches:
        result_df = pd.DataFrame(matches)

        # Reorder and sort
        result_df = result_df[[
            "My Item", "My Price",
            "Competitor", "Comp Restaurant ID", "Comp Item", "Comp Price",
            "Similarity", "Price Diff", "Shared Words",
            "Avg Comp Price", "Min Comp Price", "Max Comp Price",
            "Overpriced", "Underpriced"
        ]].sort_values(by=["Similarity", "Price Diff"], ascending=[False, True])

        pd.set_option('display.max_colwidth', None)
        return [result_df,rest_name]
        # Save if needed
        # result_df.to_csv("price_gap_analysis_output.csv", index=False)
    else:
        result_df = pd.DataFrame()  # Empty result
        print("No items from your restaurant matched with any competitor's menu based on the similarity criteria.")


if uploaded_file:
    data = pd.read_csv(uploaded_file)
    response= compare(data)
    df = response[0]
    restaurant=response[1]
    st.subheader("📊 Preview Data")
    st.write(restaurant)
    st.dataframe(df)
    # Basic filtering
    similarity_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.8, 0.01)
    print("run",df[df['Similarity'] >= similarity_threshold])
    df_filtered = df[df['Similarity'] >= similarity_threshold]

    # Prepare for chart
    chart_df = df_filtered.copy()
    chart_df['Status'] = chart_df.apply(lambda row: "Overpriced" if row['Overpriced'] else ("Underpriced" if row['Underpriced'] else "Neutral"), axis=1)

    # === 📈 Bar Chart ===
    st.subheader("📉 Price Difference by Item")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(
        data=chart_df,
        x="My Item",
        y="Price Diff",
        hue="Status",
        palette={"Overpriced": "red", "Underpriced": "green", "Neutral": "gray"},
        dodge=False
    )
    ax.set_ylabel("Price Difference ($)")
    ax.set_xlabel("My Items")
    ax.set_title("Price Differences Compared to Competitors")
    ax.axhline(0, color="black", linestyle="--")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # === 📦 Send to Groq for Natural Language Summary ===
    data_rows = df_filtered.to_dict(orient='records')

    prompt = f"""
You are a pricing analyst. Given the following data, generate a natural-language summary
explaining which items are overpriced or underpriced, how large the difference is,
and provide recommendations for price adjustments.

Data:
{json.dumps(data_rows, indent=2)}

Format output like:
Item: [name], Status: [Overpriced/Underpriced], Diff: [$X], Avg Competitor: [$Y], Similarity: [0.9]
Recommendation: [what should be done]
"""

    st.subheader("🧠 LLM Analysis (Groq API)")
    if st.button("Run Full Analysis"):
        with st.spinner("Calling Groq model..."):

            headers = {
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json"
            }

            body = {
                "model": GROQ_MODEL,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }

            response = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=body)

            if response.status_code == 200:
                result = response.json()
                summary = result['choices'][0]['message']['content']
                st.success("✅ Summary Generated")
                st.markdown("### 📋 Summary of Pricing Analysis")
                st.markdown(summary)
            else:
                st.error(f"API Error {response.status_code}: {response.text}")
