import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Load the data
df = pd.read_csv("this.csv")

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
    print(result_df.head(30).to_string(index=False))

    # Save if needed
    # result_df.to_csv("price_gap_analysis_output.csv", index=False)
else:
    result_df = pd.DataFrame()  # Empty result
    print("No items from your restaurant matched with any competitor's menu based on the similarity criteria.")
