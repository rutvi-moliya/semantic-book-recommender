import os
import html
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

import gradio as gr

load_dotenv()

# ----------------------------
# Load data
# ----------------------------
books = pd.read_csv("books_with_emotions.csv")

books["large_thumbnail"] = books["thumbnail"].fillna("")
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].str.strip() == "",
    "cover-not-found.jpg",
    books["large_thumbnail"] + "&fife=w800",
)

db_books = Chroma(
    persist_directory="./chroma_db",
    embedding_function=OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY")),
)

last_recommendations_df = pd.DataFrame()

# ----------------------------
# Helpers
# ----------------------------
def format_authors(authors: str) -> str:
    if pd.isna(authors) or not str(authors).strip():
        return "Unknown author"

    authors_split = [a.strip() for a in str(authors).split(";") if a.strip()]

    if len(authors_split) == 1:
        return authors_split[0]
    if len(authors_split) == 2:
        return f"{authors_split[0]} and {authors_split[1]}"
    return f"{', '.join(authors_split[:-1])}, and {authors_split[-1]}"


def get_tone_explanation(row: pd.Series, tone: str) -> str:
    if tone == "Happy":
        return f"Joy-focused reranking (joy={row.get('joy', 0):.2f})"
    elif tone == "Surprising":
        return f"Surprise-focused reranking (surprise={row.get('surprise', 0):.2f})"
    elif tone == "Angry":
        return f"Anger-focused reranking (anger={row.get('anger', 0):.2f})"
    elif tone == "Suspenseful":
        return f"Suspense-focused reranking (fear={row.get('fear', 0):.2f})"
    elif tone == "Sad":
        return f"Sadness-focused reranking (sadness={row.get('sadness', 0):.2f})"
    return "Ranked by semantic similarity to your query"


def retrieve_semantic_recommendations(
    query: str,
    category: str = "All",
    tone: str = "All",
    initial_top_k: int = 50,
    final_top_k: int = 12,
) -> pd.DataFrame:
    recs = db_books.similarity_search(query, k=initial_top_k)

    books_list = []
    for rec in recs:
        isbn_raw = rec.page_content.strip('"').split()[0].replace('"', "")
        try:
            books_list.append(int(isbn_raw))
        except ValueError:
            continue

    book_recs = books[books["isbn13"].isin(books_list)].copy()

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category].copy()

    if tone == "Happy":
        book_recs = book_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by="surprise", ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by="fear", ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values(by="sadness", ascending=False)

    return book_recs.head(final_top_k)


def build_gallery_items(recommendations: pd.DataFrame):
    items = []
    for _, row in recommendations.iterrows():
        title = row.get("title", "Unknown title")
        author = format_authors(row.get("authors", ""))
        items.append((row["large_thumbnail"], f"{title} — {author}"))
    return items


def build_details_html(row: pd.Series, tone: str, top_match: bool = False) -> str:
    title = html.escape(str(row.get("title", "Unknown title")))
    authors = html.escape(format_authors(row.get("authors", "")))
    category = html.escape(str(row.get("simple_categories", "Unknown")))
    description = html.escape(str(row.get("description", "No description available.")))
    explanation = html.escape(get_tone_explanation(row, tone))

    badge_html = '<div class="detail-badge">Top Match</div>' if top_match else ""

    return f"""
    <div class="detail-card">
        {badge_html}
        <div class="detail-category">{category}</div>
        <div class="detail-title">{title}</div>
        <div class="detail-author">by {authors}</div>
        <div class="detail-why"><b>Why this matches:</b> {explanation}</div>
        <div class="detail-desc">{description}</div>
    </div>
    """


def recommend_books(query: str, category: str, tone: str):
    global last_recommendations_df

    if not query or not query.strip():
        empty = """
        <div class='detail-card'>
            <div class='detail-badge'>Start Here</div>
            <div class='detail-title'>Describe the kind of book you want</div>
            <div class='detail-desc'>
                Try something like:
                <br><br>
                • A dark family saga with secrets across generations<br>
                • A hopeful novel about healing and second chances<br>
                • A historical nonfiction book about war and politics
            </div>
        </div>
        """
        return [], None, empty

    recommendations = retrieve_semantic_recommendations(query, category, tone)
    last_recommendations_df = recommendations.copy()

    if recommendations.empty:
        empty = """
        <div class='detail-card'>
            <div class='detail-title'>No matches found</div>
            <div class='detail-desc'>
                Try a broader query, change category to <b>All</b>, or switch the emotional tone.
            </div>
        </div>
        """
        return [], None, empty

    gallery_items = build_gallery_items(recommendations)
    first_row = recommendations.iloc[0]

    return gallery_items, first_row["large_thumbnail"], build_details_html(first_row, tone, top_match=True)


def on_select(evt: gr.SelectData, tone: str):
    global last_recommendations_df

    if last_recommendations_df.empty:
        return None, ""

    idx = evt.index
    if isinstance(idx, tuple):
        idx = idx[0]

    if idx >= len(last_recommendations_df):
        return None, ""

    row = last_recommendations_df.iloc[idx]
    return row["large_thumbnail"], build_details_html(row, tone, top_match=False)


categories = ["All"] + sorted(books["simple_categories"].dropna().unique().tolist())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

examples = [
    ["A story about a troubled family set across many generations", "All", "Suspenseful"],
    ["A heartwarming novel about healing and second chances", "Fiction", "Happy"],
    ["A serious book about war, politics, and real historical events", "Nonfiction", "Sad"],
    ["A magical adventure with mystery and danger", "Fiction", "Surprising"],
]

custom_css = """
.gradio-container {
    max-width: 96% !important;
    background:
        radial-gradient(circle at top left, rgba(79,70,229,0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(59,130,246,0.14), transparent 24%),
        linear-gradient(180deg, #0b1020 0%, #111827 100%) !important;
    color: white !important;
}
footer {display:none !important;}

.hero {
    padding: 8px 6px 18px 6px;
}
.hero-title {
    font-size: 46px;
    font-weight: 900;
    color: #f8fafc;
    margin-bottom: 10px;
    letter-spacing: -0.03em;
}
.hero-subtitle {
    font-size: 18px;
    color: #cbd5e1;
    max-width: 920px;
    line-height: 1.75;
}
.hero-subtitle b {
    background: linear-gradient(90deg, #60a5fa, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

.panel {
    background: rgba(255,255,255,0.045);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 22px;
    padding: 18px;
    backdrop-filter: blur(12px);
    box-shadow: 0 10px 30px rgba(0,0,0,0.22);
}

.section-title {
    color: #f8fafc;
    font-size: 30px;
    font-weight: 900;
    margin: 14px 0 10px 0;
    letter-spacing: -0.02em;
}

#find-btn {
    background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%) !important;
    border: none !important;
    color: white !important;
    font-weight: 800 !important;
    box-shadow: 0 8px 20px rgba(99,102,241,0.35);
}
#find-btn:hover {
    filter: brightness(1.08);
    transform: translateY(-1px);
    transition: all 0.2s ease;
}

.detail-card {
    background: linear-gradient(180deg, #131c31 0%, #0f172a 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 22px;
    padding: 24px;
    color: white;
    min-height: 440px;
    box-shadow: 0 14px 30px rgba(0,0,0,0.28);
}
.detail-badge {
    display: inline-block;
    background: linear-gradient(90deg, #f59e0b, #fb7185);
    color: white;
    font-size: 12px;
    font-weight: 800;
    padding: 6px 10px;
    border-radius: 999px;
    margin-bottom: 12px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.detail-category {
    font-size: 12px;
    font-weight: 700;
    color: #93c5fd;
    margin-bottom: 8px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
.detail-title {
    font-size: 32px;
    font-weight: 900;
    line-height: 1.2;
    margin-bottom: 10px;
    color: #ffffff;
}
.detail-author {
    font-size: 16px;
    color: #cbd5e1;
    margin-bottom: 16px;
}
.detail-why {
    background: rgba(59,130,246,0.12);
    border: 1px solid rgba(96,165,250,0.25);
    color: #dbeafe;
    padding: 12px 14px;
    border-radius: 12px;
    margin-bottom: 18px;
    font-size: 14px;
    line-height: 1.5;
}
.detail-desc {
    font-size: 15px;
    color: #e5e7eb;
    line-height: 1.85;
}

.tech-footer {
    margin-top: 10px;
    color: #94a3b8;
    font-size: 14px;
    line-height: 1.7;
}
.tech-footer b {
    color: #e2e8f0;
}

.gallery-wrap img {
    border-radius: 14px !important;
}
"""

with gr.Blocks(theme=gr.themes.Base(), css=custom_css, title="Semantic Book Recommender") as dashboard:
    gr.Markdown("""
    <div class="hero">
        <div class="hero-title">Semantic Book Recommender</div>
        <div class="hero-subtitle">
            Discover books using <b>semantic search</b>, then refine recommendations by
            <b>category</b> and <b>emotional tone</b>.
        </div>
    </div>
    """)

    with gr.Group(elem_classes="panel"):
        with gr.Row():
            query_box = gr.Textbox(
                label="Describe the kind of book you want",
                placeholder="e.g. A dark family saga with secrets, grief, and generational conflict",
                scale=3,
            )
            category_dropdown = gr.Dropdown(
                choices=categories,
                value="All",
                label="Category",
                scale=1,
            )
            tone_dropdown = gr.Dropdown(
                choices=tones,
                value="All",
                label="Emotional tone",
                scale=1,
            )

        with gr.Row():
            submit_button = gr.Button("Find recommendations", elem_id="find-btn")
            clear_button = gr.Button("Clear")

    gr.Examples(
        examples=examples,
        inputs=[query_box, category_dropdown, tone_dropdown],
        label="Example searches",
    )

    gr.Markdown('<div class="section-title">Recommendations</div>')
    with gr.Group(elem_classes="panel"):
        gallery = gr.Gallery(
            label="Recommended books",
            columns=6,
            rows=2,
            height=460,
            object_fit="cover",
            elem_classes="gallery-wrap",
        )

    gr.Markdown('<div class="section-title">Selected Book</div>')
    with gr.Group(elem_classes="panel"):
        with gr.Row():
            selected_image = gr.Image(label="Book cover", height=480)
            selected_details = gr.HTML()

        gr.Markdown("""
        <div class="tech-footer">
            Built with <b>OpenAI Embeddings</b>, <b>LangChain</b>, <b>ChromaDB</b>, and <b>Gradio</b>.
        </div>
        """)

    submit_button.click(
        fn=recommend_books,
        inputs=[query_box, category_dropdown, tone_dropdown],
        outputs=[gallery, selected_image, selected_details],
    )

    gallery.select(
        fn=on_select,
        inputs=[tone_dropdown],
        outputs=[selected_image, selected_details],
    )

    clear_button.click(
        fn=lambda: ("", "All", "All", [], None, ""),
        inputs=[],
        outputs=[query_box, category_dropdown, tone_dropdown, gallery, selected_image, selected_details],
    )

if __name__ == "__main__":
    dashboard.launch()