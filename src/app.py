import streamlit as st
from search import BookSearchEngine
from index import lookup_metadata
from preprocess import expand_query


# page config must be the first streamlit call in the script
st.set_page_config(page_title="Book Search", page_icon="📚")


# engine setup

# load the engine once per session and reuse it across reruns — without this,
# every widget interaction would rebuild the BM25 index, reload embeddings,
# and reinitialise the transformer model from scratch
@st.cache_resource(show_spinner="Loading search engine…")
def load_engine():
    return BookSearchEngine()

engine = load_engine()


# sidebar

with st.sidebar:
    st.title("📚 Book Search")

    # display search options
    st.markdown("**Search method**")
    method = st.radio( # radio renders one option per line and makes the active method immediately visible
        label="method",
        options=["RRF", "BM25", "Semantic", "Boolean"],
        label_visibility="collapsed",  # the header above already acts as the label
    )

    st.markdown("**Options**")
    # when enabled, expand_query() appends WordNet synonyms to the query before searching
    expand = st.toggle("WordNet query expansion", value=False)
    # controls how many results are passed to the search method
    n_results = st.slider("Number of results", min_value=1, max_value=20, value=5)

    # show a short description of whichever method is currently selected
    # so the user understands what they are getting without leaving the page
    descriptions = {
        "BM25":     "Keyword ranking based on term frequency and document length. "
                    "Best for exact word matches.",
        "Semantic": "Dense vector similarity using a sentence transformer. "
                    "Understands meaning even when exact words differ.",
        "RRF":      "Reciprocal Rank Fusion — combines BM25 and Semantic rankings. "
                    "Tends to be the most robust of the three.",
        "Boolean":  "AND-search over the inverted index. Returns every book "
                    "containing all query terms, unranked.",
    }
    st.caption(descriptions[method])


# main area

st.title("Goodreads Book Search")

query = st.text_input(
    label="query",
    placeholder="e.g. dystopian society future",
    label_visibility="collapsed", # label_visibility hides the default label since the placeholder text is descriptive enough
)


# search and results

if query.strip():
    # apply expansion here rather than via expand=True on the engine method so we can
    # inspect and display the expanded string before passing it on to the search
    effective_query = expand_query(query.strip()) if expand else query.strip()

    # only show the expanded query if expansion actually added new terms
    if expand and effective_query != query.strip():
        st.caption(f"Expanded query: *{effective_query}*")

    # dispatch to the correct search method based on the sidebar selection
    if method == "BM25":
        results = engine.bm25_search(effective_query, n=n_results)
    elif method == "Semantic":
        results = engine.semantic_search(effective_query, top_k=n_results)
    elif method == "RRF":
        results = engine.rrf_search(effective_query, n=n_results)
    else:
        # boolean returns an unranked set of all matching documents — cap at n_results
        # so the page doesn't flood when a short query matches hundreds of books
        results = engine.boolean_search(effective_query)[:n_results]

    if not results:
        st.warning("No results found. Try a different query or search method.")
    else:
        for rank, doc in enumerate(results, start=1):
            # fetch display metadata from SQLite, same as _format_result in search.py
            meta = lookup_metadata(doc["id"]) or doc
            title   = meta.get("title", "Unknown")
            authors = ", ".join(meta.get("authors") or []) or "Unknown"
            year    = meta.get("year", "")
            genres  = meta.get("genres") or []
            desc    = doc.get("description", "")
            # note: books without a title or description are already dropped in preprocess.py,
            # so these fields are always populated — the fallbacks are just defensive

            # each result is nicely rendered as a bordered card
            with st.container(border=True):
                year_str = f" · {year}" if year else ""
                st.markdown(f"**{rank}. {title}**{year_str}")
                st.markdown(f"*{authors}*")

                # cap at 5 genre tags to keep the card compact
                if genres:
                    st.caption(" · ".join(genres[:5]))

                if desc:
                    # truncate at a word boundary to avoid cutting mid-word
                    excerpt = desc[:300].rsplit(" ", 1)[0] + "…" if len(desc) > 300 else desc
                    st.markdown(excerpt)
                    # full description available on demand without cluttering the card
                    if len(desc) > 300:
                        with st.expander("Read more"):
                            st.markdown(desc)
