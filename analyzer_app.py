# -*- coding: utf-8 -*-
# analyzer_app_v1_17.py # Version with Indentation Fix at st.set_page_config + Bigger Points & Black Text for 3D Plot + ...

import streamlit as st
import pandas as pd
import re
import string
from io import BytesIO
import logging

# --- Analysis Libraries (Import with checks) ---
try:
    from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
    from sklearn.decomposition import LatentDirichletAllocation, NMF, PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.error("Scikit-learn library not found. Please install it (`pip install scikit-learn`). Analysis disabled.")
    st.stop()

try:
    import nltk
    from nltk.corpus import stopwords
    try: nltk.data.find('corpora/stopwords')
    except nltk.downloader.DownloadError:
        logging.info("NLTK stopwords not found. Attempting download.")
        try: nltk.download('stopwords', quiet=True); nltk.data.find('corpora/stopwords'); logging.info("NLTK stopwords downloaded.")
        except Exception as e_nltk: logging.error(f"Failed to download NLTK stopwords: {e_nltk}")
    STOPWORDS = stopwords.words('english')
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    STOPWORDS = []

try:
    import plotly.express as px
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# Corrected logging format string here:
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
DEFAULT_N_KEYWORDS = 20
DEFAULT_N_TOPICS = 5
DEFAULT_N_WORDS_PER_TOPIC = 10
DEFAULT_NGRAM_MAX = 2

# --- Text Preprocessing ---
def preprocess_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation + '’‘“”—'))
    text = re.sub(r'\d+', '', text)
    words = text.split()
    text = " ".join([word for word in words if len(word) > 2])
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# --- Analysis Functions ---
def perform_keyword_analysis(texts, n_keywords=DEFAULT_N_KEYWORDS, ngram_max=DEFAULT_NGRAM_MAX):
    if not texts: st.warning("No text data provided."); return None
    try:
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1500, ngram_range=(1, ngram_max))
        tfidf_matrix = vectorizer.fit_transform(texts)
        if tfidf_matrix.shape[1] == 0:
             st.warning(f"No valid terms/phrases (up to {ngram_max}-grams) found after filtering stopwords/frequency. Cannot calculate TF-IDF.")
             return None
        feature_names = vectorizer.get_feature_names_out()
        sum_tfidf = tfidf_matrix.sum(axis=0); scores = [(feature_names[col], sum_tfidf[0, col]) for col in range(tfidf_matrix.shape[1])]
        scores.sort(key=lambda x: x[1], reverse=True)
        df_keywords = pd.DataFrame(scores[:n_keywords], columns=['Keyword/Phrase', 'TF-IDF Score'])
        df_keywords['TF-IDF Score'] = df_keywords['TF-IDF Score'].map('{:,.3f}'.format)
        return df_keywords
    except ValueError as ve:
         if "empty vocabulary" in str(ve): st.warning(f"No {ngram_max}-grams found meeting criteria. Try reducing n-gram max or checking text."); return None
         else: st.error(f"Error during keyword analysis: {ve}"); logging.exception("Keyword analysis error"); return None
    except Exception as e: st.error(f"Error during keyword analysis: {e}"); logging.exception("Keyword analysis error"); return None

def perform_topic_modeling(texts, n_topics=DEFAULT_N_TOPICS, n_words=DEFAULT_N_WORDS_PER_TOPIC, method='LDA', ngram_max=DEFAULT_NGRAM_MAX):
    if not texts or len(texts) < n_topics: st.warning(f"Need at least {n_topics} documents for {n_topics} topics (found {len(texts)})."); return None, None, None
    try:
        vectorizer = CountVectorizer(stop_words='english', max_df=0.90, min_df=3, max_features=1500, ngram_range=(1, ngram_max))
        doc_term_matrix = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out()
        if doc_term_matrix.shape[1] == 0: st.warning(f"No valid terms/phrases (up to {ngram_max}-grams) found meeting vectorizer criteria. Cannot perform topic modeling."); return None, None, None
        model = None
        if method == 'LDA': model = LatentDirichletAllocation(n_components=n_topics, random_state=42, n_jobs=-1)
        elif method == 'NMF': model = NMF(n_components=n_topics, random_state=42, init='nndsvda', max_iter=300)
        else: st.error("Invalid topic modeling method."); return None, None, None
        with st.spinner(f"Fitting {method} model ({n_topics} topics, up to {ngram_max}-grams)..."): model.fit(doc_term_matrix)
        doc_topic_matrix = model.transform(doc_term_matrix)
        topics = {}
        for topic_idx, topic_weights in enumerate(model.components_):
            top_word_indices = topic_weights.argsort()[:-n_words - 1:-1]
            top_words = [feature_names[i] for i in top_word_indices]
            topics[f"Topic {topic_idx + 1}"] = ", ".join(top_words)
        return topics, doc_topic_matrix, feature_names

    except ValueError as ve:
         if "empty vocabulary" in str(ve): st.warning(f"No {ngram_max}-grams found meeting criteria. Cannot perform topic modeling."); return None, None, None
         else: st.error(f"Error during topic modeling: {ve}"); logging.exception("Topic modeling error"); return None, None, None
    except Exception as e: st.error(f"Error during topic modeling: {e}"); logging.exception("Topic modeling error"); return None, None, None


def perform_3d_visualization(df_full, abstract_col, group_col, title_col, ngram_max=DEFAULT_NGRAM_MAX):
    if not PLOTLY_AVAILABLE: st.error("Plotly library not found. Please install it (`pip install plotly`). 3D plot disabled."); return
    if abstract_col not in df_full.columns: st.error(f"Abstract column '{abstract_col}' not found."); return
    if group_col not in df_full.columns: st.error(f"Grouping column '{group_col}' not found."); return
    if title_col not in df_full.columns:
         st.warning(f"Title column '{title_col}' not found. Hover data will be limited.")
         title_col = None

    df_vis = df_full[[abstract_col, group_col] + ([title_col] if title_col else [])].copy()
    df_vis.dropna(subset=[abstract_col, group_col], inplace=True)
    df_vis['Processed_Abstract'] = df_vis[abstract_col].apply(preprocess_text)
    df_vis = df_vis[df_vis['Processed_Abstract'].str.len() > 10]

    n_docs = len(df_vis)
    if n_docs < 3: st.warning(f"Need at least 3 documents with abstracts and group labels for 3D visualization (found {n_docs})."); return

    st.write(f"Generating 3D map for **{n_docs}** abstracts...")

    try:
        with st.spinner(f"Vectorizing {n_docs} abstracts (up to {ngram_max}-grams)..."):
            vectorizer = TfidfVectorizer(stop_words='english', max_features=3000, ngram_range=(1, ngram_max))
            tfidf_matrix = vectorizer.fit_transform(df_vis['Processed_Abstract']).toarray()

        if tfidf_matrix.shape[1] < 3: st.warning(f"Only {tfidf_matrix.shape[1]} features found after vectorization. Cannot perform 3D PCA."); return

        with st.spinner("Performing PCA for dimensionality reduction..."):
            pca = PCA(n_components=3, random_state=42)
            pca_result = pca.fit_transform(tfidf_matrix)
            df_vis['PCA1'] = pca_result[:, 0]
            df_vis['PCA2'] = pca_result[:, 1]
            df_vis['PCA3'] = pca_result[:, 2]
            explained_variance = pca.explained_variance_ratio_.sum() * 100
            st.info(f"PCA complete. 3 components explain {explained_variance:.2f}% of the variance.")


        num_unique_groups = df_vis[group_col].nunique()
        if num_unique_groups > 20:
            st.warning(f"Too many groups ({num_unique_groups}) to color effectively. Plot might be hard to read.")


        with st.spinner("Generating 3D plot..."):
            hover_data_cols = [group_col]
            if title_col: hover_data_cols.append(title_col)

            fig = px.scatter_3d(df_vis, x='PCA1', y='PCA2', z='PCA3',
                                color=group_col,
                                hover_name=title_col if title_col else None,
                                hover_data=hover_data_cols,
                                title=f"3D Distribution of Abstracts by '{group_col}' (PCA Reduced, {ngram_max}-grams)",
                                labels={'PCA1': 'PCA Component 1', 'PCA2': 'PCA Component 2', 'PCA3': 'PCA Component 3'})
            fig.update_traces(marker=dict(size=5, opacity=0.8)) # Increased marker size to 5
            fig.update_layout(margin=dict(l=0, r=0, b=0, t=40),
                              legend_title_text=f'Group ({group_col})',
                              plot_bgcolor='white',
                              paper_bgcolor='white',
                              title_font=dict(color="#000"), # Black title font
                              legend_font=dict(color="#000"), # Black legend text font
                              scene=dict( # Style the 3D scene axes
                                xaxis=dict(tickfont=dict(color='#000'), titlefont=dict(color='#000'), gridcolor='lightgray', linecolor='black', zerolinecolor='lightgray'), # Black x-axis text/numbers
                                yaxis=dict(tickfont=dict(color='#000'), titlefont=dict(color='#000'), gridcolor='lightgray', linecolor='black', zerolinecolor='lightgray'), # Black y-axis text/numbers
                                zaxis=dict(tickfont=dict(color='#000'), titlefont=dict(color='#000'), gridcolor='lightgray', linecolor='black', zerolinecolor='lightgray')  # Black z-axis text/numbers
                              ))

            st.plotly_chart(fig, use_container_width=True)

    except Exception as e:
        st.error(f"Error during 3D visualization: {e}")
        logging.exception("3D Visualization Error")


def visualize_topic_distribution(df_filtered, doc_topic_matrix, topics, grouping_col, method_name, custom_topic_names=None): # Added custom_topic_names
    if not PLOTLY_AVAILABLE: st.error("Plotly library not found. Please install it (`pip install plotly`). Topic distribution plot disabled."); return
    if grouping_col not in df_filtered.columns: st.error(f"Grouping column '{grouping_col}' not found in DataFrame."); return
    if doc_topic_matrix is None or topics is None: st.warning("No topic modeling results to visualize."); return

    n_topics = len(topics)
    topic_names_default = list(topics.keys()) # Default topic names ("Topic 1", "Topic 2", ...)
    topic_names_display = []

    for i in range(n_topics):
        topic_name_default = topic_names_default[i]
        if custom_topic_names and topic_name_default in custom_topic_names and custom_topic_names[topic_name_default]:
            topic_names_display.append(custom_topic_names[topic_name_default]) # Use custom name if provided
        else:
            topic_names_display.append(topic_name_default) # Otherwise, use default name

    df_topic_dist = pd.DataFrame(doc_topic_matrix, columns=topic_names_default, index=df_filtered.index) # Use default names internally
    df_topic_dist.columns = topic_names_display # But set display names for plotting
    df_topic_dist = pd.concat([df_filtered[grouping_col], df_topic_dist], axis=1).dropna()

    if df_topic_dist.empty: st.warning("No data to visualize topic distribution after merging with grouping column."); return

    st.write(f"Visualizing **{method_name}** Topic Distribution by **'{grouping_col}'**...")

    try:
        df_grouped_topics = df_topic_dist.groupby(grouping_col)[topic_names_display].mean().reset_index()
        df_melted = pd.melt(df_grouped_topics, id_vars=[grouping_col], var_name='Topic', value_name='Average Probability')

        fig = px.bar(df_melted, x=grouping_col, y='Average Probability', color='Topic',
                     title=f"{method_name} Topic Distribution across '{grouping_col}'",
                     labels={'Average Probability': 'Average Topic Probability'})
        fig.update_layout(legend_title_text='Topics', margin=dict(l=0, r=0, b=0, t=60),
                          plot_bgcolor='white',
                          paper_bgcolor='white',
                          title_font=dict(color="#000"),
                          legend_font=dict(color="#000"),
                          xaxis_title_font=dict(color='#000'),
                          yaxis_title_font=dict(color='#000'),
                          xaxis_tickfont=dict(color='#000'),
                          yaxis_tickfont=dict(color='#000'))
        st.plotly_chart(fig, use_container_width=True)
#--- Streamlit App UI ---
st.set_page_config(layout="wide", page_title="Abstract Analyzer")
st.title("Bibliography Abstract Analyzer")
st.markdown("Upload the CSV from the Scraper. Analyze keywords/topics, visualize document distribution, or explore topic distributions.")

# File Upload
uploaded_file = st.file_uploader("Upload Processed Bibliography CSV", type="csv", key="analyzer_upload")

# Analysis Setup & Execution
if uploaded_file is not None:
    try:
        try: df = pd.read_csv(uploaded_file)
        except UnicodeDecodeError: df = pd.read_csv(uploaded_file, encoding='latin1'); st.info("Read CSV with latin1 encoding.")
        except Exception as e_read: st.error(f"Error reading CSV file: {e_read}"); st.stop()
        if df.empty: st.error("Uploaded CSV is empty."); st.stop()

        st.write("### Data Preview"); st.dataframe(df.head())
        st.write("### Analysis Setup")

        # --- UI Refactoring - Parameters First ---
        col_params1, col_params2, _, _ = st.columns(4) # Using columns for layout

        with col_params1:
            n_topics_input = st.slider("Number of Topics:", 2, 15, DEFAULT_N_TOPICS, key="n_topics_slider", help="Number of topics to discover (for Topic Modeling and Distribution).")
        with col_params2:
            n_words_input = st.slider("Terms per Topic:", 5, 20, DEFAULT_N_WORDS_PER_TOPIC, key="n_words_slider", help="Number of top terms to display per topic.")

        ngram_max = st.slider("Max N-gram Size:", min_value=1, max_value=3, value=DEFAULT_NGRAM_MAX, help="1=words, 2=word pairs, 3=word triplets.")

        # --- Column Selection and Analysis Type ---
        col_select1, col_select2, col_select3, _ = st.columns(4)

        with col_select1:
            std_abstract_col = 'Scraped_Abstract'
            abstract_col_actual = next((c for c in df.columns if c.lower() in ['abstract', 'summary', std_abstract_col.lower()]), std_abstract_col if std_abstract_col in df.columns else df.columns[0] if not df.empty else None)
            if not abstract_col_actual or abstract_col_actual not in df.columns: st.warning(f"Could not auto-detect abstract column ('{std_abstract_col}'). Please select manually."); abstract_col_actual = df.columns[0] if not df.empty else None
            abstract_column = st.selectbox("1. Select Abstract Column:", df.columns, index=df.columns.get_loc(abstract_col_actual) if abstract_col_actual in df.columns else 0)

        with col_select2:
            std_title_col = 'Title' # Standard title column name
            title_col_actual = next((c for c in df.columns if c.lower() in ['article title', std_title_col.lower()]), std_title_col if std_title_col in df.columns else None) # Find actual title column

            std_group_cols_options = ['Venue', 'Year', 'Scrape_Status']
            grouping_col_opts_actual = [next((c for c in df.columns if c.lower() == g.lower()), g) for g in std_group_cols_options if any(c.lower() == g.lower() for c in df.columns) or g in df.columns]
            other_potential_group_cols = [c for c in df.columns if c not in grouping_col_opts_actual and (df[c].dtype == 'object' or df[c].nunique() < 100)]
            all_grouping_cols = grouping_col_opts_actual + other_potential_group_cols

            if not all_grouping_cols: st.warning("No suitable grouping columns found."); grouping_column = None
            else: grouping_column = st.selectbox("2. Select Grouping Column:", all_grouping_cols, index=0, help="Column for coloring 3D plot, filtering analyses, or grouping for topic distribution.")

        with col_select3:
            analysis_options = ['Keyword Analysis (TF-IDF)', 'Topic Modeling (LDA)']
            if PLOTLY_AVAILABLE: analysis_options.extend(['3D N-gram Distribution Map', 'Topic Distribution Visualization'])
            analysis_type = st.selectbox("3. Select Analysis Type:", analysis_options)

        # Group Selection (relevant for Keyword/Topic Analysis, NOT for Topic Distribution or 3D map for now)
        selected_group = None
        is_3d_map = (analysis_type == '3D N-gram Distribution Map')
        is_topic_dist_vis = (analysis_type == 'Topic Distribution Visualization')
        if grouping_column and not (is_3d_map or is_topic_dist_vis): # Group selection only for Keyword/Topic Analysis
            try:
                groups_from_col = df[grouping_column].dropna().unique()
                try: sorted_groups = sorted(list(groups_from_col))
                except TypeError: sorted_groups = sorted(list(groups_from_col), key=str)
                unique_groups = ['All Groups Combined'] + sorted_groups
                selected_group = st.selectbox(f"Select Group from '{grouping_column}':", unique_groups, index=0, key="group_select", help="Select 'All Groups Combined' or a specific group for Keywords/Topics.")
            except KeyError: # Corrected SyntaxError here - changed ' to "
                st.error(f"Grouping column '{grouping_column}' not found.")
                selected_group = None
            except Exception as e_group: st.error(f"Error processing grouping column '{grouping_column}': {e_group}"); selected_group = None
        else: selected_group = 'All Groups Combined' # Default to all for 3D map and topic dist

        # --- Analysis Execution ---
        if selected_group and abstract_column and grouping_column:
            st.write("---")

            if is_3d_map:
                st.write(f"### 3D N-gram Distribution Map (Colored by '{grouping_column}')")
                if 'title_col_actual' not in locals() and 'title_col_actual' not in globals():
                    st.error("Error: title_col_actual is not defined before calling 3D visualization. Please report this issue.")
                else:
                    perform_3d_visualization(df, abstract_column, grouping_column, title_col_actual, ngram_max=ngram_max)
            elif is_topic_dist_vis:
                st.write(f"### Topic Distribution Visualization by '{grouping_column}'")
                df_filtered = df.copy() # Start with a copy to avoid modifying original df
                if not df_filtered.empty and abstract_column in df_filtered.columns:
                    # 1. Preprocess and store in DataFrame
                    with st.spinner("Preprocessing text..."):
                        df_filtered['Processed_Abstract'] = df_filtered[abstract_column].apply(preprocess_text)
                    # 2. Filter DataFrame for valid texts
                    df_filtered = df_filtered[df_filtered['Processed_Abstract'].str.len() > 10] # Or your length threshold
                    if df_filtered.empty: st.warning("No valid text remaining after preprocessing."); st.stop()

                    texts = df_filtered['Processed_Abstract'].tolist() # Get texts from filtered DataFrame
                    st.write(f"Performing Topic Modeling on **{len(texts)}** abstracts...")
                    if texts:
                        with st.spinner("Performing Topic Modeling (LDA)..."):
                            topics, doc_topic_matrix, _ = perform_topic_modeling(texts, n_topics=n_topics_input, n_words=n_words_input, method='LDA', ngram_max=ngram_max) # Use texts from filtered df
                            if topics and doc_topic_matrix is not None:
                                st.write("#### Discovered Topics:");
                                topic_names_input = {} # Dictionary to store custom topic names
                                for topic_name, words in topics.items():
                                    st.markdown(f"**{topic_name}:** {words}")
                                    topic_names_input[topic_name] = st.text_input(f"Name for {topic_name} (optional):", key=f"topic_name_{topic_name}") # Text input for topic name

                                visualize_topic_distribution(df_filtered, doc_topic_matrix, topics, grouping_column, method_name='LDA', custom_topic_names=topic_names_input) # Pass custom names
                            else: st.warning("Topic modeling failed, cannot visualize distribution.")
                    else: st.warning(f"No non-empty abstracts in '{abstract_column}'.")
                else: st.warning(f"No data or no '{abstract_column}' column available for topic distribution analysis.")

            else: # Keyword or Topic Analysis (No changes needed here for index fix)
                st.write(f"### Analysis Results for: {selected_group} (using up to {ngram_max}-grams)")
                df_filtered = df if selected_group == 'All Groups Combined' else df[df[grouping_column].astype(str) == str(selected_group)]
                if not df_filtered.empty and abstract_col in df_filtered.columns:
                    texts = df_filtered[abstract_col].dropna().astype(str).tolist()
                    st.write(f"Analyzing **{len(texts)}** abstracts...")
                    if texts:
                        with st.spinner("Preprocessing text..."):
                            processed_texts = [preprocess_text(text) for text in texts]
                            processed_texts = [text for text in processed_texts if text]
                        if not processed_texts: st.warning("No valid text remaining after preprocessing.")
                        else:
                            if analysis_type == 'Keyword Analysis (TF-IDF)':
                                 n_keywords = st.slider("Number of Keywords/Phrases:", 5, 50, DEFAULT_N_KEYWORDS)
                                 keywords_df = perform_keyword_analysis(processed_texts, n_keywords=n_keywords, ngram_max=ngram_max)
                                 if keywords_df is not None: st.write(f"#### Top {ngram_max}-grams (TF-IDF Score):"); st.dataframe(keywords_df, use_container_width=True)
                            elif analysis_type == 'Topic Modeling (LDA)':
                                topics, _, _ = perform_topic_modeling(processed_texts, n_topics=n_topics_input, n_words=n_words_input, method='LDA', ngram_max=ngram_max)
                                if topics: st.write(f"#### Discovered Topics (LDA, up to {ngram_max}-grams):");
                                for topic_name, words in topics.items(): st.markdown(f"**{topic_name}:** {words}")
                    else: st.warning(f"No non-empty abstracts in '{abstract_column}' for group '{selected_group}'.")
                else: st.warning(f"No data or no '{abstract_column}' column for group '{selected_group}'.")

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        logging.exception("General analysis error")

# --- Footer ---
st.markdown("---")
st.markdown("Abstract Analyzer v1.17 | Indentation Fix at st.set_page_config, Bigger Points & Black Text in 3D Plot, ..., Developed with diligence.")