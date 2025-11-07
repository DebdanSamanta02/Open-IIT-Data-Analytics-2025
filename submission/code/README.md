# Code Module Reference

This document details the purpose and key functions of the 4 core Python files and 1 Jupyter Notebook in this project.

---

## `data_preprocessing.py`

**Purpose:** This module is the foundation of the pipeline. It is responsible for all data ingestion, cleaning, merging, and feature engineering. It takes the raw CSV/TSV files and transforms them into analysis-ready Pandas DataFrames.

**Key Functions:**
* `load_netflix_datasets()`: Loads the `netflix_titles.csv` (2021) and the 2025 Movies/TV show datasets.
* `load_and_prep_imdb()`: Loads and merges the `title.basics.tsv` and `title.ratings.tsv` files from IMDb.
* `clean_2021_dataset()`: Cleans the 2021 dataset, handles missing values (director, cast, country), and parses dates.
* `clean_2025_dataset()`: Cleans and combines the 2025 movie and TV show datasets, which have a richer, different schema.
* `parse_multi_value_column()`: A helper function that splits comma-separated strings (like in `cast` or `director`) into Python lists.
* `engineer_netflix_features()`: Creates new analytical columns, including `year_added`, `month_added`, `acquisition_lag` (release year vs. add year), and `primary_genre`.
* `create_genre_matrix()`: Generates a one-hot encoded (binary) matrix of all genres, used for association rule mining.
* `map_imdb_to_netflix()`: Merges the clean Netflix data with the IMDb data (on title and year) to add `averageRating` and `numVotes` for the Quadrant Analysis.
* `main_preprocessing_pipeline()`: The main wrapper function that executes all the above steps in the correct order and returns a dictionary of the final, clean DataFrames.

---

## `netflix_analysis.ipynb`

**Purpose:** This is the main controller notebook for the entire project. It imports all other modules and orchestrates the analysis pipeline from start to finish. Running this single file executes the entire analysis and saves all outputs.

**Key Responsibilities:**
* Defines all global variables, such as file paths (`DATA_DIR`, `VIZ_DIR`), analysis parameters (`TOP_N_COUNTRIES`), and plotting themes (`PLOTLY_TEMPLATE`).
* Installs all required Python packages from a consolidated list.
* Calls `data_preprocessing.main_preprocessing_pipeline()` to load and prepare all data.
* **EDA Section:** Calls functions from `visualization_functions.py` to generate and save all exploratory plots (e.g., `plot_content_trend_lines`, `plot_choropleth_map`).
* **Text Analysis Section:** Calls functions from `text_analysis.py` to run the full NLP pipeline (e.g., `preprocess_text_column`, `plot_topic_trends_over_time`).
* **Statistical Analysis Section:** Calls functions from `statistical_analysis.py` to run advanced models (e.g., `plot_time_series_decomposition`, `perform_quadrant_analysis`, `build_collaboration_network`).
* Prints status updates and analysis summaries (like top genres or network centrality) to the console.
* Ensures all outputs (HTML files, PNGs, and data CSVs) are saved to the correct `visualizations/` and `analysis_outputs/` directories.

---

## `statistical_analysis.py`

**Purpose:** This module contains all advanced statistical models, network analysis, and strategic business analysis functions. It focuses on generating high-level, data-driven insights beyond basic EDA.

**Key Functions:**
* `plot_time_series_decomposition()`: Uses `statsmodels` to decompose monthly content additions into `trend`, `seasonal`, and `residual` components.
* `calculate_hhi()`: Computes the Herfindahl-Hirschman Index (HHI) to measure genre concentration and portfolio diversification.
* `plot_long_tail()`: Visualizes the "long tail" distribution of the content library to see the split between hit titles and niche content.
* `perform_quadrant_analysis()`: The core logic for the strategic quadrant plot. It calculates the supply growth (CAGR) and average quality (IMDb rating) for each genre.
* `plot_quadrant_analysis()`: Plots the 2x2 scatter plot from the data generated above.
* `build_collaboration_network()`: Uses `NetworkX` to create a graph object from the 2025 dataset, linking directors and actors.
* `get_network_centrality()`: Calculates `degree`, `betweenness`, and `eigenvector` centrality to find the most important creators in the network.
* `run_association_rules()`: Uses `mlxtend`'s Apriori algorithm on the genre matrix to find which genres are most frequently paired together (e.g., "International" and "Drama").

---

## `text_analysis.py`

**Purpose:** This module handles all Natural Language Processing (NLP) tasks. It focuses on extracting insights from the `description` column of the Netflix dataset.

**Key Functions:**
* `download_nltk_data()`: A helper to ensure `stopwords` and `punkt` (tokenizer) are available.
* `clean_text()`: A text-cleaning function that converts text to lowercase, removes punctuation, URLs, and stopwords.
* `preprocess_text_column()`: Applies the cleaning function to the entire `description` column.
* `get_sentiment()`: Calculates the sentiment polarity (positive/negative) of each description using `TextBlob`.
* `get_top_ngrams()`: Uses `sklearn.feature_extraction.text.CountVectorizer` to find the most frequent multi-word phrases (bigrams and trigrams) like "high school" or "save the world".
* `perform_lda_topic_modeling()`: Trains a Latent Dirichlet Allocation (LDA) model to automatically discover a set number of hidden "topics" or "themes" from the descriptions (e.g., "Crime & Mystery", "Family & Relationships").
* `plot_topic_trends_over_time()`: Visualizes how the prevalence of these LDA topics has changed year-over-year.
* `generate_wordcloud()`: Creates and saves a `.png` image of a word cloud from the corpus of descriptions.

---

## `visualization_functions.py`

**Purpose:** This module is a dedicated library of reusable plotting functions. It uses `Plotly` to generate all the interactive charts for the project. Using this module ensures all visualizations have a consistent theme, color scheme, and structure.

**Key Functions:**
* `plot_content_type_pie()`: Creates the main "Movie vs. TV Show" pie chart.
* `plot_content_trend_lines()`: Generates the line chart of content additions over time.
* `plot_top_n_bar()`: A generic function to create horizontal bar charts, used for Top 10 Genres, Top 15 Countries, etc.
* `plot_choropleth_map()`: Generates the world map (choropleth) shaded by content production.
* `plot_lag_time_trend()`: Creates the line chart for the "Acquisition Lag" analysis.
* `plot_monthly_heatmap()`: Generates the heatmap showing content additions by month and year.
* `plot_sankey_collaborations()`: A specialized function to create Sankey diagrams, used for showing co-production flows between countries.
* `plot_bubble_chart()`: Creates 3-dimensional bubble charts (e.g., popularity vs. rating, with bubble size as volume).
* `plot_dual_axis_chart()`: Creates charts with two different Y-axes (e.g., bar chart for movie count, line chart for revenue).
