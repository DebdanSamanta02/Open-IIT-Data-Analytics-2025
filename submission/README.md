# Netflix Data Analysis Project

## Getting Started

### 1. Data Setup

Before running the project, you must create a `data/` directory in the project's root folder and place the required data files inside it:

- `netflix_titles.csv` (The 2021 dataset)
- `netflix_movies_detailed_up_to_2025.csv` (The 2025 movies dataset)
- `netflix_tv_shows_detailed_up_to_2025.csv` (The 2025 TV shows dataset)
- `title.basics.tsv` (IMDb title basics)
- `title.ratings.tsv` (IMDb title ratings)

### 2. Install Dependencies

The project uses several Python libraries. Install all of them with this command:

```bash
pip install pandas rapidfuzz plotly scikit-learn nltk textblob wordcloud matplotlib scipy statsmodels networkx mlxtend
```

### 3. Run the Analysis

To run the entire analysis pipeline, simply execute the main script from your terminal:
```bash
python netflix_analysis.py
```

### 4. Code Architecture

This project follows a modular design with the following core Python scripts:

#### `netflix_analysis.py` (Main Script)
- Central orchestrator importing all modules.
- Defines global variables like file paths and analysis parameters.
- Runs the complete pipeline in the correct order.

#### `data_preprocessing.py`
- Handles all data ingestion, cleaning, and feature engineering.
- Loads the 2021, 2025, and IMDb datasets and merges them.
- Fills missing values and creates analytical features like `acquisition_lag` and `primary_genre`.

#### `visualization_functions.py`
- Contains reusable Plotly functions.
- Ensures consistent dark theme visualizations and Netflix brand colors.

#### `text_analysis.py`
- Handles the NLP pipeline for text cleaning, sentiment analysis, N-gram extraction, and LDA topic modeling for discovering “Narrative DNA.”

#### `statistical_analysis.py`
- Provides advanced modeling and strategic analytics:
  - Time-series decomposition
  - HHI and Quadrant Analysis
  - NetworkX centrality analyses
  - Apriori association rule mining

### 5. Project Outputs

After running `netflix_analysis.py`, two result directories will be created:

#### `visualizations/`
- Contains interactive `.html` charts and static `.png` wordclouds.

#### `analysis_outputs/`
- Exports analytical results as:
  - `quadrant_analysis_data.csv`
  - `genre_association_rules.csv`
  - `network_centrality.csv`

