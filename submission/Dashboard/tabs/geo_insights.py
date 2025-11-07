from dash import Dash, html, dcc, dash_table, Output, Input, State, callback_context
import pandas as pd
import plotly.express as px
import numpy as np

# --- data preprocessing --- 

# Load data
df = pd.read_csv('./data/netflix_titles.csv')

# Process data
# Type data
type_counts = df['type'].value_counts().reset_index()
type_counts.columns = ['Type', 'Count']

# year data
year_counts = df['release_year'].value_counts().sort_index().reset_index()
year_counts.columns = ['Release Year', 'Count']

# country data
country_list = df["country"].str.split(",").explode().str.strip()
country_counts = country_list.value_counts().head(20)
country_normalized_counts = country_list.value_counts(normalize=True)
shannon_index = -np.sum(country_normalized_counts * np.log2(country_normalized_counts))

titles = df['title'].count()
countries = df['country'].dropna().count()
start_value = year_counts['Count'].iloc[0]
end_value = year_counts['Count'].iloc[-1]
num_years = len(year_counts) - 1
CAGR = ((end_value / start_value) ** (1 / num_years) - 1) * 100

# language data
lang_keywords = {
    'Hindi': ['Bollywood', 'Indian'],
    'Japanese': ['Anime', 'Japanese'],
    'Korean': ['Korean', 'K-drama'],
    'Spanish': ['Spanish', 'Español', 'Mexico', 'Spanish-language'],
    'French': ['French', 'Paris'],
    'English': ['British', 'American', 'English']
}

language_counts = {}
for lang, keywords in lang_keywords.items():
    count = df[df['listed_in'].str.contains('|'.join(keywords), case=False, na=False) |
               df['description'].str.contains('|'.join(keywords), case=False, na=False)].shape[0]
    language_counts[lang] = count

lang_df = pd.DataFrame(list(language_counts.items()), columns=['Language', 'Count']).sort_values(by='Count', ascending=False)

# Histogram by Category
rating_groups = {
    'Kids': ['TV-Y', 'TV-Y7', 'TV-G', 'G', 'TV-Y7-FV'],
    'Teens': ['TV-PG', 'PG', 'PG-13', 'TV-14'],
    'Adults': ['TV-MA', 'R', 'NC-17', 'NR', 'UR']
}
def cat(x):
    for k, v in rating_groups.items():
        if x['rating'] in v:
            return k
    return 'Unknown'
df['category'] = df.apply(lambda x: cat(x), axis=1)

# --- Interactive World Map ---

# Country data for world map and top producers
country_list = df["country"].dropna().str.split(",").explode().str.strip()

# World map data
world_counts = country_list.value_counts().reset_index()
world_counts.columns = ['Country', 'Count']

# Add Rank for better hover info
world_counts['Rank'] = world_counts['Count'].rank(ascending=False).astype(int)
world_counts = world_counts.rename(columns={'Count': 'Titles'})

fig_world = px.choropleth(
    world_counts,
    locations='Country',
    locationmode='country names',
    color='Titles',
    color_continuous_scale='Reds',
    hover_name='Country',
    hover_data={
        'Titles': ':,',
        'Rank': True,
        'Country': False
    }
)

fig_world.update_layout(      
    title={
        'text': 'Netflix Titles by Each Country',
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 22, 'color': 'white'}
    },   
    height=600,
    margin=dict(l=0, r=0, t=40, b=0),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    geo=dict(
        showframe=False,
        showcoastlines=True,
        coastlinecolor='gray',
        projection_type='natural earth',
        bgcolor='rgba(0,0,0,0)',
        showocean=True,
        oceancolor='rgba(40,40,40,0.90)'   
    ),
     dragmode=False
)




# Country Comparison (Top 20)
country_counts = country_list.value_counts().head(20)

fig_country_bar = px.bar(
    x=country_counts.values[::-1],
    y=country_counts.index[::-1],
    orientation='h',
    color=country_counts.values[::-1],
    color_continuous_scale='Reds',
    title='Country Comparison: Top 20 Producers'
)
fig_country_bar.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    yaxis=dict(autorange='reversed')
)


# --- regional deep dives ---

# Create smaller DataFrame with necessary columns
df_small = df[['country', 'listed_in']].dropna(subset=['country', 'listed_in'])

# Explode multiple countries per title
df_exploded = df_small.assign(country_list=df_small['country'].str.split(','), genres_list=df_small['listed_in'].str.split(','))
df_exploded = df_exploded.explode('country_list').explode('genres_list')

# Clean text
df_exploded['country'] = df_exploded['country_list'].str.strip()
df_exploded['genre'] = df_exploded['genres_list'].str.strip()

# Remove blanks
df_exploded = df_exploded[
    (df_exploded['country'] != '') & (df_exploded['genre'] != '')
]

# Aggregate counts
df_agg = df_exploded.groupby(['country', 'genre']).size().reset_index(name='Count')

# Filter Top 15 countries and Top 15 genres
top_countries = df_agg.groupby('country')['Count'].sum().nlargest(15).index
top_genres = df_agg.groupby('genre')['Count'].sum().nlargest(15).index
df_filtered = df_agg[
    df_agg['country'].isin(top_countries) & df_agg['genre'].isin(top_genres)
]

# Pivot for heatmap
df_pivot = df_filtered.pivot(index='country', columns='genre', values='Count').fillna(0)

# Plot heatmap
fig_heatmap = px.imshow(
    df_pivot,
    text_auto=True,
    aspect="auto",
    color_continuous_scale='YlOrRd',
    title="Genre Concentration by Country (Top 15 Countries & Genres)",
    labels={'color': 'Titles'}  # shows 'Titles' instead of 'Count'
)

fig_heatmap.update_layout(
    title={
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 22, 'color': 'white'}
    },
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    margin=dict(l=80, r=40, t=80, b=60),
    height=700,
)

fig_heatmap.update_xaxes(title="Genre", tickangle=45, showgrid=False)
fig_heatmap.update_yaxes(title="Country", showgrid=False)


# Define Regional Mapping

region_map = {
    'North America': ['United States', 'Canada', 'Mexico'],
    'Europe': ['United Kingdom', 'France', 'Germany', 'Spain', 'Italy', 'Poland', 'Sweden'],
    'Asia-Pacific': ['India', 'Japan', 'South Korea', 'China', 'Thailand', 'Indonesia', 'Singapore'],
    'Latin America': ['Brazil', 'Argentina', 'Chile', 'Colombia', 'Peru'],
    'Middle East & Africa': ['United Arab Emirates', 'Egypt', 'South Africa', 'Nigeria', 'Turkey', 'Saudi Arabia']
}

# Create reversed lookup (country → region)
country_to_region = {country: region for region, countries in region_map.items() for country in countries}


# Prepare Region Data

df_region = df.copy()
df_region['country_list'] = df_region['country'].dropna().str.split(',')
df_region = df_region.explode('country_list')
df_region['country_list'] = df_region['country_list'].str.strip()
df_region['region'] = df_region['country_list'].map(country_to_region)
df_region = df_region.dropna(subset=['region'])


# Regional Content Mix

region_mix = df_region.groupby(['region', 'type']).size().reset_index(name='Count')

fig_region_bar = px.bar(
    region_mix,
    x='region',
    y='Count',
    color='type',
    barmode='group',
    color_discrete_sequence=['#E50914', '#B20710'],
    title='Regional Content Mix: Movies vs. TV Shows'
)
fig_region_bar.update_layout(
    title_x=0.5,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    height=500
)


# Top Genres per Region

df_genre = df_region.copy()
df_genre = df_genre.explode('listed_in')
df_genre['listed_in'] = df_genre['listed_in'].str.strip()

region_genre = (
    df_genre.groupby(['region', 'listed_in']).size().reset_index(name='Count')
)

# Keep top 5 genres per region
region_genre_top = region_genre.sort_values(['region', 'Count'], ascending=[True, False])
region_genre_top = region_genre_top.groupby('region').head(5)

fig_region_genre = px.bar(
    region_genre_top,
    x='region',
    y='Count',
    color='listed_in',
    title='Top 5 Genres per Region',
    barmode='stack',
    color_discrete_sequence=px.colors.sequential.Reds
)
fig_region_genre.update_layout(
    title_x=0.5,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    height=600
)


# Regional Growth Trends

df_region['date_added'] = pd.to_datetime(df_region['date_added'], errors='coerce')
df_region['year_added'] = df_region['date_added'].dt.year
region_trend = (
    df_region.groupby(['region', 'year_added']).size().reset_index(name='Count')
)
region_trend = region_trend[
    (region_trend['year_added'] > 2010) &
    (region_trend['year_added'] < region_trend['year_added'].max())
]

fig_region_trend = px.line(
    region_trend,
    x='year_added',
    y='Count',
    color='region',
    markers=True,
    title='Content Growth Over Time by Region',
    color_discrete_sequence=px.colors.sequential.Reds
)
fig_region_trend.update_layout(
    title_x=0.5,
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    height=500
)


# --- production hubs ---

df_prod = df.copy()
df_prod['country'] = df_prod['country'].dropna().str.split(',')
df_prod = df_prod.explode('country')
df_prod['country'] = df_prod['country'].str.strip()

# Aggregate total titles per country
prod_counts = df_prod['country'].value_counts().reset_index()
prod_counts.columns = ['Country', 'Total Titles']

# Take top 15 production hubs
top_hubs = prod_counts.head(15)

# Visualization
fig_prod_hubs = px.bar(
    top_hubs[::-1],
    x='Total Titles',
    y='Country',
    orientation='h',
    color='Total Titles',
    color_continuous_scale='Reds',
    title='Top Production Hubs on Netflix'
)

fig_prod_hubs.update_layout(
    title={
        'x': 0.5,
        'xanchor': 'center',
        'font': {'size': 22, 'color': 'white'}
    },
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    yaxis=dict(autorange='reversed')
)

# --- market opportunities ---

# Clean up and explode by country
df_market = df.copy()
df_market['country'] = df_market['country'].dropna().str.split(',')
df_market = df_market.explode('country')
df_market['country'] = df_market['country'].str.strip()

# Convert 'date_added' to datetime
df_market['date_added'] = pd.to_datetime(df_market['date_added'], errors='coerce')
df_market['year_added'] = df_market['date_added'].dt.year

# Aggregate titles by country and year
country_year = (
    df_market.groupby(['country', 'year_added'])
    .size()
    .reset_index(name='Count')
)

# Focus on countries with valid recent data (after 2015)
country_year = country_year[country_year['year_added'] >= 2015]

# Compute growth rate (CAGR) for each country
growth_data = []
for country, group in country_year.groupby('country'):
    group = group.sort_values('year_added')
    if len(group) > 3 and group['Count'].iloc[0] > 0:
        start_val = group['Count'].iloc[0]
        end_val = group['Count'].iloc[-1]
        years = group['year_added'].iloc[-1] - group['year_added'].iloc[0]
        if years > 0:
            cagr = ((end_val / start_val) ** (1 / years) - 1) * 100
            growth_data.append((country, cagr, end_val))

df_growth = pd.DataFrame(growth_data, columns=['Country', 'CAGR (%)', 'Recent Titles'])

# Identify potential markets - moderate current titles but high growth
df_opportunity = df_growth[
    (df_growth['Recent Titles'] < df_growth['Recent Titles'].quantile(0.75)) &
    (df_growth['CAGR (%)'] > df_growth['CAGR (%)'].quantile(0.75))
].sort_values(by='CAGR (%)', ascending=False).head(15)

# Visualization
fig_market = px.scatter(
    df_growth,
    x='Recent Titles',
    y='CAGR (%)',
    text='Country',
    color='CAGR (%)',
    color_continuous_scale='Reds',
    title='Market Opportunities: Growth vs. Current Presence',
    hover_data={'Recent Titles': True, 'CAGR (%)': ':.2f', 'Country': True}
)

fig_market.update_traces(textposition='top center', marker=dict(size=10, line=dict(width=1, color='white')))

fig_market.update_layout(
    title={'x': 0.5, 'xanchor': 'center', 'font': {'size': 22, 'color': 'white'}},
    xaxis_title="Recent Titles (as of latest year)",
    yaxis_title="CAGR in Titles (%)",
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    height=600
)

# layout

def card_style():
    return {
        'flex': '1',
        'margin': '10px',
        'background': 'rgba(30,30,30,0.85)',
        'borderRadius': '16px',
        'padding': '30px',
        'boxShadow': '0 4px 30px rgba(0, 0, 0, 0.4)',
        'minWidth': '40%'
    }


layout = html.Div(
    children=[
        html.H1(
            [
                html.Span("G", style={'color': '#E50914'}),
                "eographic ",
                html.Span("I", style={'color': '#E50914'}),
                "nsights"
            ],
            style={
                'textAlign': 'center',
                'fontFamily': 'Segoe UI, sans-serif',
                'color': 'white',
                'fontWeight': '700',
                'fontSize': '2.5rem',
                'marginBottom': '10px',
                'marginTop': 0,
                'padding': 0
            }
        ),
        html.P(
            "Comparative overview of Netflix’s performance, content availability, and viewer preferences across different countries.",
            style={
                'textAlign': 'center',
                'color': '#ccc',
                'fontSize': '1.1rem',
                'marginBottom': '50px'
            }
        ),
        html.Div(
            [
                html.Div(
                    html.Div(dcc.Graph(figure=fig_world, config={'displayModeBar': False}), style=card_style())
                )
            ],
            style={'width': 'full'}
        ),
        # Country comparison table card
        html.Div(
            [
                html.Div(
                    [
                        html.H3("Country Comparison", style={'color': 'white', 'textAlign': 'center'}),

                        # Country selection dropdown
                        dcc.Dropdown(
                            id='country-dropdown',
                            options=[{'label': c, 'value': c} for c in sorted(world_counts['Country'].unique())],
                            value=['United States', 'India', 'Japan', 'South Korea', 'China'],  # default
                            multi=True,
                            placeholder="Select countries to compare...",
                            style={
                                'backgroundColor': '#1e1e1e',
                                'color': 'black',
                                'marginBottom': '20px'
                            }
                        ),

                        # Table
                        dash_table.DataTable(
                            id='country-comparison-table',
                            columns=[
                                {"name": "Country", "id": "Country"},
                                {"name": "Total Titles", "id": "Total Titles"},
                                {"name": "Movies", "id": "Movies"},
                                {"name": "TV Shows", "id": "TV Shows"},
                                {"name": "Top Genre", "id": "Top Genre"},
                                {"name": "Avg Release Year", "id": "Avg Release Year"}
                            ],
                            style_table={
                                'overflowX': 'auto',
                                'border': 'none',
                                'padding': '10px'
                            },
                            style_header={
                                'backgroundColor': '#E50914',
                                'color': 'white',
                                'fontWeight': 'bold',
                                'textAlign': 'center'
                            },
                            style_cell={
                                'backgroundColor': 'rgba(30,30,30,0.9)',
                                'color': 'white',
                                'textAlign': 'center',
                                'padding': '10px'
                            },
                            style_data_conditional=[
                                {'if': {'row_index': 'odd'}, 'backgroundColor': 'rgba(40,40,40,0.85)'}
                            ],
                            page_size=10
                        )
                    ],
                    style={
                        'flex': '1',
                        'margin': '10px',
                        'background': 'rgba(30,30,30,0.85)',
                        'borderRadius': '16px',
                        'padding': '30px',
                        'boxShadow': '0 4px 30px rgba(0, 0, 0, 0.4)',
                        'minWidth': '80%'
                    }
                )
            ],
            style={'display': 'flex', 'justifyContent': 'center'}
        ),

        html.Div(
            [
                html.Div(
                    html.P(
                        "Regional Deep dives",
                        style={
                            'flex': '1',
                            'margin': '10px',
                            'background': 'rgba(30,30,30,0.85)',
                            'backdropFilter': 'blur(8px)',
                            'borderRadius': '16px',
                            'padding': '30px',
                            'boxShadow': '0 4px 30px rgba(0, 0, 0, 0.4)',
                            'textAlign': 'center',
                            'font-size': '1.2rem',
                            'font-weight': 'bold'
                        }
                    ),
                )
            ],
            style={'width': 'full'}
        ),
        html.Div(
            [
                html.Div(
                    html.Div(dcc.Graph(figure=fig_heatmap, config={'displayModeBar': False}), style=card_style()),
                )
            ],
            style={'width': 'full'}
        ),
        html.Div(
            [
                html.Div(dcc.Graph(figure=fig_region_bar, config={'displayModeBar': False}), style=card_style()),
                html.Div(dcc.Graph(figure=fig_region_trend, config={'displayModeBar': False}), style=card_style()),
                html.Div(dcc.Graph(figure=fig_region_genre, config={'displayModeBar': False}), style=card_style()),
            ],
            style={
                'display': 'flex',
                'flexWrap': 'wrap',
                'justifyContent': 'center',
                'alignItems': 'stretch',
                'margin': 'auto'
            }
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H3(
                            "Production Hubs",
                            style={
                                'textAlign': 'center',
                                'color': 'white',
                                'fontWeight': 'bold',
                                'fontSize': '1.8rem',
                                'marginBottom': '10px'
                            }
                        ),
                        html.P(
                            "Production Hubs represent the leading countries driving Netflix’s global content library. "
                            "These are the regions where most shows and movies are produced, highlighting dominant and emerging centers of creation.",
                            style={
                                'textAlign': 'center',
                                'color': '#ccc',
                                'fontSize': '1.1rem',
                                'marginBottom': '30px',
                                'marginTop': '30px'
                            }
                        ),
                        html.Div(dcc.Graph(figure=fig_prod_hubs, config={'displayModeBar': False}), style=card_style()),
                    ],
                    style={
                        'flex': '1',
                        'margin': '10px',
                        'background': 'rgba(30,30,30,0.85)',
                        'borderRadius': '16px',
                        'padding': '30px',
                        'boxShadow': '0 4px 30px rgba(0, 0, 0, 0.4)',
                        'minWidth': '70%'
                    }
                )
            ],
            style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '40px'}
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.H3(
                            "Market Opportunities",
                            style={
                                'textAlign': 'center',
                                'color': 'white',
                                'fontWeight': 'bold',
                                'fontSize': '1.8rem',
                                'marginBottom': '10px'
                            }
                        ),
                        html.P(
                            "Market Opportunities highlight countries where Netflix’s content library is growing rapidly "
                            "despite a smaller existing base. These emerging regions indicate potential for future expansion "
                            "and audience engagement.",
                            style={
                                'textAlign': 'center',
                                'color': '#ccc',
                                'fontSize': '1.1rem',
                                'marginBottom': '30px',
                                'marginTop': '30px'
                            }
                        ),
                        html.Div(dcc.Graph(figure=fig_market, config={'displayModeBar': False}), style=card_style()),
                    ],
                    style={
                        'flex': '1',
                        'margin': '10px',
                        'background': 'rgba(30,30,30,0.85)',
                        'borderRadius': '16px',
                        'padding': '30px',
                        'boxShadow': '0 4px 30px rgba(0, 0, 0, 0.4)',
                        'minWidth': '70%'
                    }
                )
            ],
            style={'display': 'flex', 'justifyContent': 'center', 'marginBottom': '40px'}
        ),
    ],
    className='geo-insights',
    style={
        'backgroundColor': '#121212',
        'minHeight': '100vh',
        'padding': '60px 20px',
        'fontFamily': 'Segoe UI, sanautosize=Falses-serif',
        'overflow': 'scroll-y'
    }
)
def register_callbacks(app):
    @app.callback(
        Output('country-comparison-table', 'data'),
        Input('country-dropdown', 'value')
    )
    def update_comparison_table(selected_countries):
        if not selected_countries:
            return []

        # Split and explode multiple countries per title
        df_exp = df.copy()
        df_exp['country'] = df_exp['country'].str.split(',')
        df_exp = df_exp.explode('country')
        df_exp['country'] = df_exp['country'].str.strip()

        # Filter by selected countries
        df_selected = df_exp[df_exp['country'].isin(selected_countries)]

        if df_selected.empty:
            return []

        # Clean up genres for counting
        df_genres = df_selected.copy()
        df_genres['listed_in'] = df_genres['listed_in'].fillna('')
        df_genres['genre'] = df_genres['listed_in'].str.split(',')
        df_genres = df_genres.explode('genre')
        df_genres['genre'] = df_genres['genre'].str.strip()

        # Remove generic genres
        exclude_genres = ['Independent Movies', 'International Movies', 'International TV Shows']
        df_genres = df_genres[~df_genres['genre'].isin(exclude_genres)]

        # Find top genre per country
        top_genre = (
            df_genres.groupby(['country', 'genre'])
            .size()
            .reset_index(name='Count')
            .sort_values(['country', 'Count'], ascending=[True, False])
            .drop_duplicates(subset='country', keep='first')
            [['country', 'genre']]
        )
        top_genre.rename(columns={'genre': 'Top Genre'}, inplace=True)

        # Compute main metrics
        comparison = df_selected.groupby('country').agg(
            **{
                'Total Titles': ('title', 'count'),
                'Movies': ('type', lambda x: (x == 'Movie').sum()),
                'TV Shows': ('type', lambda x: (x == 'TV Show').sum()),
                'Avg Release Year': ('release_year', 'mean')
            }
        ).reset_index()

        comparison.rename(columns={'country': 'Country'}, inplace=True)
        comparison['Avg Release Year'] = comparison['Avg Release Year'].round(1)

        # Merge top genre
        comparison = comparison.merge(top_genre, left_on='Country', right_on='country', how='left').drop(columns=['country'])

        #Sort for clean presentation
        comparison = comparison.sort_values(by='Total Titles', ascending=False)

        return comparison.to_dict('records')