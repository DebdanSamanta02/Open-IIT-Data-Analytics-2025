from dash import html, dcc, Input, Output, callback
import plotly.express as px
import pandas as pd

# --- Load & Prepare Data ---
df = pd.read_csv('./data/netflix_titles.csv')

df['country'] = df['country'].fillna('Unknown')
df['listed_in'] = df['listed_in'].fillna('Unknown')
df['rating'] = df['rating'].fillna('Not Rated')
df['release_year'] = df['release_year'].fillna(0).astype(int)

# Explode genres
df['genre_list'] = df['listed_in'].apply(lambda x: [g.strip() for g in str(x).split(',')])
df_exploded = df.explode('genre_list').rename(columns={'genre_list': 'genre'})
df_exploded['genre'] = df_exploded['genre'].str.strip()
df_exploded = df_exploded[df_exploded['genre'] != 'Unknown']

# Filter options
unique_types = sorted(df['type'].unique())
unique_countries = sorted(set(c.strip() for lst in df['country'].dropna().str.split(',') for c in lst if c.strip() != 'Unknown'))
unique_ratings = sorted([r for r in df['rating'].unique() if r != 'Not Rated'])

# --- LAYOUT ---
layout = html.Div([
    html.H1(
        [
            html.Span("G", style={'color': '#E50914'}),
            "enre ",
            html.Span("I", style={'color': '#E50914'}),
            "ntelligence"
        ],
        style={
            'textAlign': 'center',
            'fontFamily': 'Segoe UI, sans-serif',
            'color': 'white',
            'fontWeight': '700',
            'fontSize': '2.5rem',
            'marginBottom': '10px',
            'marginTop': 0,
        }
    ),
    html.P(
        "Explore genre trends, category relationships, and identify strategic content opportunities.",
        style={'textAlign': 'center', 'color': '#ccc', 'fontSize': '1.1rem', 'marginBottom': '30px'}
    ),

    # --- Filter row ---
    html.Div([
        dcc.Dropdown(
            id='genre-type',
            options=[{'label': 'All Types', 'value': 'all'}] + [{'label': t, 'value': t} for t in unique_types],
            value='all',
            placeholder='Type',
            className='filter-dropdown',
            style={
                'backgroundColor': '#1b1b1b',
                'color': 'white',
            }
        ),
        dcc.Dropdown(
            id='genre-country',
            options=[{'label': 'All Countries', 'value': 'all'}] + [{'label': c, 'value': c} for c in unique_countries],
            value='all',
            placeholder='Country',
            className='filter-dropdown',
            style={
                'backgroundColor': '#1b1b1b',
                'color': 'white',
            }
        ),
        dcc.Dropdown(
            id='genre-rating',
            options=[{'label': 'All Ratings', 'value': 'all'}] + [{'label': r, 'value': r} for r in unique_ratings],
            value='all',
            placeholder='Rating',
            className='filter-dropdown',
            style={
                'backgroundColor': '#1b1b1b',
                'color': 'white',
            }
        ),
        # Year range slider (matching content explorer)
        html.Div([
            html.Label('Release Year Range:', style={'color': '#ccc', 'marginBottom': '8px', 'fontSize': '0.9rem'}),
            dcc.RangeSlider(
                id='genre-year',
                min=int(df['release_year'].min()),
                max=int(df['release_year'].max()),
                value=[int(df['release_year'].min()), int(df['release_year'].max())],
                marks={year: str(year) for year in range(int(df['release_year'].min()), int(df['release_year'].max())+1, 10)},
                tooltip={"placement": "bottom", "always_visible": False},
                className='year-slider',
                updatemode='drag'
            ),
        ], style={'gridColumn': '1 / -1', 'marginTop': '15px'}),
    ], style={
        'display': 'grid',
        'gridTemplateColumns': '1fr 1fr 1fr',
        'gap': '12px',
        'alignItems': 'start',
        'width': '95%',
        'margin': 'auto',
        'marginBottom': '25px'
    }),

    # --- Stats cards ---
    html.Div(id='genre-stats', style={
        'display': 'flex',
        'justifyContent': 'space-evenly',
        'flexWrap': 'wrap',
        'gap': '25px',
        'width': '95%',
        'margin': '0 auto 40px auto',
        'padding': '0',
        'boxSizing': 'border-box'
    }),

    # --- Charts full width ---
    html.Div([
        dcc.Graph(id='genre-top-chart', style={'height': '480px', 'width': '100%', 'display': 'block'}),
        dcc.Graph(id='genre-trend-chart', style={'height': '480px', 'width': '100%', 'display': 'block'})
    ], style={'width': '95%', 'margin': '0 auto', 'padding': '0', 'boxSizing': 'border-box'})
], style={
    'backgroundColor': '#121212',
    'minHeight': '100vh',
    'padding': '50px 10px',
    'overflowX': 'hidden'
})

# --- CALLBACK ---
@callback(
    Output('genre-top-chart', 'figure'),
    Output('genre-trend-chart', 'figure'),
    Output('genre-stats', 'children'),
    Input('genre-type', 'value'),
    Input('genre-country', 'value'),
    Input('genre-rating', 'value'),
    Input('genre-year', 'value')
)
def update_genre_tab(type_, country, rating, year_range):
    filtered = df_exploded.copy()

    # Apply filters
    if type_ != 'all':
        filtered = filtered[filtered['type'] == type_]
    if country != 'all':
        filtered = filtered[filtered['country'].str.contains(country, case=False, na=False)]
    if rating != 'all':
        filtered = filtered[filtered['rating'] == rating]
    filtered = filtered[(filtered['release_year'] >= year_range[0]) & (filtered['release_year'] <= year_range[1])]

    # --- Top genres ---
    genre_counts = filtered['genre'].value_counts().nlargest(10).reset_index()
    genre_counts.columns = ['Genre', 'Count']
    fig_top = px.bar(
        genre_counts, y='Genre', x='Count', orientation='h',
        color='Count', color_continuous_scale=['#440000', '#E50914'],
        title='Top 10 Genres'
    )
    fig_top.update_layout(
        plot_bgcolor='#121212',
        paper_bgcolor='#121212',
        font_color='white',
        title_font_color='#E50914',
        yaxis={'categoryorder': 'total ascending'},
        autosize=True,
        margin=dict(l=10, r=10, t=50, b=10)
    )

    # --- Genre trends ---
    top_list = genre_counts['Genre'].tolist()
    trend_df = filtered[filtered['genre'].isin(top_list)]
    trend_df = trend_df.groupby(['release_year', 'genre']).size().reset_index(name='count')
    fig_trend = px.line(
        trend_df, x='release_year', y='count', color='genre',
        markers=True, title='Genre Popularity Over Time'
    )
    fig_trend.update_layout(
        plot_bgcolor='#121212',
        paper_bgcolor='#121212',
        font_color='white',
        title_font_color='#E50914',
        legend_title_text='Genre',
        autosize=True,
        margin=dict(l=10, r=10, t=50, b=10)
    )

    # --- Stats cards ---
    total_genres = filtered['genre'].nunique()
    top_genre = genre_counts.iloc[0]['Genre'] if not genre_counts.empty else 'N/A'
    avg_year = int(filtered['release_year'].mean()) if not filtered.empty else 0

    card_style = {
        'textAlign': 'center',
        'padding': '20px',
        'backgroundColor': '#1b1b1b',
        'borderRadius': '12px',
        'minWidth': '200px',
        'flex': '1',
        'boxShadow': '0 0 10px rgba(229,9,20,0.3)'
    }

    cards = [
        html.Div([
            html.H3(str(total_genres), style={'color': '#E50914', 'margin': 0}),
            html.P("Unique Genres", style={'color': '#ccc', 'margin': 0})
        ], style=card_style),
        html.Div([
            html.H3(top_genre, style={'color': '#E50914', 'margin': 0}),
            html.P("Top Genre", style={'color': '#ccc', 'margin': 0})
        ], style=card_style),
        html.Div([
            html.H3(str(avg_year), style={'color': '#E50914', 'margin': 0}),
            html.P("Avg Release Year", style={'color': '#ccc', 'margin': 0})
        ], style=card_style)
    ]

    return fig_top, fig_trend, cards
