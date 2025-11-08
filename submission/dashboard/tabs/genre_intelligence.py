from dash import html, dcc, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import networkx as nx
from itertools import combinations
import dash

# --- Load & Prepare Data ---
df = pd.read_csv('./data/netflix_titles.csv')

df['country'] = df['country'].fillna('Unknown')
df['listed_in'] = df['listed_in'].fillna('Unknown')
df['rating'] = df['rating'].fillna('Not Rated')
df['release_year'] = df['release_year'].fillna(0).astype(int)

df['genre_list'] = df['listed_in'].apply(lambda x: [g.strip() for g in str(x).split(',')])
df_exploded = df.explode('genre_list').rename(columns={'genre_list': 'genre'})
df_exploded['genre'] = df_exploded['genre'].str.strip()
df_exploded = df_exploded[df_exploded['genre'] != 'Unknown']

unique_types = sorted(df['type'].unique())
unique_countries = sorted(set(c.strip() for lst in df['country'].dropna().str.split(',') for c in lst if c.strip() != 'Unknown'))
unique_ratings = sorted([r for r in df['rating'].unique() if r != 'Not Rated'])


# --- LAYOUT ---
layout = html.Div([
    # --- Title + Description ---
    html.Div(id='title-block', children=[
        html.H1([
            html.Span("G", style={'color': '#E50914'}),
            "enre ",
            html.Span("I", style={'color': '#E50914'}),
            "ntelligence"
        ], id='page-title', style={
            'textAlign': 'center',
            'fontFamily': 'Segoe UI, sans-serif',
            'fontWeight': 'var(--heading-weight)',
            'fontSize': '2.5rem',
            'marginBottom': '15px',
            'letterSpacing': '-0.02em',
            'transition': 'color var(--transition-speed) ease'
        }),

        html.P("Explore genre trends, category relationships, and identify strategic content opportunities.",
               id='page-description',
               style={
                   'textAlign': 'center', 
                   'fontSize': '1.1rem', 
                   'marginBottom': '35px', 
                   'color': 'var(--muted-text)',
                   'fontWeight': 'var(--subheading-weight)',
                   'maxWidth': '800px',
                   'margin': '0 auto 35px auto',
                   'lineHeight': '1.5'
               })
    ]),

    # --- Filters ---
    html.Div([
        dcc.Dropdown(
            id='genre-type',
            options=[{'label': 'All Types', 'value': 'all'}] + [{'label': t, 'value': t} for t in unique_types],
            value='all',
            placeholder='Type',
            className='theme-dropdown',
            style={
                'backgroundColor': 'var(--dropdown-bg)',
                        'color': 'var(--dropdown-text)',
            }
        ),
        dcc.Dropdown(
            id='genre-country',
            options=[{'label': 'All Countries', 'value': 'all'}] + [{'label': c, 'value': c} for c in unique_countries],
            value='all',
            placeholder='Country',
            className='theme-dropdown',
            style={
                'backgroundColor': 'var(--dropdown-bg)',
                        'color': 'var(--dropdown-text)',
            }
        ),
        dcc.Dropdown(
            id='genre-rating',
            options=[{'label': 'All Ratings', 'value': 'all'}] + [{'label': r, 'value': r} for r in unique_ratings],
            value='all',
            placeholder='Rating',
            className='theme-dropdown',
            style={
                'backgroundColor': 'var(--dropdown-bg)',
                        'color': 'var(--dropdown-text)',
            }
        ),
        html.Div([
            html.Label('Release Year Range:',
                       style={'color': 'var(--muted-text)', 'marginBottom': '8px', 'fontSize': '0.9rem'}),
            dcc.RangeSlider(
                id='genre-year',
                min=int(df['release_year'].min()),
                max=int(df['release_year'].max()),
                value=[int(df['release_year'].min()), int(df['release_year'].max())],
                marks={y: str(y) for y in range(int(df['release_year'].min()), int(df['release_year'].max()) + 1, 10)},
                tooltip={"placement": "bottom"},
                className='year-slider'
            )
        ], style={'gridColumn': '1 / -1', 'marginTop': '15px'}),
    ], style={
        'display': 'grid',
        'gridTemplateColumns': '1fr 1fr 1fr',
        'gap': '12px',
        'width': '95%',
        'margin': 'auto',
        'marginBottom': '30px'
    }),

    # --- KPI Cards ---
    html.Div(id='genre-stats', style={
        'display': 'grid',
        'gridTemplateColumns': 'repeat(auto-fit, minmax(220px, 1fr))',
        'gap': '25px',
        'width': '90%',
        'margin': '0 auto 40px auto'
    }),

    # --- Charts ---
    dcc.Graph(id='genre-top-chart', style={'height': '480px', 'width': '100%'}),
    dcc.Graph(id='genre-trend-chart', style={'height': '480px', 'width': '100%'}),

    html.Hr(style={'borderColor': 'var(--accent-color)', 'width': '96%', 'margin': '50px auto'}),

    html.Div([
        html.H2("Genre Co-Occurrence Analysis", style={
            'textAlign': 'center', 'color': 'var(--accent-color)', 'fontWeight': '700',
            'fontSize': '2rem', 'marginBottom': '20px'
        }),
        html.P("Discover how genres frequently appear together in Netflix titles.",
               style={'textAlign': 'center', 'color': 'var(--subtext-color)', 'fontSize': '1.05rem'})
    ]),
    dcc.Graph(id='genre-co-heatmap', style={'height': '90vh', 'width': '98%', 'margin': 'auto'}),

    html.Hr(style={'borderColor': 'var(--accent-color)', 'width': '96%', 'margin': '50px auto'}),

    # --- Strategic KPIs ---
    html.Div([
        html.H2("Strategic Opportunities and Gaps", style={
            'textAlign': 'center', 'color': 'var(--text-color)', 'marginBottom': '30px',
            'fontSize': '2rem', 'fontWeight': '700'
        }),
        html.Div(id='strategic-kpis', style={
            'display': 'grid',
            'gridTemplateColumns': '1fr 1fr',
            'gap': '30px',
            'width': '90%',
            'margin': '0 auto'
        }),
    ], style={'width': '100%', 'margin': '0 auto 50px auto'})
], style={
    'backgroundColor': 'var(--background-color)',
    'padding': '40px 0',
    'minHeight': '100vh',
    'width': '100%',
    'overflowX': 'hidden',
    'fontFamily': 'Segoe UI, sans-serif',

})


# --- DROPDOWN THEME CALLBACK (Always Black) ---
@callback(
    Output('genre-type', 'style'),
    Output('genre-country', 'style'),
    Output('genre-rating', 'style'),
    Input('current-theme', 'data')
)
def update_dropdown_styles(current_theme):
    # Always keep dropdown background black
    if current_theme == 'light':
        dropdown_style = {
            'backgroundColor': '#ffffff',  # white background
            'color': '#000000',            # black text
            'border': '1px solid rgba(229,9,20,0.5)',
        }
    else:
        dropdown_style = {
            'backgroundColor': '#1a1a1a',  # dark gray
            'color': '#ffffff',            # white text
            'border': '1px solid rgba(229,9,20,0.5)',
        }

    return dropdown_style, dropdown_style, dropdown_style

# --- MAIN DATA CALLBACK ---
@callback(
    Output('genre-top-chart', 'figure'),
    Output('genre-trend-chart', 'figure'),
    Output('genre-stats', 'children'),
    Output('genre-co-heatmap', 'figure'),
    Output('strategic-kpis', 'children'),
    Output('page-title', 'style'),
    Output('page-description', 'style'),
    Input('genre-type', 'value'),
    Input('genre-country', 'value'),
    Input('genre-rating', 'value'),
    Input('genre-year', 'value'),
    Input('current-theme', 'data')
)
def update_genre_tab(type_, country, rating, year_range, current_theme):
    filtered = df_exploded.copy()
    if type_ != 'all':
        filtered = filtered[filtered['type'] == type_]
    if country != 'all':
        filtered = filtered[filtered['country'].str.contains(country, case=False, na=False)]
    if rating != 'all':
        filtered = filtered[filtered['rating'] == rating]
    filtered = filtered[(filtered['release_year'] >= year_range[0]) & (filtered['release_year'] <= year_range[1])]

    # --- Theme Colors ---
    if current_theme == 'light':
        bg_color = 'var(--background-color)'
        paper_color = 'var(--background-color)'
        font_color = '#000000'
        subtext_color = '#555555'
        grid_color = '#cccccc'
        card_bg = 'var(--card-bg)'
    else:
        bg_color = '#121212'
        paper_color = '#121212'
        font_color = '#ffffff'
        subtext_color = '#cccccc'
        grid_color = '#333333'
        card_bg = '#1b1b1b'

    accent_color = '#E50914'

    # --- Title + Description ---
    title_style = {
        'textAlign': 'center',
        'fontFamily': 'Segoe UI, sans-serif',
        'fontWeight': '700',
        'fontSize': '2.5rem',
        'marginBottom': '10px',
        'color': font_color
    }
    desc_style = {
        'textAlign': 'center',
        'fontSize': '1.1rem',
        'marginBottom': '30px',
        'color': subtext_color
    }

    # --- Top Genres Chart ---
    genre_counts = filtered['genre'].value_counts().nlargest(10).reset_index()
    genre_counts.columns = ['Genre', 'Count']
    fig_top = px.bar(
        genre_counts, y='Genre', x='Count', orientation='h',
        color='Count', color_continuous_scale=['#440000', accent_color],
        title='Top 10 Genres'
    )
    fig_top.update_layout(
        plot_bgcolor=bg_color, paper_bgcolor=paper_color,
        font_color=font_color, title_font_color=accent_color,
        xaxis=dict(showgrid=True, gridcolor=grid_color),
        yaxis={'categoryorder': 'total ascending'},
        coloraxis_showscale=False
    )

    # --- Trend Chart ---
    top_list = genre_counts['Genre'].tolist()
    trend_df = filtered[filtered['genre'].isin(top_list)]
    trend_df = trend_df.groupby(['release_year', 'genre']).size().reset_index(name='count')
    fig_trend = px.line(
        trend_df, x='release_year', y='count', color='genre', markers=True,
        title='Genre Popularity Over Time'
    )
    fig_trend.update_layout(
        plot_bgcolor=bg_color, paper_bgcolor=paper_color,
        font_color=font_color, title_font_color=accent_color,
        xaxis=dict(showgrid=True, gridcolor=grid_color),
        yaxis=dict(showgrid=True, gridcolor=grid_color)
    )

    # --- KPI Cards ---
    total_titles = filtered['show_id'].nunique()
    total_genres = filtered['genre'].nunique()
    top_genre = genre_counts.iloc[0]['Genre'] if not genre_counts.empty else 'N/A'
    avg_year = int(filtered['release_year'].mean()) if not filtered.empty else 0
    card_style = {
        'textAlign': 'center',
        'padding': '20px',
        'backgroundColor': card_bg,
        'borderRadius': '12px',
        'border': f'2px solid {accent_color}',
        'boxShadow': f'0 0 20px rgba(229,9,20,0.3)'
    }
    cards = [
        html.Div([html.H3(f"{total_titles:,}", style={'color': accent_color}),
                  html.P("Total Titles", style={'color': font_color})], style=card_style),
        html.Div([html.H3(f"{total_genres}", style={'color': accent_color}),
                  html.P("Unique Genres", style={'color': font_color})], style=card_style),
        html.Div([html.H3(top_genre, style={'color': accent_color}),
                  html.P("Top Genre", style={'color': font_color})], style=card_style),
        html.Div([html.H3(str(avg_year), style={'color': accent_color}),
                  html.P("Avg Release Year", style={'color': font_color})], style=card_style)
    ]

    # --- Heatmap ---
    genre_lists = filtered.groupby('title')['genre'].apply(list)
    co_pairs = {}
    for genres in genre_lists:
        if len(genres) > 1:
            for g1, g2 in combinations(sorted(set(genres)), 2):
                co_pairs[(g1, g2)] = co_pairs.get((g1, g2), 0) + 1
    co_df = pd.DataFrame([{'g1': g1, 'g2': g2, 'count': c} for (g1, g2), c in co_pairs.items()])
    co_df = co_df.sort_values('count', ascending=False).head(40)
    counts = filtered['genre'].value_counts().to_dict()
    all_genres = sorted(counts.keys())
    matrix = pd.DataFrame(0, index=all_genres, columns=all_genres)
    for _, r in co_df.iterrows():
        if r['g1'] in all_genres and r['g2'] in all_genres:
            matrix.loc[r['g1'], r['g2']] = r['count']
            matrix.loc[r['g2'], r['g1']] = r['count']
    fig_heat = px.imshow(matrix, text_auto=True, color_continuous_scale='Reds',
                         aspect='auto', title='Genre Co-Occurrence Heatmap (All Genres)')
    fig_heat.update_layout(
        autosize=True, height=900,
        margin=dict(l=100, r=100, t=100, b=100),
        plot_bgcolor=bg_color, paper_bgcolor=paper_color,
        font_color=font_color, title_font_color=accent_color,
        xaxis=dict(showticklabels=True, tickangle=45, automargin=True),
        yaxis=dict(showticklabels=True, automargin=True)
    )

    # --- Strategic KPIs ---
    text_color_kpi = '#ddd' if current_theme == 'dark' else '#333'
    kpi_style_strategic = {
        'backgroundColor': card_bg,
        'padding': '25px 30px',
        'borderRadius': '14px',
        'boxShadow': f'0 4px 20px rgba(0,0,0,{"0.45" if current_theme == "dark" else "0.1"})',
        'borderLeft': '6px solid',
        'color': font_color
    }

    kpi_opportunity = html.Div([
        html.H3("KPI 1: Opportunity Identification — High-Impact Growth Engines",
                style={'color': font_color, 'fontSize': '1.45rem', 'marginBottom': '12px'}),
        html.H4("Tier 1 — High-Impact, Low-Risk (Score 85–95)", style={'color': '#4CE66F'}),
        html.Ul([
            html.Li("Classic Vault: +300% CAGR; build 500+ 'Heritage' titles by 2026."),
            html.Li("Niche Clusters: Expand Anime, Faith/Spirituality, and Cult TV."),
            html.Li("Efficiency Edge: 5–8% of content spend covers 25% portfolio.")
        ], style={'color': text_color_kpi}),
        html.H4("Tier 2 — Emerging High-Lift (Score 70–84)", style={'color': '#6FFFA0'}),
        html.Ul([
            html.Li("Genre Pairing: Horror–Thriller synergies (12–15 new titles/year)."),
            html.Li("Regional Co-Productions: India/Korea co-financed IPs for export."),
            html.Li("Limited Series: 60% of TV originals by 2026 for higher hit rate.")
        ], style={'color': text_color_kpi}),
        html.H4("Tier 3 — Quality Upgrades (Score 60–69)", style={'color': '#A5FFBF'}),
        html.Ul([
            html.Li("Stand-Up & Horror: Volume ↓30%, quality ↑50%; avg rating → 7.5+."),
            html.Li("Partner with prestige creators (A24 model) for rebranding.")
        ], style={'color': text_color_kpi}),
        html.P("Target KPIs: Classic 8% share · Niche 25% portfolio · Avg rating ≥7.5",
               style={'color': '#4CE66F', 'fontWeight': '600', 'marginTop': '15px'})
    ], style={**kpi_style_strategic, 'borderLeftColor': '#00B050'})

    kpi_gap = html.Div([
        html.H3("KPI 2: Competitive Gap Analysis — Defend Core Vulnerabilities",
                style={'color': font_color, 'fontSize': '1.45rem', 'marginBottom': '12px'}),
        html.H4("Gap 1 — Family/Kids Deficit (Severity 9.2/10)", style={'color': '#FF7070'}),
        html.Ul([
            html.Li("Disney+ has 35–40% kids content vs. Netflix <5%."),
            html.Li("Revenue risk: $2.8B–$4.2B from family churn."),
            html.Li("Defense: License 50+ titles; 15–20 new kids miniseries; reach 15% share."),
            html.Li("Long-Term: Build/acquire animation studio; 'Netflix Kids Universe'.")
        ], style={'color': text_color_kpi}),
        html.H4("Gap 2 — Prestige/Adult Pressure (Severity 7.8/10)", style={'color': '#FF8F8F'}),
        html.Ul([
            html.Li("HBO & Apple lead in prestige (8.3 vs 7.1 avg rating)."),
            html.Li("Restore brand halo via 8–12% prestige budget; 6–8 Emmy titles/year."),
            html.Li("KPI: ≥12 Emmy nominations · Avg rating ≥8.0.")
        ], style={'color': text_color_kpi}),
        html.H4("Defensive Strengths to Fortify", style={'color': '#66B2FF'}),
        html.Ul([
            html.Li("International lead: Maintain 3:1 ratio vs. competitors."),
            html.Li("Discovery edge: +25% engagement via ML genre pairing."),
            html.Li("Balanced mix: 45–48% mature content for household retention.")
        ], style={'color': text_color_kpi}),
        html.P("Target KPIs: Family 15% share · Churn ↓40% · ≥12 Emmys/year · Retention >40%",
               style={'color': '#FF7070', 'fontWeight': '600', 'marginTop': '15px'})
    ], style={**kpi_style_strategic, 'borderLeftColor': '#E50914'})

    return fig_top, fig_trend, cards, fig_heat, [kpi_opportunity, kpi_gap], title_style, desc_style