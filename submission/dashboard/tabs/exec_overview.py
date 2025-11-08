from dash import html, dcc, callback, Output, Input
import pandas as pd
import plotly.express as px
import numpy as np

# Loadind and Processing data
df = pd.read_csv('./data/netflix_titles.csv')

# Type data
type_counts = df['type'].value_counts().reset_index()
type_counts.columns = ['Type', 'Count']

# year data
year_counts = df['release_year'].value_counts().sort_index().reset_index()
year_counts.columns = ['Release Year', 'Count']

df['date_added'] = pd.to_datetime(df['date_added'].str.strip(), format='%B %d, %Y', errors='coerce')
df['year_added'] = df['date_added'].dt.year
content_trends = df.groupby(['year_added', 'type']).size().reset_index(name='Count')
content_trends = content_trends[
    (content_trends['year_added'] > 2010) &
    (content_trends['year_added'] < content_trends['year_added'].max())
]


# country data
country_list = df["country"].str.split(",").explode().str.strip()  # parsing multiple countries and adding them to the count
country_counts = country_list.value_counts().head(20)  # top 20 countries
country_normalized_counts = country_list.value_counts(normalize=True)
shannon_index = -np.sum(country_normalized_counts * np.log2(country_normalized_counts))  # Shannon diversity index

titles = df['title'].count()
countries = country_list.nunique()
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

# Convert to DataFrame
lang_df = pd.DataFrame(list(language_counts.items()), columns=['Language', 'Count']).sort_values(by='Count', ascending=False)

# ---- PIE CHART ----
fig_pie = px.pie(
    type_counts,
    names='Type',
    values='Count',
    color='Type',
    color_discrete_sequence=['#E50914', '#B20710'],  # Netflix reds
    hole=0.45,
)
fig_pie.update_layout(
    title=dict(
        text='Movies vs TV Shows',
        font=dict(size=18, color='white'),
        x=0.5,
        xanchor='center'
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
)
fig_pie.update_traces(
    textinfo='percent+label',
    textfont_size=14,
    marker=dict(line=dict(color='#121212', width=2)),
    hovertemplate='<b>%{label}</b><br>%{value} Titles<br>%{percent}',
)

# ---- BAR CHART (Years) ----
fig_bar = px.bar(
    year_counts.tail(30),  # Show last 30 years for clarity
    x='Release Year',
    y='Count',
    text='Count',
    color='Count',
    color_continuous_scale=['#B20710', '#E50914']
)
fig_bar.update_layout(
    title=dict(
        text='Content Releases Over the Years',
        font=dict(size=18, color='white'),
        x=0.5
    ),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    xaxis=dict(showgrid=False, tickfont=dict(size=11)),
    yaxis=dict(showgrid=False, tickfont=dict(size=11)),
    coloraxis_showscale=False,
)
fig_bar.update_traces(
    textposition='outside',
    marker_line_color='#121212',
    marker_line_width=1.2,
    hovertemplate='<b>%{x}</b><br>%{y} Titles'
)

# ---- CATEGORY HISTOGRAM ----
rating_groups = {
    'Kids': ['TV-Y', 'TV-Y7', 'TV-G', 'G', 'TV-Y7-FV'],
    'Teens': ['TV-PG', 'PG', 'PG-13', 'TV-14'],
    'Adults': ['TV-MA', 'R', 'NC-17', 'NR', 'UR']
}
def cat(x):
    for k, v in rating_groups.items():
        if x['rating'] in v:
            return k
df['category'] = df.apply(lambda x: cat(x), axis=1)

fig_hist = px.histogram(
    df,
    x='category',
    color='type',
    barmode='group',
    color_discrete_sequence=['#E50914', "#B0262D"]
)
fig_hist.update_layout(
    title=dict(
        text='Rates of Category by Type',
        font=dict(size=18, color='white'),
        x=0.5
    ),
    xaxis=dict(
        title='Category',
        tickangle=45,
        tickfont=dict(size=11, color='white')
    ),
    yaxis=dict(
        title='Count',
        tickfont=dict(size=11, color='white')
    ),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
    legend=dict(title='Type')
)

# ---- Top Countries ----
fig_country_bar = px.bar(
    country_counts,
    x=country_counts.values[::-1],
    y=country_counts.index[::-1],
    color='count',
    color_continuous_scale=['#B20710', '#E50914']
)
fig_country_bar.update_layout(
    title=dict(
        text='Top 20 countries ',
        font=dict(size=18, color='white'),
        x=0.5
    ),
    xaxis=dict(
        title='Country',
        tickangle=0,
        tickfont=dict(size=11, color='white'),
        showgrid=False
    ),
    yaxis=dict(
        title='Count',
        tickfont=dict(size=11, color='white'),
        showgrid=False
    ),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
)

# ---- Language Bar Graph ----
fig_lang_bar = px.bar(
    lang_df,
    x=lang_df['Language'],
    y=lang_df['Count'],
    color='Count',
     color_continuous_scale=['#B20710', "#EB1D27"]
)
fig_lang_bar.update_layout(
    title=dict(
        text='Language Based Content Division',
        font=dict(size=18, color='white'),
        x=0.5
    ),
    xaxis=dict(
        title='Language',
        tickangle=0,
        tickfont=dict(size=11, color='white'),
    ),
    yaxis=dict(
        title='Count',
        tickfont=dict(size=11, color='white'),
        showgrid=False
    ),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
)

# --- Content Volume Trends Over Time ---

fig_line = px.line(
    content_trends,
    x='year_added',
    y='Count',
    color='type',
    title=' Content Volume Added to Netflix (2011-Present)',
    markers=True,
    color_discrete_map={'Movie': '#E50914', 'TV Show': '#B00000'}
)
fig_line.update_layout(
    title=dict(
        text='Content Volume over the year',
        font=dict(size=18, color='white'),
        x=0.5
    ),
    xaxis=dict(
        title='Volume',
        tickangle=0,
        tickfont=dict(size=11, color='white'),
        showgrid=False
    ),
    yaxis=dict(
        title='Count',
        tickfont=dict(size=11, color='white'),
    ),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
)

# --- Genre Distribution & Popularity ---
genre_df = df.copy()
genre_df = genre_df.set_index('title')['listed_in'].str.split(', ', expand=True).stack()
genre_df = genre_df.reset_index(level=1, drop=True).value_counts().reset_index()
genre_df.columns = ['Genre', 'Count']
genres = genre_df['Genre'].count()

fig_genre_bar = px.bar(
    genre_df.head(20),
    x='Count',
    y='Genre',
    orientation='h',
    title=' Top 20 Genres on Netflix',
    color='Count',
    color_continuous_scale='Reds'
)
fig_genre_bar.update_layout(
    title=dict(
        text='Content Volume over the year',
        font=dict(size=18, color='white'),
        x=0.5
    ),
    xaxis=dict(
        title='Volume',
        tickangle=0,
        tickfont=dict(size=11, color='white'),
        showgrid=False
    ),
    yaxis=dict(
        title='Count',
        tickfont=dict(size=11, color='white'),
        autorange='reversed'
    ),
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='white'),
)

# ---- LAYOUT ----
layout = html.Div(
    children=[
        html.H1(
            [
                html.Span("E", style={'color': '#E50914'}),
                "xecutive ",
                html.Span("O", style={'color': '#E50914'}),
                "verview"
            ],
            style={
                'textAlign': 'center',
                'fontFamily': 'Segoe UI, sans-serif',
                'color': 'var(--font-color)',
                'fontWeight': '700',
                'fontSize': '2.5rem',
                'marginBottom': '10px',
                'marginTop': 0,
                'padding': 0
            }
        ),
        html.P(
            "High-level summary of Netflix content trends and release evolution.",
            style={
                'textAlign': 'center',
                'fontSize': '1.1rem',
                'marginBottom': '50px',
                'color': 'var(--font-color)'
            }
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.P(
                            "KPIs (Key Performance Indicators)",
                            style={
                                'textAlign': 'center',
                                'font-size': '1.3rem',
                                'font-weight': 'bold',
                                'marginBottom': '20px'
                            }
                        ),

                        # KPI Tiles Container
                        html.Div(
                            [
                                # --- Titles ---
                                html.Div(
                                    [
                                        html.P(f'{titles}', style={
                                            'textAlign': 'center',
                                            'font-size': '2rem',
                                            'margin': '0'
                                        }),
                                        html.P('Titles', style={
                                            'textAlign': 'center',
                                            'margin': '0 0 10px 0'
                                        }),
                                        dcc.Graph(
                                            figure=px.area(
                                                year_counts,
                                                x='Release Year', y='Count',
                                                height=100, width=200,
                                                color_discrete_sequence=['#E50914']
                                            ).update_layout(
                                                margin=dict(l=0, r=0, t=0, b=0),
                                                xaxis_visible=False, yaxis_visible=False,
                                                paper_bgcolor='rgba(0,0,0,0)',
                                                plot_bgcolor='rgba(0,0,0,0)'
                                            ).update_traces(mode='lines', fill='tozeroy'),
                                            config={'displayModeBar': False},
                                            style={'marginTop': '5px'}
                                        )
                                    ],
                                    style={
                                        'flex': '1 1 220px',
                                        'maxWidth': '250px',
                                        'minWidth': '220px',
                                        'background': 'var(--graph-color)',
                                        'borderRadius': '16px',
                                        'padding': '20px',
                                        'margin': '10px',
                                        'textAlign': 'center',
                                        'boxShadow': '0 4px 20px rgba(0, 0, 0, 0.3)'
                                    }
                                ),

                                # --- CAGR ---
                                html.Div(
                                    [
                                        html.P(f'{CAGR:.2f}%', style={
                                            'textAlign': 'center',
                                            'font-size': '2rem',
                                            'margin': '0'
                                        }),
                                        html.P('CAGR (Growth Rate)', style={
                                            'textAlign': 'center',
                                            'margin': '0 0 10px 0'
                                        }),
                                        dcc.Graph(
                                            figure=px.area(
                                                year_counts,
                                                x='Release Year', y='Count',
                                                height=100, width=200,
                                                color_discrete_sequence=['#B20710']
                                            ).update_layout(
                                                margin=dict(l=0, r=0, t=0, b=0),
                                                xaxis_visible=False, yaxis_visible=False,
                                                paper_bgcolor='rgba(0,0,0,0)',
                                                plot_bgcolor='rgba(0,0,0,0)'
                                            ).update_traces(mode='lines', fill='tozeroy'),
                                            config={'displayModeBar': False},
                                            style={'marginTop': '5px'}
                                        )
                                    ],
                                    style={
                                        'flex': '1 1 220px',
                                        'maxWidth': '250px',
                                        'minWidth': '220px',
                                        'background': 'var(--graph-color)',
                                        'borderRadius': '16px',
                                        'padding': '20px',
                                        'margin': '10px',
                                        'textAlign': 'center',
                                        'boxShadow': '0 4px 20px rgba(0, 0, 0, 0.3)'
                                    }
                                ),

                                # --- Countries ---
                                html.Div(
                                    [
                                        html.P(f'{countries}', style={
                                            'textAlign': 'center',
                                            'font-size': '2rem',
                                            'margin': '0'
                                        }),
                                        html.P('Countries', style={
                                            'textAlign': 'center',
                                            'margin': '0 0 10px 0'
                                        }),
                                        dcc.Graph(
                                            figure=px.line(
                                                df.groupby('release_year')['country'].nunique().reset_index(),
                                                x='release_year', y='country',
                                                height=100, width=200,
                                                color_discrete_sequence=['#E50914']
                                            ).update_layout(
                                                margin=dict(l=0, r=0, t=0, b=0),
                                                xaxis_visible=False, yaxis_visible=False,
                                                paper_bgcolor='rgba(0,0,0,0)',
                                                plot_bgcolor='rgba(0,0,0,0)'
                                            ),
                                            config={'displayModeBar': False},
                                            style={'marginTop': '5px'}
                                        )
                                    ],
                                    style={
                                        'flex': '1 1 220px',
                                        'maxWidth': '250px',
                                        'minWidth': '220px',
                                        'background': 'var(--graph-color)',
                                        'borderRadius': '16px',
                                        'padding': '20px',
                                        'margin': '10px',
                                        'textAlign': 'center',
                                        'boxShadow': '0 4px 20px rgba(0, 0, 0, 0.3)'
                                    }
                                ),

                                # --- Genres ---
                                html.Div(
                                    [
                                        html.P(f'{genres}', style={
                                            'textAlign': 'center',
                                            'font-size': '2rem',
                                            'margin': '0'
                                        }),
                                        html.P('Genres', style={
                                            'textAlign': 'center',
                                            'margin': '0 0 10px 0'
                                        }),
                                        dcc.Graph(
                                            figure=px.area(
                                                df.groupby('release_year')['listed_in']
                                                .apply(lambda x: x.str.split(',').explode().nunique())
                                                .reset_index(name='Genre Count'),
                                                x='release_year', y='Genre Count',
                                                height=100, width=200,
                                                color_discrete_sequence=['#E50914']
                                            ).update_layout(
                                                margin=dict(l=0, r=0, t=0, b=0),
                                                xaxis_visible=False, yaxis_visible=False,
                                                paper_bgcolor='rgba(0,0,0,0)',
                                                plot_bgcolor='rgba(0,0,0,0)'
                                            ).update_traces(mode='lines', fill='tozeroy'),
                                            config={'displayModeBar': False},
                                            style={'marginTop': '5px'}
                                        )
                                    ],
                                    style={
                                        'flex': '1 1 220px',
                                        'maxWidth': '250px',
                                        'minWidth': '220px',
                                        'background': 'var(--graph-color)',
                                        'borderRadius': '16px',
                                        'padding': '20px',
                                        'margin': '10px',
                                        'textAlign': 'center',
                                        'boxShadow': '0 4px 20px rgba(0, 0, 0, 0.3)'
                                    }
                                ),

                                # --- Shannon Diversity Index ---
                                html.Div(
                                    [
                                        html.P(f'{shannon_index:.2f}', style={
                                            'textAlign': 'center',
                                            'font-size': '2rem',
                                            'margin': '0'
                                        }),
                                        html.P('Shannon Diversity Index', style={
                                            'textAlign': 'center',
                                            'margin': '0 0 10px 0'
                                        }),
                                        dcc.Graph(
                                            figure=px.area(
                                                df.groupby('release_year')['country']
                                                .apply(lambda x: x.dropna().str.split(',').explode().nunique())
                                                .reset_index(name='Diversity'),
                                                x='release_year', y='Diversity',
                                                height=100, width=200,
                                                color_discrete_sequence=['#B0262D']
                                            ).update_layout(
                                                margin=dict(l=0, r=0, t=0, b=0),
                                                xaxis_visible=False, yaxis_visible=False,
                                                paper_bgcolor='rgba(0,0,0,0)',
                                                plot_bgcolor='rgba(0,0,0,0)'
                                            ).update_traces(mode='lines', fill='tozeroy'),
                                            config={'displayModeBar': False},
                                            style={'marginTop': '5px'}
                                        )
                                    ],
                                    style={
                                        'flex': '1 1 220px',
                                        'maxWidth': '250px',
                                        'minWidth': '220px',
                                        'background': 'var(--graph-color)',
                                        'borderRadius': '16px',
                                        'padding': '20px',
                                        'margin': '10px',
                                        'textAlign': 'center',
                                        'boxShadow': '0 4px 20px rgba(0, 0, 0, 0.3)'
                                    }
                                ),
                            ],
                            style={
                                'display': 'flex',
                                'flex-wrap': 'wrap',
                                'justify-content': 'center',
                                'align-items': 'stretch',
                                'gap': '1rem'
                            }
                        )
                    ],
                    style={
                        'width': 'full',
                        'display': 'flex',
                        'flex-direction': 'column',
                        'margin': '10px',
                        'background': 'var(--graph-color)',
                        'borderRadius': '16px',
                        'padding': '30px',
                    }
                ),

            ],
            style={
                'width': 'full',
                'display': 'flex',
                'flex-direction': 'column',
                'margin': '10px',
                'background': 'var(--graph-color)',
                'backdropFilter': 'blur(8px)',
                'borderRadius': '16px',
                'padding': '30px',
                'boxShadow': '0 4px 30px rgba(0, 0, 0, 0.4)',
            }
        ),
        html.Div(
            [
                html.Div(
                    html.P(
                        "High Level Summary Charts",
                        style={
                            'flex': '1',
                            'margin': '10px',
                            'background': 'var(--graph-color)',
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
                    dcc.Graph(id='year-bar', figure=fig_bar, config={'displayModeBar': False}),
                    className='chart-card',
                    style={
                        'flex': '1',
                        'margin': '10px',
                        'background': 'var(--graph-color)',
                        'backdropFilter': 'blur(8px)',
                        'borderRadius': '16px',
                        'padding': '30px',
                        'boxShadow': '0 4px 30px rgba(0, 0, 0, 0.4)',
                        'minWidth': '50%'
                    }
                ),
                html.Div(
                    dcc.Graph(id='type-pie', figure=fig_pie, config={'displayModeBar': False}),
                    className='chart-card',
                    style={
                        'flex': '1',
                        'margin': '10px',
                        'background': 'var(--graph-color)',
                        'backdropFilter': 'blur(8px)',
                        'borderRadius': '16px',
                        'padding': '30px',
                        'boxShadow': '0 4px 30px rgba(0, 0, 0, 0.4)',
                        'minWidth': '30%'
                    }
                ),
                html.Div(
                    dcc.Graph(id='category-hist', figure=fig_hist, config={'displayModeBar': False}),
                    className='chart-card',
                    style={
                        'flex': '1',
                        'margin': '10px',
                        'background': 'var(--graph-color)',
                        'backdropFilter': 'blur(8px)',
                        'borderRadius': '16px',
                        'padding': '30px',
                        'boxShadow': '0 4px 30px rgba(0, 0, 0, 0.4)',
                        'minWidth': '30%'
                    }
                ),
                html.Div(
                    dcc.Graph(id='country-bar', figure=fig_country_bar, config={'displayModeBar': False}),
                    className='chart-card',
                    style={
                        'flex': '1',
                        'margin': '10px',
                        'background': 'var(--graph-color)',
                        'backdropFilter': 'blur(8px)',
                        'borderRadius': '16px',
                        'padding': '30px',
                        'boxShadow': '0 4px 30px rgba(0, 0, 0, 0.4)',
                        'minWidth': '50%'
                    }
                ),
                html.Div(
                    dcc.Graph(id='lang-bar', figure=fig_lang_bar, config={'displayModeBar': False}),
                    className='chart-card',
                    style={
                        'flex': '1',
                        'margin': '10px',
                        'background': 'var(--graph-color)',
                        'backdropFilter': 'blur(8px)',
                        'borderRadius': '16px',
                        'padding': '30px',
                        'boxShadow': '0 4px 30px rgba(0, 0, 0, 0.4)',
                        'minWidth': '40%'
                    }
                ),
                html.Div(
                    dcc.Graph(id='line-chart', figure=fig_line, config={'displayModeBar': False}),
                    className='chart-card',
                    style={
                        'flex': '1',
                        'margin': '10px',
                        'background': 'var(--graph-color)',
                        'backdropFilter': 'blur(8px)',
                        'borderRadius': '16px',
                        'padding': '30px',
                        'boxShadow': '0 4px 30px rgba(0, 0, 0, 0.4)',
                        'minWidth': '40%'
                    }
                ),
                html.Div(
                    dcc.Graph(id='genre-bar', figure=fig_genre_bar, config={'displayModeBar': False}),
                    className='chart-card',
                    style={
                        'flex': '1',
                        'margin': '10px',
                        'background': 'var(--graph-color)',
                        'backdropFilter': 'blur(8px)',
                        'borderRadius': '16px',
                        'padding': '30px',
                        'boxShadow': '0 4px 30px rgba(0, 0, 0, 0.4)',
                        'minWidth': '40%'
                    }
                ),
            ],
            style={
                'display': 'flex',
                'flexWrap': 'wrap',
                'justifyContent': 'center',
                'alignItems': 'stretch',
                'margin': 'auto'
            }
        ),

        # Insight cards
        html.Div(
            [
                html.Div([
                    html.H4("Systemic Data Anomaly", style={'color': '#E50914'}),
                    html.P("~30% of titles have 'Unknown Director', mostly TV shows — a signal of missing metadata integrity.")
                ], style={'minWidth': '250px', 'background': 'var(--graph-color)', 'padding': '20px', 'borderRadius': '12px', 'backdropFilter': 'blur(8px)', 'boxShadow': '0 4px 30px rgba(0, 0, 0, 0.4)'}),

                html.Div([
                    html.H4("Bifurcated Production Model", style={'color': '#E50914'}),
                    html.P("Distinct creator pipelines emerging: High-volume TV Factory vs. Project-based Movie Studio.")
                ], style={'minWidth': '250px', 'background': 'var(--graph-color)', 'padding': '20px', 'borderRadius': '12px', 'backdropFilter': 'blur(8px)', 'boxShadow': '0 4px 30px rgba(0, 0, 0, 0.4)'}),

                html.Div([
                    html.H4("Talent Concentration Risk", style={'color': '#E50914'}),
                    html.P("A small elite set of directors and actors dominate global output — concentration bottleneck risk.")
                ], style={'minWidth': '250px', 'background': 'var(--graph-color)', 'padding': '20px', 'borderRadius': '12px', 'backdropFilter': 'blur(8px)', 'boxShadow': '0 4px 30px rgba(0, 0, 0, 0.4)'}),

                html.Div([
                    html.H4("Strategic Versatility Assets", style={'color': '#E50914'}),
                    html.P("Multi-genre creators act as bridge assets to expand new verticals efficiently.")
                ], style={'minWidth': '250px', 'background': 'var(--graph-color)', 'padding': '20px', 'borderRadius': '12px', 'backdropFilter': 'blur(8px)', 'boxShadow': '0 4px 30px rgba(0, 0, 0, 0.4)'})
            ],
            style={
                'display': 'flex',
                'overflowX': 'auto',
                'gap': '1rem',
                'marginTop': '40px',
                'paddingBottom': '20px',
                'color': 'var(--font-color)'
            }
        ),

        # Recommendations
        html.Div(
            [
                html.H3("Strategic Recommendations Preview", style={'color': '#E50914'}),
                html.Ul([
                    html.Li("A strategic shift toward retention-driven Originals and globally scalable content."),
                    html.Li("Targeted genre diversification to super-serve niches and boost discovery."),
                    html.Li("Household-wide engagement through family content and micro-genre personalization."),
                    html.Li("Smarter investment allocation guided by growth, quality, and creator strategic value.")
                ], style={'color': 'var(--font-color)', 'fontSize': '1.1rem', 'lineHeight': '1.6'})
            ],
            style={
                'background': 'var(--graph-color)',
                'borderRadius': '16px',
                'padding': '30px',
                'marginTop': '40px',
                'marginBottom': '40px',
                'boxShadow': '0 4px 30px rgba(0,0,0,0.4)'
            }
        )
    ],
    className='exec-overview',
    style={
        'backgroundColor': 'var(--background-color)',
        'minHeight': '100vh',
        'padding': '60px 20px',
        'fontFamily': 'Segoe UI, sans-serif',
        'overflow': 'scroll-y'
    }
)

# =========================
# THEME RESTYLE (no redesign)
# =========================

def _apply_theme(fig, theme):
    """Restyle an existing figure to match the theme, without changing data/structure."""
    if theme == "light":
        text_color = "#0F0F0F"
        bg_color = "#FFFFFF"
        grid_color = "#DDDDDD"
    else:
        text_color = "#FFFFFF"
        bg_color = "rgba(0,0,0,0)"
        grid_color = "#333333"

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font=dict(color=text_color),
        xaxis=dict(
            showgrid=False,
            tickfont=dict(color=text_color),
            gridcolor=grid_color
        ),
        yaxis=dict(
            showgrid=False,
            tickfont=dict(color=text_color),
            gridcolor=grid_color
        ),
        legend=dict(font=dict(color=text_color))
    )
    return fig


@callback(Output("type-pie", "figure", allow_duplicate=True),
          Input("current-theme", "data"),
          prevent_initial_call="initial_duplicate")
def _theme_pie(theme):
    return _apply_theme(fig_pie, theme)

@callback(Output("year-bar", "figure", allow_duplicate=True),
          Input("current-theme", "data"),
          prevent_initial_call="initial_duplicate")
def _theme_year_bar(theme):
    return _apply_theme(fig_bar, theme)

@callback(Output("category-hist", "figure", allow_duplicate=True),
          Input("current-theme", "data"),
          prevent_initial_call="initial_duplicate")
def _theme_hist(theme):
    return _apply_theme(fig_hist, theme)

@callback(Output("country-bar", "figure", allow_duplicate=True),
          Input("current-theme", "data"),
          prevent_initial_call="initial_duplicate")
def _theme_country(theme):
    return _apply_theme(fig_country_bar, theme)

@callback(Output("lang-bar", "figure", allow_duplicate=True),
          Input("current-theme", "data"),
          prevent_initial_call="initial_duplicate")
def _theme_lang(theme):
    return _apply_theme(fig_lang_bar, theme)

@callback(Output("line-chart", "figure", allow_duplicate=True),
          Input("current-theme", "data"),
          prevent_initial_call="initial_duplicate")
def _theme_line(theme):
    return _apply_theme(fig_line, theme)

@callback(Output("genre-bar", "figure", allow_duplicate=True),
          Input("current-theme", "data"),
          prevent_initial_call="initial_duplicate")
def _theme_genre(theme):
    return _apply_theme(fig_genre_bar, theme)
