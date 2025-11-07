from dash import html, dcc, callback, Output, Input
import pandas as pd
import numpy as np
import plotly.express as px
import dash_cytoscape as cyto
from collections import defaultdict, Counter


# Load Data
df = pd.read_csv('./data/netflix_titles.csv')
df.fillna('', inplace=True)
df['release_year'] = pd.to_numeric(df['release_year'], errors='coerce')
df['listed_in'] = df['listed_in'].astype(str)

def split_people(s):
    return [x.strip() for x in s.split(',') if x.strip()] if isinstance(s, str) and s.strip() else []

def parse_year_added(s):
    try:
        return pd.to_datetime(s).year
    except Exception:
        return np.nan

df['year_added'] = df['date_added'].apply(parse_year_added)

dir_rows, act_rows = [], []

for _, row in df.iterrows():
    base = {
        'title': row.get('title', ''),
        'type': row.get('type', ''),
        'release_year': row.get('release_year', np.nan),
        'genres': [g.strip() for g in row.get('listed_in', '').split(',') if g.strip()],
        'country': row.get('country', ''),
        'rating': row.get('rating', ''),
        'duration': row.get('duration', ''),
        'year_added': row.get('year_added', np.nan),
        'description': row.get('description', ''),
    }

    if isinstance(row['director'], str) and row['director'].strip():
        dir_rows.append({
            **base,
            'name': row['director'].strip(),
            'role': 'Director'
        })

    for actor in split_people(row['cast']):
        act_rows.append({
            **base,
            'name': actor,
            'role': 'Actor'
        })

creators_df = pd.DataFrame(dir_rows + act_rows)
creator_names = sorted(creators_df['name'].dropna().unique().tolist())


TITLE_PEOPLE = {}
for _, row in df.iterrows():
    people = set()
    if isinstance(row.get('director', ''), str) and row['director'].strip():
        people.add(row['director'].strip())
    people.update(split_people(row.get('cast', '')))
    if people:
        TITLE_PEOPLE[row['title']] = people

NAME_TITLES = defaultdict(list)
for title, people in TITLE_PEOPLE.items():
    for p in people:
        NAME_TITLES[p].append(title)

NAME_ROLE = (
    creators_df.drop_duplicates(['name', 'role'])
    .groupby('name')['role']
    .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else 'Actor')
    .to_dict()
)


# ---------- Rising Stars Computation ----------
# ---------- Rising Stars Computation (correct: use release_year, not year_added) ----------
RECENT_YEARS = 5
# latest release year in your dataset
max_rel_year = int(creators_df['release_year'].dropna().max()) if creators_df['release_year'].notna().any() else None

if max_rel_year:
    recent_cut = max_rel_year - RECENT_YEARS + 1

    # only rows with a real release year
    grp = creators_df.dropna(subset=['release_year']).groupby('name', as_index=True)

    stats = grp['release_year'].agg(
        recent_count=lambda s: (s >= recent_cut).sum(),
        old_count=lambda s: (s < recent_cut).sum(),
        total_count='count',
        first_release='min',
        last_release='max'
    )

    # True newcomers: first ever work is within the last RECENT_YEARS
    rising = stats[
        (stats['first_release'] >= recent_cut) &   # started recently (excludes veterans like Anupam Kher)
        (stats['recent_count'] >= 2) &             # at least 2 recent titles
        (stats['total_count'] >= 2)                # at least 2 total titles
    ].copy()

    # Score: emphasize volume and recency span within window
    rising['rising_score'] = (
        (rising['recent_count'] / rising['total_count']) *
        (1 + (rising['last_release'] - rising['first_release']) / max(1, RECENT_YEARS - 1))
    )

    rising_top10 = (
        rising.sort_values(['rising_score', 'recent_count'], ascending=[False, False])
              .head(10)
              .reset_index()
    )
else:
    rising_top10 = pd.DataFrame(columns=['name', 'recent_count', 'old_count', 'total_count', 'rising_score'])




# Layout (Netflix-Themed)
layout = html.Div(
    children=[
        html.H1(
            children=[
                html.Span("C", style={'color': '#E50914'}),
                "reator and ",
                html.Span("T", style={'color': '#E50914'}),
                "alent Overview"
            ],
            style={'padding': 0, 'margin': 0, 'color': 'var(--font-color)'}
        ),
        html.P(
            "Select a creator to see their genre distribution.",
            style={'padding': 0, 'margin': 0, 'color': 'var(--font-color)'}
        ),
        dcc.Dropdown(
            id='creator-search',
            options=[{'label': name, 'value': name} for name in creator_names],
            placeholder='Search for a director or actor...',
            style={
                'width': '50%',
                'margin': '20px auto',
                'color': 'var(--font-color)',
                'backgroundColor': 'var(--background-color)'
            },
            value='Anupam Kher',
            searchable=True,
            clearable=True
        ),
        html.Div(
            [
                html.Div(
                    dcc.Graph(id='bar-chart', figure={}, config={'displayModeBar': False}),
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
                    dcc.Graph(id='pie-chart', figure={}, config={'displayModeBar': False}),
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
                    dcc.Graph(id='line-chart', figure={}, config={'displayModeBar': False}),
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
                    cyto.Cytoscape(
                        id='collab-graph',
                        layout={'name': 'cose', 'animate': False},
                        stylesheet=[
                            {
                                'selector': 'node',
                                'style': {
                                    'content': 'data(label)',
                                    'font-size': '12px',
                                    'color': '#FFFFFF',
                                    'text-outline-color': '#E50914',
                                    'text-outline-width': 2,
                                    'background-color': '#E50914',
                                    'border-color': '#FFFFFF',
                                    'border-width': 1
                                }
                            },
                            {
                                'selector': '[role = "Director"]',
                                'style': {'background-color': '#B81D24', 'shape': 'triangle'}
                            },
                            {
                                'selector': '[role = "Actor"]',
                                'style': {'background-color': '#E50914', 'shape': 'ellipse'}
                            },
                            {
                                'selector': 'edge',
                                'style': {
                                    'line-color': '#777777',
                                    'width': 1.5,
                                    'target-arrow-color': '#777777',
                                    'target-arrow-shape': 'vee',
                                    'curve-style': 'bezier'
                                }
                            },
                            {
                                'selector': ':selected',
                                'style': {
                                    'border-color': '#FFFFFF',
                                    'border-width': 3,
                                    'background-color': '#FFFFFF',
                                    'color': '#E50914'
                                }
                            }
                        ],
                        style={'width': '100%', 'height': '420px', 'backgroundColor': 'rgba(0,0,0,0)'},
                        elements=[]
                    ),
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
                    [
                        html.P(
                            "Rising Stars (5+)",
                            style={
                                'textAlign': 'center',
                                'fontWeight': 'bold',
                                'fontSize': '1.2rem',
                                'color': 'var(--font-color)',
                                'marginBottom': '10px'
                            }
                        ),
                        html.Div(
                            dcc.Graph(id='rising-stars-bar', figure={}, config={'displayModeBar': False}),
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
                        'flexDirection': 'column',
                        'alignItems': 'center',
                        'justifyContent': 'center',
                        'width': '100%',
                    }
                )

            ],
            style={
                'display': 'flex',
                'flexWrap': 'wrap',
                'justifyContent': 'center',
                'alignItems': 'stretch',
                'margin': 'auto'
            }
        ),
    ],
    className='genre-int',
    style={
        'backgroundColor': 'var(--background-color)',
        'minHeight': '100vh',
        'minWidth': '90vw',
        'padding': '60px 20px',
        'fontFamily': 'Segoe UI, sans-serif',
        'overflowY': 'scroll',
        'display': 'flex',
        'flexDirection': 'column',
        'alignItems': 'center'
    }
)


# Callbacks
@callback(
    Output("bar-chart", "figure", allow_duplicate=True),
    [Input('creator-search', "value"),
     Input('current-theme', 'data')],
    prevent_initial_call='initial_duplicate'
)
def generate_genre_bar_graph(selected_name, current_theme):
    if current_theme == 'light':
        text_color = '#0F0F0F'
        bg_color = '#FFFFFF'
    else:
        text_color = '#FFFFFF'
        bg_color = 'rgba(0,0,0,0)'

    if not selected_name:
        fig = px.bar(pd.DataFrame({'x': [], 'y': []}), x='x', y='y')
        fig.update_layout(
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            font_color=text_color
        )
        return fig

    sub_df = creators_df[creators_df['name'] == selected_name]
    all_genres = [g for lst in sub_df['genres'] for g in lst if g]

    if not all_genres:
        fig = px.bar(title=f"No genre data available for {selected_name}")
        fig.update_layout(
            paper_bgcolor=bg_color,
            plot_bgcolor=bg_color,
            font_color=text_color
        )
        return fig

    genre_counts = pd.Series(all_genres).value_counts().reset_index()
    genre_counts.columns = ['Genre', 'Count']

    fig = px.bar(
        genre_counts,
        x='Genre',
        y='Count',
        title=f"Genre Distribution for {selected_name}",
        text='Count',
        color_discrete_sequence=['#E50914']
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(
        xaxis_title='Genre',
        yaxis_title='Count',
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font=dict(size=20, color=text_color),
        margin=dict(l=40, r=20, t=60, b=60)
    )
    return fig


# PIE CHART
@callback(
    Output("pie-chart", "figure", allow_duplicate=True),
    [Input('creator-search', "value"),
     Input('current-theme', 'data')],
    prevent_initial_call='initial_duplicate'
)
def generate_type_pie_chart(selected_name, current_theme):
    if current_theme == 'light':
        text_color = '#0F0F0F'
        bg_color = '#FFFFFF'
    else:
        text_color = '#FFFFFF'
        bg_color = 'rgba(0,0,0,0)'

    if not selected_name:
        fig = px.pie(pd.DataFrame({'x': [], 'y': []}), names='x', values='y')
        fig.update_layout(paper_bgcolor=bg_color, plot_bgcolor=bg_color, font_color=text_color)
        return fig

    sub_df = creators_df[creators_df['name'] == selected_name]
    if sub_df.empty:
        fig = px.pie(pd.DataFrame({'x': [], 'y': []}), names='x', values='y')
        fig.update_layout(paper_bgcolor=bg_color, plot_bgcolor=bg_color, font_color=text_color)
        return fig

    type_counts = sub_df['type'].value_counts().reset_index()
    type_counts.columns = ['Type', 'Count']

    fig = px.pie(
        type_counts,
        names='Type',
        values='Count',
        title=f"Content Type Split for {selected_name}",
        hole=0.4,
        color_discrete_sequence=['#E50914', '#B81D24']
    )
    fig.update_traces(textinfo='label+percent', pull=[0.05]*len(type_counts))
    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font=dict(size=20, color=text_color),
        margin=dict(l=40, r=20, t=60, b=60),
        legend=dict(title='', orientation='h', y=-0.1, font=dict(color=text_color))
    )
    return fig


# LINE CHART
@callback(
    Output("line-chart", "figure", allow_duplicate=True),
    [Input('creator-search', "value"),
     Input('current-theme', 'data')],
    prevent_initial_call='initial_duplicate'
)
def generate_active_year_line_chart(selected_name, current_theme):
    if current_theme == 'light':
        text_color = '#0F0F0F'
        bg_color = '#FFFFFF'
        grid_color = '#CCCCCC'
    else:
        text_color = '#FFFFFF'
        bg_color = 'rgba(0,0,0,0)'
        grid_color = '#333333'

    if not selected_name:
        fig = px.line(pd.DataFrame({'x': [], 'y': []}), x='x', y='y')
        fig.update_layout(paper_bgcolor=bg_color, plot_bgcolor=bg_color, font_color=text_color)
        return fig

    sub_df = creators_df[creators_df['name'] == selected_name]
    yearly_counts = (
        sub_df.dropna(subset=['release_year'])
              .groupby('release_year').size()
              .reset_index(name='Count')
              .sort_values('release_year')
    )

    if yearly_counts.empty:
        fig = px.line(pd.DataFrame({'x': [], 'y': []}), x='x', y='y')
        fig.update_layout(paper_bgcolor=bg_color, plot_bgcolor=bg_color, font_color=text_color)
        return fig

    fig = px.line(yearly_counts, x='release_year', y='Count',
                  title=f"Yearly Activity of {selected_name}", markers=True)
    fig.update_traces(line=dict(width=3, color='#E50914'),
                      marker=dict(size=8, color='#B81D24'))
    fig.update_layout(
        yaxis_title='Titles',
        xaxis=dict(title='Release Year', showgrid=False),
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        title_font=dict(size=20, color=text_color),
        margin=dict(l=40, r=20, t=60, b=60),
        hovermode='x unified',
        yaxis=dict(gridcolor=grid_color)
    )
    return fig


# RISING STARS BAR
@callback(
    Output('rising-stars-bar', 'figure', allow_duplicate=True),
    [Input('creator-search', 'value'),
     Input('current-theme', 'data')],
    prevent_initial_call='initial_duplicate'
)
def rising_stars(_, current_theme):
    if current_theme == 'light':
        text_color = '#0F0F0F'
        bg_color = '#FFFFFF'
        grid_color = '#CCCCCC'
    else:
        text_color = '#FFFFFF'
        bg_color = 'rgba(0,0,0,0)'
        grid_color = '#333333'

    if rising_top10.empty:
        fig = px.bar(title="Rising Stars (Insufficient Data)")
    else:
        plotdf = rising_top10.copy()
        plotdf['Label'] = (
            plotdf['name'] + ' (' +
            plotdf['recent_count'].astype(int).astype(str) + '/' +
            plotdf['total_count'].astype(int).astype(str) + ')'
        )
        fig = px.bar(
            plotdf,
            x='Label',
            y='rising_score',
            text='rising_score',
            title=f"Rising Stars (Last {RECENT_YEARS} Years)",
            color_discrete_sequence=['#E50914']
        )
        fig.update_traces(texttemplate='%{text:.2%}', textposition='outside')
        fig.update_xaxes(tickangle=45)

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font_color=text_color,
        margin=dict(l=40, r=20, t=60, b=120),
        yaxis=dict(tickformat=".0%", gridcolor=grid_color),
        title_font=dict(size=20, color=text_color)
    )
    return fig

from collections import Counter

MAX_NEIGHBORS = 35  

@callback(Output('collab-graph', 'elements'), Input('creator-search', 'value'))
def build_collab_graph_fast(selected_name):
    if not selected_name:
        return []

    titles = NAME_TITLES.get(selected_name, [])
    if not titles:
        return [{'data': {'id': selected_name, 'label': selected_name, 'role': NAME_ROLE.get(selected_name, 'Actor')}}]

    collab_counter = Counter()
    for t in titles:
        for p in TITLE_PEOPLE.get(t, ()):
            if p != selected_name:
                collab_counter[p] += 1

    MAX_NEIGHBORS = 35
    top_collabs = [name for name, _ in collab_counter.most_common(MAX_NEIGHBORS)]

    elements = []
    elements.append({
        'data': {
            'id': selected_name,
            'label': selected_name,
            'role': NAME_ROLE.get(selected_name, 'Actor')
        }
    })
    for c in top_collabs:
        elements.append({
            'data': {
                'id': c,
                'label': c,
                'role': NAME_ROLE.get(c, 'Actor')
            }
        })

    for c in top_collabs:
        elements.append({'data': {'source': selected_name, 'target': c}})

    return elements
