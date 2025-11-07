from dash import Dash, html, dcc, dash_table, Output, Input, State, callback_context
import pandas as pd
import plotly.express as px
import numpy as np

# # --- data preprocessing --- 

# # Load data
df = pd.read_csv('./data/netflix_titles.csv')

# --- Clean & Prepare Data ---
df_trend = df.copy()
df_trend['date_added'] = pd.to_datetime(df_trend['date_added'], errors='coerce')
df_trend['year_added'] = df_trend['date_added'].dt.year
df_trend = df_trend[df_trend['year_added'].notna()]
df_trend = df_trend[df_trend['year_added'] >= 2010]

# --- Genre prep ---
df_trend['listed_in'] = df_trend['listed_in'].str.split(',')
df_genre_time = df_trend.explode('listed_in')
df_genre_time['listed_in'] = df_genre_time['listed_in'].str.strip()

# --- Dropdown options ---
type_options = [
    {'label': 'Movies', 'value': 'Movie'},
    {'label': 'TV Shows', 'value': 'TV Show'},
    {'label': 'Both', 'value': 'Both'}
]

# --- Layout ---
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
                html.Span("T", style={'color': '#E50914'}),
                "rend ",
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
                'padding': 0
            }
        ),
        html.P(
            "Dynamic overview of Netflix’s evolving content trends, showcasing shifts in genres, release volumes, and audience preferences over time.",
            style={
                'textAlign': 'center',
                'color': '#ccc',
                'fontSize': '1.1rem',
                'marginBottom': '50px'
            }
        ),
        html.Div(
            [
                html.Label("Filter Content Type:", style={'color': 'white', 'marginRight': '10px'}),
                dcc.Dropdown(
                    id='trend-type-dropdown',
                    options=type_options,
                    value='Both',
                    clearable=False,
                    style={
                        'width': '250px',
                        'backgroundColor': '#1e1e1e',
                        'color': 'white',
                        'align': 'center'
                    }
                )
            ],
            style={
                'textAlign': 'center', 
                'marginBottom': '30px', 
                'width': 'full',
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
                    html.P(
                        "Interactive Time series Graphs",
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
                html.Div(dcc.Graph(id='trend-growth-graph', config={'displayModeBar': False}), style=card_style()),
                html.Div(dcc.Graph(id='trend-genre-graph', config={'displayModeBar': False}), style=card_style()),
                html.Div(dcc.Graph(id='trend-month-graph', config={'displayModeBar': False}), style=card_style()),
                html.Div(dcc.Graph(id='trend-emerging-graph', config={'displayModeBar': False}), style=card_style()),
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
    className='geo-insights',
    style={
        'backgroundColor': '#121212',
        'minHeight': '100vh',
        'padding': '60px 20px',
        'fontFamily': 'Segoe UI, sanautosize=Falses-serif',
        'overflow': 'scroll-y'
    }
)

def register_trend_callbacks(app):
    @app.callback(
        Output('trend-growth-graph', 'figure'),
        Output('trend-genre-graph', 'figure'),
        Output('trend-month-graph', 'figure'),
        Output('trend-emerging-graph', 'figure'),
        Input('trend-type-dropdown', 'value')
    )
    def update_trend_charts(selected_type):
        # --- 0️⃣ Base filtering ---
        df_filtered = df_trend.copy()
        if selected_type in ['Movie', 'TV Show']:
            df_filtered = df_filtered[df_filtered['type'] == selected_type]

        # --- 1️⃣ Content Growth ---
        growth_trend = (
            df_filtered
            .groupby(['year_added', 'type'])
            .size()
            .reset_index(name='Count')
        )
        fig_growth_trend = px.line(
            growth_trend,
            x='year_added',
            y='Count',
            color='type',
            markers=True,
            title='Content Growth Over Time',
            color_discrete_sequence=['#E50914', '#B20710']
        )
        fig_growth_trend.update_layout(
            title={'x': 0.5, 'xanchor': 'center', 'font': {'size': 22, 'color': 'white'}},
            xaxis_title="Year Added",
            yaxis_title="Number of Titles",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=500
        )

        # --- 2️⃣ Genre Evolution ---
        genre_df = df_genre_time[df_genre_time['year_added'] >= 2010].copy()
        if selected_type in ['Movie', 'TV Show']:
            genre_df = genre_df[genre_df['type'] == selected_type]

        genre_trend = (
            genre_df
            .groupby(['year_added', 'listed_in'])
            .size()
            .reset_index(name='Count')
        )

        # Select top 10 genres overall
        top_genres = genre_trend.groupby('listed_in')['Count'].sum().nlargest(10).index
        genre_trend = genre_trend[genre_trend['listed_in'].isin(top_genres)]

        fig_genre_trend = px.line(
            genre_trend,
            x='year_added',
            y='Count',
            color='listed_in',
            markers=True,
            title='Top 10 Genres Evolution Over Time',
            color_discrete_sequence=px.colors.sequential.Reds
        )
        fig_genre_trend.update_layout(
            title={'x': 0.5, 'xanchor': 'center', 'font': {'size': 22, 'color': 'white'}},
            xaxis_title="Year Added",
            yaxis_title="Number of Titles",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=600
        )

        # --- 3️⃣ Seasonal Trends ---
        df_filtered['month_added'] = df_filtered['date_added'].dt.month
        month_map = {
            1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
            7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'
        }
        df_filtered['month_name'] = df_filtered['month_added'].map(month_map)

        monthly_trend = (
            df_filtered
            .groupby(['month_name', 'type'])
            .size()
            .reset_index(name='Count')
        )
        month_order = list(month_map.values())

        fig_month_trend = px.bar(
            monthly_trend,
            x='month_name',
            y='Count',
            color='type',
            barmode='group',
            category_orders={'month_name': month_order},
            color_discrete_sequence=['#E50914', '#B20710'],
            title='Seasonal Content Additions (By Month)'
        )
        fig_month_trend.update_layout(
            title={'x': 0.5, 'xanchor': 'center', 'font': {'size': 22, 'color': 'white'}},
            xaxis_title="Month",
            yaxis_title="Titles Added",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=500
        )

        # --- 4️⃣ Emerging Genres ---
        recent_years = df_filtered[df_filtered['year_added'] >= df_filtered['year_added'].max() - 2].copy()
        recent_genres = recent_years.explode('listed_in')
        recent_genres['listed_in'] = recent_genres['listed_in'].str.strip()

        emerging_genres = (
            recent_genres['listed_in']
            .value_counts()
            .reset_index()
            .head(10)
        )
        emerging_genres.columns = ['Genre', 'Titles']

        fig_emerging_genres = px.bar(
            emerging_genres,
            x='Titles',
            y='Genre',
            orientation='h',
            color='Titles',
            color_continuous_scale='Reds',
            title='Top 10 Emerging Genres (Last 3 Years)'
        )
        fig_emerging_genres.update_layout(
            title={'x': 0.5, 'xanchor': 'center', 'font': {'size': 22, 'color': 'white'}},
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='white'),
            height=600
        )

        return fig_growth_trend, fig_genre_trend, fig_month_trend, fig_emerging_genres
