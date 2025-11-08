from dash import html, dcc, Output, Input
import pandas as pd
import plotly.express as px

# --- Data Preprocessing ---
df = pd.read_csv('./data/netflix_titles.csv')

df_trend = df.copy()
df_trend['date_added'] = pd.to_datetime(df_trend['date_added'], errors='coerce')
df_trend['year_added'] = df_trend['date_added'].dt.year
df_trend = df_trend[df_trend['year_added'].notna()]
df_trend = df_trend[df_trend['year_added'] >= 2010]

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
layout = html.Div(
    children=[
        html.H1(
            [
                html.Span("T", style={'color': '#E50914'}),
                "rend ",
                html.Span("I", style={'color': '#E50914'}),
                "ntelligence"
            ],
            id='trend-title',
            style={
                'textAlign': 'center',
                'fontFamily': 'Segoe UI, sans-serif',
                'fontWeight': '700',
                'fontSize': '2.5rem',
                'marginBottom': '10px'
            }
        ),
        html.P(
            "Dynamic overview of Netflix’s evolving content trends, showcasing shifts in genres, release volumes, and audience preferences over time.",
            id='trend-desc',
            style={
                'textAlign': 'center',
                'fontSize': '1.1rem',
                'marginBottom': '50px'
            }
        ),
        html.Div(
            [
                html.Label("Filter Content Type:", id='trend-label', style={'marginRight': '10px'}),
                dcc.Dropdown(
                    id='trend-type-dropdown',
                    options=type_options,
                    value='Both',
                    clearable=False,
                    style={
                        'width': '250px',
                        'backgroundColor': 'var(--background-color)',  # Always black
                        'color': 'white',
                        'border': '1.2px solid rgba(229,9,20,0.5)',
                        'textAlign': 'center'
                    }
                )
            ],
            style={
                'textAlign': 'center',
                'marginBottom': '40px',
                'display': 'flex',
                'justifyContent': 'center',
                'alignItems': 'center',
                'flexWrap': 'wrap'
            }
        ),

        html.P(
            "Interactive Time Series Graphs",
            id='trend-header-text',
            style={
                'textAlign': 'center',
                'fontSize': '1.2rem',
                'fontWeight': 'bold',
                'color': '#E50914',
                'marginBottom': '20px'
            }
        ),

        # --- Graphs Grid 2x2 ---
        html.Div(
            [
                html.Div(dcc.Graph(id='trend-growth-graph', config={'displayModeBar': False})),
                html.Div(dcc.Graph(id='trend-genre-graph', config={'displayModeBar': False})),
                html.Div(dcc.Graph(id='trend-month-graph', config={'displayModeBar': False})),
                html.Div(dcc.Graph(id='trend-emerging-graph', config={'displayModeBar': False})),
            ],
            id='trend-graphs-container',
            style={
                'display': 'flex',
                'flex-wrap': 'wrap',
                'gap': '20px',
                'justifyItems': 'center',
                'alignItems': 'stretch',
                'width': '100%',
                'margin': 'auto',
                'justify-content': 'center'
            }
        ),
    ],
    id='trend-page',
    style={
        'minHeight': '100vh',
        'padding': '60px 20px',
        'fontFamily': 'Segoe UI, sans-serif',
        'transition': 'background-color 0.3s ease'
    }
)


# --- Main Callback ---
def register_trend_callbacks(app):
    @app.callback(
        Output('trend-growth-graph', 'figure'),
        Output('trend-genre-graph', 'figure'),
        Output('trend-month-graph', 'figure'),
        Output('trend-emerging-graph', 'figure'),
        Output('trend-title', 'style'),
        Output('trend-desc', 'style'),
        Output('trend-label', 'style'),
        Output('trend-header-text', 'style'),
        Output('trend-page', 'style'),
        Input('trend-type-dropdown', 'value'),
        Input('current-theme', 'data')
    )
    def update_trend_charts(selected_type, current_theme):
        df_filtered = df_trend.copy()
        if selected_type in ['Movie', 'TV Show']:
            df_filtered = df_filtered[df_filtered['type'] == selected_type]

        # --- Theme Colors ---
        if current_theme == 'light':
            bg_color = '#ffffff'
            font_color = '#000000'
            subtext_color = '#555555'
            card_bg = '#ffffff'
        else:
            bg_color = '#121212'
            font_color = '#ffffff'
            subtext_color = '#cccccc'
            card_bg = '#1e1e1e'

        accent_color = '#E50914'

        # --- 1️⃣ Content Growth ---
        growth_trend = df_filtered.groupby(['year_added', 'type']).size().reset_index(name='Count')
        fig_growth_trend = px.line(
            growth_trend, x='year_added', y='Count', color='type', markers=True,
            title='Content Growth Over Time',
            color_discrete_sequence=['#E50914', '#B20710']
        )
        fig_growth_trend.update_layout(
            title={'x': 0.5, 'font': {'size': 22, 'color': accent_color}},
            paper_bgcolor=card_bg, plot_bgcolor=card_bg,
            font=dict(color=font_color), height=450
        )

        # --- 2️⃣ Genre Evolution ---
        genre_df = df_genre_time[df_genre_time['year_added'] >= 2010].copy()
        if selected_type in ['Movie', 'TV Show']:
            genre_df = genre_df[genre_df['type'] == selected_type]
        genre_trend = genre_df.groupby(['year_added', 'listed_in']).size().reset_index(name='Count')
        top_genres = genre_trend.groupby('listed_in')['Count'].sum().nlargest(10).index
        genre_trend = genre_trend[genre_trend['listed_in'].isin(top_genres)]

        fig_genre_trend = px.line(
            genre_trend, x='year_added', y='Count', color='listed_in', markers=True,
            title='Top 10 Genres Evolution Over Time',
            color_discrete_sequence=px.colors.sequential.Reds
        )
        fig_genre_trend.update_layout(
            title={'x': 0.5, 'font': {'size': 22, 'color': accent_color}},
            paper_bgcolor=card_bg, plot_bgcolor=card_bg,
            font=dict(color=font_color), height=450
        )

        # --- 3️⃣ Seasonal Trends ---
        df_filtered['month_added'] = df_filtered['date_added'].dt.month
        month_map = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun',
                     7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
        df_filtered['month_name'] = df_filtered['month_added'].map(month_map)
        monthly_trend = df_filtered.groupby(['month_name', 'type']).size().reset_index(name='Count')
        month_order = list(month_map.values())

        fig_month_trend = px.bar(
            monthly_trend, x='month_name', y='Count', color='type', barmode='group',
            category_orders={'month_name': month_order},
            color_discrete_sequence=['#E50914', '#B20710'],
            title='Seasonal Content Additions (By Month)'
        )
        fig_month_trend.update_layout(
            title={'x': 0.5, 'font': {'size': 22, 'color': accent_color}},
            paper_bgcolor=card_bg, plot_bgcolor=card_bg,
            font=dict(color=font_color), height=450
        )

        # --- 4️⃣ Emerging Genres (Balanced Contrast) ---
        recent_years = df_filtered[df_filtered['year_added'] >= df_filtered['year_added'].max() - 2].copy()
        recent_genres = recent_years.explode('listed_in')
        recent_genres['listed_in'] = recent_genres['listed_in'].str.strip()
        emerging_genres = recent_genres['listed_in'].value_counts().reset_index().head(10)
        emerging_genres.columns = ['Genre', 'Titles']

        # Balanced Netflix reds for both themes
        if current_theme == 'light':
            color_scale = ['#660000', '#B22222', '#FF6347', '#FF9999']  # Light mode contrast
        else:
            color_scale = ['#4C0000', '#8B0000', '#C41E3A', '#FF4C4C']  # Dark mode visibility

        fig_emerging_genres = px.bar(
            emerging_genres,
            x='Titles',
            y='Genre',
            orientation='h',
            color='Titles',
            color_continuous_scale=color_scale,
            title='Top 10 Emerging Genres (Last 3 Years)'
        )
        fig_emerging_genres.update_traces(marker_line_color='#333', marker_line_width=0.7)
        fig_emerging_genres.update_layout(
            title={'x': 0.5, 'font': {'size': 22, 'color': accent_color}},
            paper_bgcolor=card_bg, plot_bgcolor=card_bg,
            font=dict(color=font_color),
            coloraxis_colorbar=dict(title="Titles", tickcolor=font_color, tickfont=dict(color=font_color)),
            height=450
        )

        # --- Dynamic Styles ---
        title_style = {'textAlign': 'center', 'fontWeight': '700', 'fontSize': '2.5rem', 'color': font_color}
        desc_style = {'textAlign': 'center', 'fontSize': '1.1rem', 'marginBottom': '50px', 'color': subtext_color}
        label_style = {'color': font_color, 'marginRight': '10px'}
        header_style = {'color': accent_color, 'fontSize': '1.2rem', 'fontWeight': 'bold', 'textAlign': 'center'}
        page_style = {'backgroundColor': bg_color, 'minHeight': '100vh', 'padding': '60px 20px'}

        return (
            fig_growth_trend, fig_genre_trend, fig_month_trend, fig_emerging_genres,
            title_style, desc_style, label_style, header_style, page_style
        )