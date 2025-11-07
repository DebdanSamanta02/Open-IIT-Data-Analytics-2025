from dash import html, dcc, dash_table, Input, Output, callback, State
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import dash

# Load and preprocess data
df = pd.read_csv('./data/netflix_titles.csv')

# Clean data
df['country'] = df['country'].fillna('Unknown')
df['listed_in'] = df['listed_in'].fillna('Unknown')
df['rating'] = df['rating'].fillna('Not Rated')
df['release_year'] = df['release_year'].fillna(0).astype(int)
df['director'] = df['director'].fillna('Unknown')
df['cast'] = df['cast'].fillna('Unknown')
df['description'] = df['description'].fillna('')

# Parse date_added
df['date_added'] = pd.to_datetime(df['date_added'], errors='coerce')
df['year_added'] = df['date_added'].dt.year
df['month_added'] = df['date_added'].dt.month

# Split multi-country and multi-genre values for filters
unique_countries = sorted(set(c.strip() for lst in df['country'].dropna().str.split(',') for c in lst if c.strip() != 'Unknown'))
unique_genres = sorted(set(g.strip() for lst in df['listed_in'].dropna().str.split(',') for g in lst if g.strip() != 'Unknown'))
unique_ratings = sorted([r for r in df['rating'].unique() if r != 'Not Rated'])
unique_types = sorted(df['type'].unique())

# ----------- LAYOUT ------------
layout = html.Div(
    [
        html.H1(
            [
                html.Span("C", style={'color': '#E50914'}),
                "ontent ",
                html.Span("E", style={'color': '#E50914'}),
                "xplorer"
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
            "Search and filter through Netflix's content library for strategic exploration.",
            style={'textAlign': 'center', 'color': '#ccc', 'fontSize': '1.1rem', 'marginBottom': '30px'}
        ),

        # --- ADVANCED FILTER BAR ---
        html.Div(
            [
                # Row 1: Search and primary filters
                html.Div([
                    dcc.Input(
                        id='search-title',
                        type='text',
                        placeholder='Search Title, Director, Cast, or Genre...',
                        debounce=True,
                        style={
                            'width': '100%',
                            'padding': '12px 15px',
                            'borderRadius': '8px',
                            'border': '1px solid rgba(229,9,20,0.5)',
                            'backgroundColor': 'var(--background-color)',
                            'color': 'white',
                            'fontSize': '1rem',
                            'boxShadow': 'inset 0 0 8px rgba(229,9,20,0.1)',
                            'outline': 'none',
                        }
                    ),
                ], style={'gridColumn': '1 / -1', 'marginBottom': '12px'}),
                
                # Row 2: Dropdowns
                dcc.Dropdown(
                    id='filter-type',
                    options=[{'label': 'All Types', 'value': 'all'}] + [{'label': t, 'value': t} for t in unique_types],
                    value='all',
                    placeholder='Type',
                    className='filter-dropdown',
                    style={
                        'backgroundColor': 'var(--background-color)',
                        'color': 'white',
                    }
                ),
                dcc.Dropdown(
                    id='filter-country',
                    options=[{'label': 'All Countries', 'value': 'all'}] + [{'label': c, 'value': c} for c in unique_countries],
                    value='all',
                    placeholder='Country',
                    className='filter-dropdown',
                    style={
                        'backgroundColor': 'var(--background-color)',
                        'color': 'white',
                    }
                ),
                dcc.Dropdown(
                    id='filter-genre',
                    options=[{'label': 'All Genres', 'value': 'all'}] + [{'label': g, 'value': g} for g in unique_genres],
                    value='all',
                    placeholder='Genre',
                    className='filter-dropdown',
                    style={
                        'backgroundColor': 'var(--background-color)',
                        'color': 'white',
                    }
                ),
                dcc.Dropdown(
                    id='filter-rating',
                    options=[{'label': 'All Ratings', 'value': 'all'}] + [{'label': r, 'value': r} for r in unique_ratings],
                    value='all',
                    placeholder='Rating',
                    className='filter-dropdown',
                    style={
                        'backgroundColor': 'var(--background-color)',
                        'color': 'white',
                    }
                ),
                
                # Row 3: Year range slider
                html.Div([
                    html.Label('Release Year Range:', style={'color': '#ccc', 'marginBottom': '8px', 'fontSize': '0.9rem'}),
                    dcc.RangeSlider(
                        id='year-range',
                        min=int(df['release_year'].min()),
                        max=int(df['release_year'].max()),
                        value=[int(df['release_year'].min()), int(df['release_year'].max())],
                        marks={year: str(year) for year in range(int(df['release_year'].min()), int(df['release_year'].max())+1, 10)},
                        tooltip={"placement": "bottom", "always_visible": False},
                        className='year-slider',
                        updatemode='drag'
                    ),
                ], style={'gridColumn': '1 / -1', 'marginTop': '15px'}),
            ],
            style={
                'display': 'grid',
                'gridTemplateColumns': '1fr 1fr 1fr 1fr',
                'gap': '12px',
                'alignItems': 'start',
                'width': '95%',
                'margin': 'auto',
                'marginBottom': '25px'
            }
        ),

        # --- ACTION BUTTONS ---
        html.Div([
            html.Button(
                '🔄 Reset Filters',
                id='reset-button',
                n_clicks=0,
                style={
                    'padding': '10px 24px',
                    'backgroundColor': '#E50914',
                    'color': 'white',
                    'border': 'none',
                    'borderRadius': '6px',
                    'cursor': 'pointer',
                    'fontSize': '0.95rem',
                    'fontWeight': '600',
                    'marginRight': '12px',
                    'boxShadow': '0 2px 8px rgba(229,9,20,0.3)',
                    'transition': 'all 0.3s ease'
                }
            ),
            html.Button(
                id='stats-button',
                n_clicks=0,
                children=['📊 View Quick Stats'],
                style={
                    'padding': '10px 24px',
                    'backgroundColor': '#333',
                    'color': 'white',
                    'border': '1px solid #555',
                    'borderRadius': '6px',
                    'cursor': 'pointer',
                    'fontSize': '0.95rem',
                    'fontWeight': '600',
                    'boxShadow': '0 2px 8px rgba(0,0,0,0.3)',
                    'transition': 'all 0.3s ease'
                }
            ),
        ], style={'textAlign': 'center', 'marginBottom': '20px'}),

        # --- QUICK STATS PANEL ---
        html.Div(id='quick-stats', style={'marginBottom': '20px'}),

        # --- SORT & EXPORT OPTIONS ---
        html.Div([
            html.Div([
                html.Label('Sort by: ', style={'color': '#ccc', 'marginRight': '10px', 'fontSize': '0.95rem'}),
                dcc.Dropdown(
                    id='sort-by',
                    options=[
                        {'label': 'Title (A-Z)', 'value': 'title_asc'},
                        {'label': 'Title (Z-A)', 'value': 'title_desc'},
                        {'label': 'Release Year (Newest)', 'value': 'year_desc'},
                        {'label': 'Release Year (Oldest)', 'value': 'year_asc'},
                        {'label': 'Date Added (Recent)', 'value': 'added_desc'},
                        {'label': 'Date Added (Oldest)', 'value': 'added_asc'},
                    ],
                    value='added_desc',
                    style={
                        'width': '250px',
                        'backgroundColor': 'var(--background-color)',
                        'color': 'white'
                    }
                ),
            ], style={'display': 'flex', 'alignItems': 'center'}),
            
            html.Div([
                html.Div(id='content-count', style={'color': '#ccc', 'fontSize': '1.05rem', 'marginRight': '20px'}),
                html.Button(
                    '📥 Export CSV',
                    id='export-button',
                    n_clicks=0,
                    style={
                        'padding': '8px 20px',
                        'backgroundColor': '#444',
                        'color': 'white',
                        'border': 'none',
                        'borderRadius': '6px',
                        'cursor': 'pointer',
                        'fontSize': '0.9rem',
                        'fontWeight': '600',
                        'transition': 'all 0.3s ease'
                    }
                ),
                dcc.Download(id='download-data'),
            ], style={'display': 'flex', 'alignItems': 'center'}),
        ], style={
            'display': 'flex', 
            'justifyContent': 'space-between', 
            'alignItems': 'center', 
            'width': '96%',
            'margin': 'auto',
            'marginBottom': '15px'
        }),

        # --- DATA TABLE ---
        html.Div(
            dash_table.DataTable(
                id='content-table',
                columns=[
                    {'name': 'Title', 'id': 'title'},
                    {'name': 'Type', 'id': 'type'},
                    {'name': 'Director', 'id': 'director'},
                    {'name': 'Cast', 'id': 'cast'},
                    {'name': 'Country', 'id': 'country'},
                    {'name': 'Release Year', 'id': 'release_year'},
                    {'name': 'Rating', 'id': 'rating'},
                    {'name': 'Duration', 'id': 'duration'},
                    {'name': 'Genre(s)', 'id': 'listed_in'},
                ],
                page_size=15,
                style_table={'overflowX': 'auto', 'width': '100%'},
                style_header={
                    'backgroundColor': '#E50914',
                    'color': 'white',
                    'fontWeight': 'bold',
                    'textAlign': 'center',
                    'fontSize': '0.95rem',
                    'height': '45px'
                },
                style_cell={
                    'backgroundColor': '#181818',
                    'color': 'white',
                    'textAlign': 'left',
                    'padding': '12px 10px',
                    'fontFamily': 'Segoe UI',
                    'fontSize': '0.9rem',
                    'lineHeight': '1.3rem',
                    'minHeight': '40px',
                    'whiteSpace': 'normal',
                    'width': 'auto',
                },
                style_data_conditional=[
                    {
                        'if': {'row_index': 'odd'},
                        'backgroundColor': '#202020'
                    },
                    
                ],
                css=[
                    {
                        'selector': 'tr:hover td',
                        'rule': 'background-color: #2a2a2a !important; cursor: pointer !important;'
                    },
                    {
                        'selector': '.dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner td',
                        'rule': 'border: none !important;'
                    },
                    {
                        'selector': '.dash-table-container .dash-spreadsheet-container .dash-spreadsheet-inner td.focused',
                        'rule': 'background-color: #2a2a2a; border: white !important; outline: none !important;'
                    },
                    {
                        'selector': '.Select-menu-outer',
                        'rule': 'background-color: var(--background-color) !important; border: 1px solid #555 !important;'
                    },
                    {
                        'selector': '.Select-option',
                        'rule': 'background-color: var(--background-color) !important; color: white !important;'
                    },
                    {
                        'selector': '.Select-option:hover',
                        'rule': 'background-color: #333 !important;'
                    },
                    {
                        'selector': '.paging-container',
                        'rule': 'background-color: var(--background-color) !important; border-top: 1px solid rgba(255, 255, 255, 0.1) !important; padding: 20px 24px !important; display: flex !important; align-items: center !important; justify-content: center !important; gap: 12px !important; border-radius: 0 !important; min-height: 70px !important;'
                    },
                    {
                        'selector': '.paging-container button',
                        'rule': 'background-color: #333 !important; color: white !important; border: 1px solid #555 !important; border-radius: 6px !important; padding: 8px 14px !important; font-size: 0.9rem !important; font-weight: 600 !important; cursor: pointer !important; transition: all 0.2s ease !important; min-width: 40px !important; height: 38px !important; font-family: "Segoe UI", sans-serif !important;'
                    },
                    {
                        'selector': '.paging-container button:hover:not(:disabled)',
                        'rule': 'background-color: #E50914 !important; border-color: #E50914 !important; transform: translateY(-1px) !important; box-shadow: 0 2px 8px rgba(229, 9, 20, 0.3) !important;'
                    },
                    {
                        'selector': '.paging-container button:disabled',
                        'rule': 'background-color: #1a1a1a !important; color: #666 !important; border-color: #2a2a2a !important; cursor: not-allowed !important; opacity: 0.5 !important;'
                    },
                    {
                        'selector': '.paging-container .current-page',
                        'rule': 'background-color: white !important; border: 2px solid white !important; color: black !important; border-radius: 8px !important; padding: 10px 18px !important; font-size: 1rem !important; font-weight: 700 !important; box-shadow: 0 4px 12px rgba(255, 255, 255, 0.3), 0 2px 6px rgba(0, 0, 0, 0.2) !important; min-width: 50px !important; height: 42px !important;'
                    },
                    {
                        'selector': '.paging-container',
                        'rule': 'color: white !important;'
                    },
                    {
                        'selector': '.paging-container *',
                        'rule': 'color: white !important;'
                    },
                    {
                        'selector': '.paging-container span',
                        'rule': 'color: white !important; font-size: 0.95rem !important; font-weight: 500 !important; margin: 0 4px !important; font-family: "Segoe UI", sans-serif !important;'
                    },
                    {
                        'selector': '.paging-container div',
                        'rule': 'color: white !important; font-size: 0.95rem !important; font-weight: 500 !important;'
                    },
                    {
                        'selector': '.paging-container .paging-nav *:not(button)',
                        'rule': 'color: white !important;'
                    },
                    {
                        'selector': '.paging-container *:not(button)',
                        'rule': 'color: white !important;'
                    },
                    {
                        'selector': '.paging-container [style*="color"]',
                        'rule': 'color: white !important;'
                    },
                    {
                        'selector': '.paging-container div div',
                        'rule': 'color: white !important;'
                    },
                    {
                        'selector': '.paging-container span span',
                        'rule': 'color: white !important;'
                    },
                    {
                        'selector': '.paging-container button.current-page',
                        'rule': 'color: black !important; background-color: white !important; border: 2px solid white !important; border-radius: 8px !important; padding: 10px 18px !important; font-size: 1rem !important; font-weight: 700 !important; box-shadow: 0 4px 12px rgba(255, 255, 255, 0.3), 0 2px 6px rgba(0, 0, 0, 0.2) !important; min-width: 50px !important; height: 42px !important; display: inline-flex !important; align-items: center !important; justify-content: center !important;'
                    },
                    {
                        'selector': '.paging-container input.current-page',
                        'rule': 'color: black !important; background-color: white !important; border: 2px solid white !important; border-radius: 8px !important; padding: 10px 18px !important; font-size: 1rem !important; font-weight: 700 !important; box-shadow: 0 4px 12px rgba(255, 255, 255, 0.3), 0 2px 6px rgba(0, 0, 0, 0.2) !important; min-width: 50px !important; height: 42px !important; text-align: center !important;'
                    },
                    {
                        'selector': '.paging-container .paging-page-info',
                        'rule': 'display: inline-flex !important; align-items: center !important; gap: 10px !important; color: white !important; font-size: 1rem !important; font-weight: 500 !important;'
                    },
                    {
                        'selector': '.paging-container div.last-page',
                        'rule': 'color: white !important; font-size: 1rem !important; font-weight: 600 !important; padding: 0 4px !important;'
                    },
                    {
                        'selector': '.paging-container div.page-number',
                        'rule': 'color: white !important; font-size: 1rem !important; font-weight: 600 !important;'
                    },
                    {
                        'selector': '.paging-container button.current-page ~ *',
                        'rule': 'color: white !important;'
                    },
                    {
                        'selector': '.paging-container button.current-page + *',
                        'rule': 'color: white !important;'
                    },
                ],
                sort_action='native',
                filter_action='none',
            ),
            style={
                'margin': 'auto',
                'width': '96%',
                'boxShadow': '0 4px 20px rgba(0,0,0,0.4)',
                'marginBottom': '30px'
            }
        ),
        
        # --- VIEW DETAILS INSTRUCTIONS ---
        html.P(
            "Click on any row to view full details",
            style={
                'textAlign': 'center',
                'color': '#888',
                'fontSize': '0.9rem',
                'fontStyle': 'italic',
                'marginBottom': '40px'
            }
        ),

        # --- POPUP MODAL ---
        html.Div(
            id='detail-popup',
            style={'display': 'none'},
            children=[
                # Overlay background
                html.Div(
                    id='popup-overlay',
                    style={
                        'position': 'fixed',
                        'top': 0,
                        'left': 0,
                        'width': '100%',
                        'height': '100%',
                        'backgroundColor': 'rgba(0, 0, 0, 0.85)',
                        'zIndex': 9998,
                        'backdropFilter': 'blur(5px)',
                        'opacity': 0,
                        'transition': 'opacity 0.3s ease-in-out',
                    }
                ),
                # Popup content
                html.Div(
                    id='popup-content',
                    style={
                        'position': 'fixed',
                        'top': '50%',
                        'left': '50%',
                        'transform': 'translate(-50%, -50%) scale(0.7)',
                        'backgroundColor': 'var(--background-color)',
                        'padding': '0',
                        'borderRadius': '15px',
                        'boxShadow': '0 10px 40px rgba(229, 9, 20, 0.5)',
                        'zIndex': 9999,
                        'width': '90%',
                        'maxWidth': '900px',
                        'maxHeight': '85vh',
                        'overflowY': 'auto',
                        'border': '2px solid #E50914',
                        'opacity': 0,
                        'transition': 'all 0.3s ease-in-out',
                    },
                    children=[
                        html.Button(
                            '✕',
                            id='close-popup',
                            n_clicks=0,
                            style={
                                'position': 'absolute',
                                'top': '20px',
                                'right': '20px',
                                'backgroundColor': '#E50914',
                                'color': 'white',
                                'border': 'none',
                                'borderRadius': '50%',
                                'width': '45px',
                                'height': '45px',
                                'fontSize': '1.5rem',
                                'cursor': 'pointer',
                                'zIndex': 10000,
                                'fontWeight': 'bold',
                                'display': 'flex',
                                'alignItems': 'center',
                                'justifyContent': 'center',
                                'boxShadow': '0 4px 12px rgba(0,0,0,0.4)',
                                'transition': 'transform 0.2s ease',
                            }
                        ),
                        html.Div(id='popup-details', style={'padding': '50px 40px'})
                    ]
                )
            ]
        ),

    ],
    style={
        'backgroundColor': 'var(--background-color)',
        'minHeight': '100vh',
        'padding': '50px 10px',
        'overflowX': 'hidden'
    }
)


# ----------- CALLBACKS ------------

# Main filter callback
@callback(
    Output('content-table', 'data'),
    Output('content-count', 'children'),
    Output('quick-stats', 'children'),
    Output('stats-button', 'children'),
    Output('stats-button', 'style'),
    Input('search-title', 'value'),
    Input('filter-type', 'value'),
    Input('filter-country', 'value'),
    Input('filter-genre', 'value'),
    Input('filter-rating', 'value'),
    Input('year-range', 'value'),
    Input('sort-by', 'value'),
    Input('stats-button', 'n_clicks'),
)
def update_table(search, type_, country, genre, rating, year_range, sort_by, stats_clicks):
    filtered_df = df.copy()

    # Apply filters
    if search:
        mask = (
            filtered_df['title'].str.contains(search, case=False, na=False) |
            filtered_df['director'].str.contains(search, case=False, na=False) |
            filtered_df['cast'].str.contains(search, case=False, na=False) |
            filtered_df['listed_in'].str.contains(search, case=False, na=False)
        )
        filtered_df = filtered_df[mask]
    
    if type_ and type_ != 'all':
        filtered_df = filtered_df[filtered_df['type'] == type_]
    
    if country and country != 'all':
        filtered_df = filtered_df[filtered_df['country'].str.contains(country, case=False, na=False)]
    
    if genre and genre != 'all':
        filtered_df = filtered_df[filtered_df['listed_in'].str.contains(genre, case=False, na=False)]
    
    if rating and rating != 'all':
        filtered_df = filtered_df[filtered_df['rating'] == rating]
    
    if year_range:
        filtered_df = filtered_df[
            (filtered_df['release_year'] >= year_range[0]) &
            (filtered_df['release_year'] <= year_range[1])
        ]

    # Apply sorting
    if sort_by == 'title_asc':
        filtered_df = filtered_df.sort_values('title', ascending=True)
    elif sort_by == 'title_desc':
        filtered_df = filtered_df.sort_values('title', ascending=False)
    elif sort_by == 'year_desc':
        filtered_df = filtered_df.sort_values('release_year', ascending=False)
    elif sort_by == 'year_asc':
        filtered_df = filtered_df.sort_values('release_year', ascending=True)
    elif sort_by == 'added_desc':
        filtered_df = filtered_df.sort_values('date_added', ascending=False, na_position='last')
    elif sort_by == 'added_asc':
        filtered_df = filtered_df.sort_values('date_added', ascending=True, na_position='last')

    # Prepare display data
    display_df = filtered_df[['title', 'type', 'director', 'cast', 'country', 'release_year', 'rating', 'duration', 'listed_in']].copy()
    
    # Truncate long text for display
    for col in ['director', 'cast', 'listed_in']:
        display_df[col] = display_df[col].apply(lambda x: x[:50] + '...' if len(str(x)) > 50 else x)

    count_text = f"Showing {len(filtered_df):,} titles"
    
    # Determine if stats should be shown (toggle behavior)
    show_stats = stats_clicks and stats_clicks % 2 == 1
    
    # Quick stats (show when button clicked - toggle)
    stats_panel = None
    button_text = '📊 View Quick Stats'
    button_style = {
        'padding': '10px 24px',
        'backgroundColor': '#333',
        'color': 'white',
        'border': '1px solid #555',
        'borderRadius': '6px',
        'cursor': 'pointer',
        'fontSize': '0.95rem',
        'fontWeight': '600',
        'boxShadow': '0 2px 8px rgba(0,0,0,0.3)',
        'transition': 'all 0.3s ease'
    }
    
    if show_stats:
        button_text = '📊 Hide Quick Stats'
        button_style['backgroundColor'] = '#E50914'
        button_style['border'] = '1px solid #E50914'
        
        movies = len(filtered_df[filtered_df['type'] == 'Movie'])
        shows = len(filtered_df[filtered_df['type'] == 'TV Show'])
        avg_year = filtered_df['release_year'].mean() if len(filtered_df) > 0 else 0
        top_country = filtered_df['country'].str.split(',').explode().str.strip().value_counts().index[0] if len(filtered_df) > 0 else 'N/A'
        
        stats_panel = html.Div([
            html.Div([
                html.Div([
                    html.H3(f"{movies:,}", style={'color': '#E50914', 'margin': '0', 'fontSize': '1.8rem'}),
                    html.P("Movies", style={'color': '#ccc', 'margin': '0', 'fontSize': '0.9rem'})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': 'var(--background-color)', 'borderRadius': '8px'}),
                
                html.Div([
                    html.H3(f"{shows:,}", style={'color': '#E50914', 'margin': '0', 'fontSize': '1.8rem'}),
                    html.P("TV Shows", style={'color': '#ccc', 'margin': '0', 'fontSize': '0.9rem'})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': 'var(--background-color)', 'borderRadius': '8px'}),
                
                html.Div([
                    html.H3(f"{avg_year:.0f}", style={'color': '#E50914', 'margin': '0', 'fontSize': '1.8rem'}),
                    html.P("Avg Release Year", style={'color': '#ccc', 'margin': '0', 'fontSize': '0.9rem'})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': 'var(--background-color)', 'borderRadius': '8px'}),
                
                html.Div([
                    html.H3(f"{top_country}", style={'color': '#E50914', 'margin': '0', 'fontSize': '1.5rem'}),
                    html.P("Top Country", style={'color': '#ccc', 'margin': '0', 'fontSize': '0.9rem'})
                ], style={'textAlign': 'center', 'padding': '15px', 'backgroundColor': 'var(--background-color)', 'borderRadius': '8px'}),
            ], style={'display': 'grid', 'gridTemplateColumns': 'repeat(4, 1fr)', 'gap': '15px', 'width': '95%', 'margin': 'auto'})
        ], style={'marginBottom': '25px'})

    return display_df.to_dict('records'), count_text, stats_panel, button_text, button_style


# Reset filters callback
@callback(
    Output('search-title', 'value'),
    Output('filter-type', 'value'),
    Output('filter-country', 'value'),
    Output('filter-genre', 'value'),
    Output('filter-rating', 'value'),
    Output('year-range', 'value'),
    Input('reset-button', 'n_clicks'),
    prevent_initial_call=True
)
def reset_filters(n_clicks):
    return '', 'all', 'all', 'all', 'all', [int(df['release_year'].min()), int(df['release_year'].max())]


# Export callback
@callback(
    Output('download-data', 'data'),
    Input('export-button', 'n_clicks'),
    State('content-table', 'data'),
    prevent_initial_call=True
)
def export_data(n_clicks, table_data):
    if n_clicks and table_data:
        export_df = pd.DataFrame(table_data)
        return dcc.send_data_frame(export_df.to_csv, f"netflix_filtered_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)
    return None


# ------------------ SHOW MOVIE DETAILS POPUP ------------------
@callback(
    Output('detail-popup', 'style'),
    Output('popup-details', 'children'),
    Output('popup-overlay', 'style'),
    Output('popup-content', 'style'),
    Input('content-table', 'active_cell'),
    Input('close-popup', 'n_clicks'),
    State('content-table', 'data'),
    State('detail-popup', 'style'),
    prevent_initial_call=True
)
def show_movie_details(active_cell, close_click, table_data, current_popup_style):
    ctx = dash.callback_context

    # Default styles
    overlay_hidden = {
        'position': 'fixed', 'top': 0, 'left': 0, 'width': '100%', 'height': '100%',
        'backgroundColor': 'rgba(0, 0, 0, 0.85)', 'zIndex': 9998,
        'backdropFilter': 'blur(5px)', 'opacity': 0,
        'transition': 'opacity 0.3s ease-in-out',
    }
    overlay_visible = {**overlay_hidden, 'opacity': 1}
    
    content_hidden = {
        'position': 'fixed', 'top': '50%', 'left': '50%',
        'transform': 'translate(-50%, -50%) scale(0.7)',
        'backgroundColor': 'var(--background-color)', 'padding': '0', 'borderRadius': '15px',
        'boxShadow': '0 10px 40px rgba(229, 9, 20, 0.5)', 'zIndex': 9999,
        'width': '90%', 'maxWidth': '900px', 'maxHeight': '85vh',
        'overflowY': 'auto', 'border': '2px solid #E50914',
        'opacity': 0, 'transition': 'all 0.3s ease-in-out',
    }
    content_visible = {
        **content_hidden,
        'transform': 'translate(-50%, -50%) scale(1)',
        'opacity': 1,
    }

    # Close button pressed
    if ctx.triggered_id == 'close-popup' or not active_cell:
        return {'display': 'none'}, None, overlay_hidden, content_hidden

    row_idx = active_cell.get('row')
    if row_idx is None or row_idx >= len(table_data):
        return {'display': 'none'}, None, overlay_hidden, content_hidden

    record = table_data[row_idx]

    # Find full data from original df
    full = df[df['title'] == record['title']].iloc[0]

    details = html.Div([
        # Header with title and type badge
        html.Div([
            html.H2(
                full['title'], 
                style={
                    'color': 'white',
                    'marginBottom': '10px',
                    'fontSize': '2rem',
                    'fontWeight': '700',
                    'lineHeight': '1.2'
                }
            ),
            html.Span(
                full['type'],
                style={
                    'display': 'inline-block',
                    'backgroundColor': '#E50914',
                    'color': 'white',
                    'padding': '5px 15px',
                    'borderRadius': '20px',
                    'fontSize': '0.85rem',
                    'fontWeight': '600',
                    'marginBottom': '20px'
                }
            ),
        ]),
        
        # Description
        html.Div([
            html.P(
                full['description'],
                style={
                    'color': '#ddd',
                    'lineHeight': '1.8',
                    'fontSize': '1.05rem',
                    'marginBottom': '30px',
                    'paddingBottom': '20px',
                    'borderBottom': '1px solid #333'
                }
            ),
        ]),
        
        # Details grid
        html.Div([
            # Left column
            html.Div([
                html.Div([
                    html.Div('⭐ Rating', style={'color': '#999', 'fontSize': '0.85rem', 'marginBottom': '5px'}),
                    html.Div(full['rating'], style={'color': 'white', 'fontSize': '1.1rem', 'fontWeight': '600'})
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    html.Div('📅 Release Year', style={'color': '#999', 'fontSize': '0.85rem', 'marginBottom': '5px'}),
                    html.Div(str(full['release_year']), style={'color': 'white', 'fontSize': '1.1rem', 'fontWeight': '600'})
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    html.Div('⏱ Duration', style={'color': '#999', 'fontSize': '0.85rem', 'marginBottom': '5px'}),
                    html.Div(full['duration'], style={'color': 'white', 'fontSize': '1.1rem', 'fontWeight': '600'})
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    html.Div('🌍 Country', style={'color': '#999', 'fontSize': '0.85rem', 'marginBottom': '5px'}),
                    html.Div(full['country'], style={'color': 'white', 'fontSize': '1.05rem', 'lineHeight': '1.5'})
                ], style={'marginBottom': '20px'}),
            ], style={'flex': '1', 'paddingRight': '30px'}),
            
            # Right column
            html.Div([
                html.Div([
                    html.Div('🎭 Genres', style={'color': '#999', 'fontSize': '0.85rem', 'marginBottom': '5px'}),
                    html.Div(full['listed_in'], style={'color': 'white', 'fontSize': '1.05rem', 'lineHeight': '1.5'})
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    html.Div('🎥 Director', style={'color': '#999', 'fontSize': '0.85rem', 'marginBottom': '5px'}),
                    html.Div(full['director'], style={'color': 'white', 'fontSize': '1.05rem', 'lineHeight': '1.5'})
                ], style={'marginBottom': '20px'}),
                
                html.Div([
                    html.Div('👥 Cast', style={'color': '#999', 'fontSize': '0.85rem', 'marginBottom': '5px'}),
                    html.Div(full['cast'], style={'color': 'white', 'fontSize': '1.05rem', 'lineHeight': '1.5'})
                ], style={'marginBottom': '0'}),
            ], style={'flex': '1'}),
        ], style={
            'display': 'flex',
            'gap': '20px',
            'marginTop': '10px'
        }),
    ], style={'color': '#fff'})

    return {'display': 'block'}, details, overlay_visible, content_visible