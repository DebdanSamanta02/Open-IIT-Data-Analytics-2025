import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
from tabs import content, creator_talent, exec_overview, genre_intelligence, geo_insights, strat_recom, trend

app = dash.Dash(__name__, 
                external_stylesheets=['/assets/style.css'],
                title='Team 13',
                suppress_callback_exceptions=True)

app.layout = dbc.Container([

    # themer
    html.Div(id='theme-output-link', children=[
        html.Link(rel='stylesheet', href='/assets/dark.css', id='theme-link')
    ], style={'display': 'none'}),

    # page container
    html.Div(
    children=[     
        # --- NAVBAR (Fixed Left Sidebar) ---
        html.Div(
            children=[
                html.Div(style={'height': '1px'}),  # small spacing
                html.Div(
                    children=[
                        dbc.NavItem(html.Img(src=app.get_asset_url('Tab/Tab1.svg'), id='tab-icon-1', className='tab-icon')),
                        dbc.NavItem(html.Img(src=app.get_asset_url('Tab/Tab2.svg'), id='tab-icon-2', className='tab-icon')),
                        dbc.NavItem(html.Img(src=app.get_asset_url('Tab/Tab3.svg'), id='tab-icon-3', className='tab-icon')),
                        dbc.NavItem(html.Img(src=app.get_asset_url('Tab/Tab4.svg'), id='tab-icon-4', className='tab-icon')),
                        dbc.NavItem(html.Img(src=app.get_asset_url('Tab/Tab5.svg'), id='tab-icon-5', className='tab-icon')),
                        dbc.NavItem(html.Img(src=app.get_asset_url('Tab/Tab6.svg'), id='tab-icon-6', className='tab-icon')),
                        dbc.NavItem(html.Img(src=app.get_asset_url('Tab/Tab7.svg'), id='tab-icon-7', className='tab-icon')),
                    ],
                    className='tabs',
                    style={
                        'display': 'flex',
                        'flexDirection': 'column',
                        'alignItems': 'center',
                        'gap': '20px',
                        'marginTop': '40px'
                    }
                ),
                dbc.NavItem(
                    html.Img(
                        src=app.get_asset_url('Theme.svg'),
                        id='theme-switcher',
                        className='theme-switcher-icon',
                        style={
                            'width': '30px',
                            'height': '30px',
                            'cursor': 'pointer',
                            'marginTop': '40px'
                        }
                    )
                ),
            ],
            className='nav-icons',
            style={
                'position': 'fixed',        # stays fixed on screen
                'top': '0',
                'left': '0',
                'height': '100vh',          # full height
                'width': '40px',            # sidebar width
                'backgroundColor': 'var(--background-color)',
                'display': 'flex',
                'flexDirection': 'column',
                'alignItems': 'center',
                'justifyContent': 'flex-start',
                'paddingTop': '20px',
                'paddingBottom': '20px',
                'boxShadow': '2px 0 10px rgba(0,0,0,0.3)',
                'zIndex': '1000'
            }
        ),

        # --- PAGE CONTENT (shifted right of sidebar) ---
        html.Div(
            id='page-content',
            style={
                'marginLeft': '100px',       # creates space for fixed sidebar
                'padding': '20px',
                'width': 'calc(100% - 100px)',
                'overflowY': 'auto',
                'minHeight': '100vh',
                'backgroundColor': 'var(--background-color)',
            }
        ),
    ],
    className='container'
)
,
    dcc.Store(id='current-theme', data='dark'),
], fluid=True, className="dashboard-container")


# hover over nav items
tab_icon_outputs = [Output(f'tab-icon-{i}', 'className') for i in range(1, 8)]
tab_icon_inputs = [Input(f'tab-icon-{i}', 'n_clicks') for i in range(1, 8)]

@app.callback(
    tab_icon_outputs,
    tab_icon_inputs,
)
def update_active_tab_style(*args):
    
    ctx = dash.callback_context
    triggered_id = ctx.triggered_id
    
    base_class = 'tab-icon'
    new_class_names = []
    
    for i in range(1, 8):
        icon_id = f'tab-icon-{i}'
        
        if not triggered_id and i == 1:
            new_class_names.append(f'{base_class} active')
        elif triggered_id == icon_id:
            new_class_names.append(f'{base_class} active')
        else:
            new_class_names.append(base_class)
            
    return new_class_names


# nav router
@app.callback(
    Output('page-content', 'children'),
    [Input('tab-icon-1', 'n_clicks'),
     Input('tab-icon-2', 'n_clicks'),
     Input('tab-icon-3', 'n_clicks'),
     Input('tab-icon-4', 'n_clicks'),
     Input('tab-icon-5', 'n_clicks'),
     Input('tab-icon-6', 'n_clicks'),
     Input('tab-icon-7', 'n_clicks')]
)
def render_page_content(*args):
    ctx = dash.callback_context

    if not ctx.triggered_id:
        tab_id = '1'
    else:
        tab_id = ctx.triggered_id.replace('tab-icon-', '')

    if tab_id == '1':
        return exec_overview.layout
    elif tab_id == '2':
        return content.layout
    elif tab_id == '3':
        return trend.layout
    elif tab_id == '4':
        return geo_insights.layout
    elif tab_id == '5':
        return genre_intelligence.layout
    elif tab_id == '6':
        return creator_talent.layout
    elif tab_id == '7':
        return strat_recom.layout
    return html.H2("Error, Please reload.")


# theme switcher
@app.callback(
    Output('current-theme', 'data'),
    Output('theme-link', 'href'),
    Input('theme-switcher', 'n_clicks'),
    State('current-theme', 'data'),
    prevent_initial_call=True
)
def switch_theme(n_clicks, current_theme):
    if current_theme == 'dark':
        new_theme = 'light'
        new_css_href = '/assets/light.css'
    else:
        new_theme = 'dark'
        new_css_href = '/assets/dark.css'
        
    return new_theme, new_css_href

trend.register_trend_callbacks(app)

if __name__ == '__main__':
    app.run(debug=True)

server = app.server