from dash import html

layout = html.Div(
    children=[
        html.H1(
            children=[
                html.Span("S", style={'color': 'var(--red)'}),
                "trategic ",
                html.Span("R", style={'color': 'var(--red)'}),
                "ecommendations"
            ]
        ),
        html.P("This is the Executive Overview tab content.")
    ],
    className='strat-recom'
)