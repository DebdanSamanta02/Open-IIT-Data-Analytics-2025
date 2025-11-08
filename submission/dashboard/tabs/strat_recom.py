from dash import html

def card_style():
    return {
        'background': 'var(--graph-color)',
        'borderRadius': '16px',
        'padding': '30px',
        'marginTop': '40px',
        'marginBottom': '40px',
        'boxShadow': '0 4px 30px rgba(0,0,0,0.4)'
    }

layout = html.Div(
    children=[
        html.H1(
            [
                html.Span("S", style={'color': '#E50914'}),
                "trategic ",
                html.Span("R", style={'color': '#E50914'}),
                "ecommendations"
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
            "Data-driven actions to optimize content strategy, global expansion, and audience engagement.",
            style={
                'textAlign': 'center',
                'fontSize': '1.1rem',
                'marginBottom': '50px',
                'color': 'var(--font-color)'
            }
        ),
        html.Div(
            [
                html.H3("Content Acquisition Priorities", style={'color': '#E50914'}),
                html.Ul([
                    html.Li("Pivot to a Retention-Focused Model: Prioritize Originals and TV series that reduce churn rather than mass licensed volume."),
                    html.Li("De-risk TV Production with the \"Limited Series\" Model: Pivot to 1–2.5 season limited series to reduce cost risk and increase high-impact creative “at-bats.”"),
                    html.Li("Formalize a Bifurcated Talent Pipeline: Establish two distinct talent pipelines to separately manage high-volume TV specialists and project-based prestige film creators."),
                    html.Li("Invest in a Low-Cost \"Classic Vault\": Leverage rapidly growing classic films to build a low-cost, high-loyalty content vault competitors can’t easily replicate."),
                    html.Li("Fix the TV Director Data Gap: Launch a data governance effort to fix director credit gaps in TV content, removing a major blind spot in talent analysis.")
                ], style={'color': 'var(--font-color)', 'fontSize': '1.1rem', 'lineHeight': '1.6'})
            ],
            style=card_style()
        ),
        html.Div(
            [
                html.H3("Geographic Expansion Opportunities", style={'color': '#E50914'}),
                html.Ul([
                    html.Li("Prioritize global-first growth by scaling international co-productions with export-ready IP."),
                    html.Li("Pursue selective, high-impact blockbusters in China to capture outsized revenue per title."),
                    html.Li("Back mature markets like Japan and South Korea with increased investment in high-demand Korean TV."),
                    html.Li("Defend share in Southeast Asia through strategic local originals and leverage of Korean content pull."),
                    html.Li("Build cross-border scale in Eastern Europe and MENA with regional partnerships and culturally aligned premium productions.")
                ], style={'color': 'var(--font-color)', 'fontSize': '1.1rem', 'lineHeight': '1.6'})
            ],
            style=card_style()
        ),
        html.Div(
            [
                html.H3("Genre Diversification Strategies", style={'color': '#E50914'}),
                html.Ul([
                    html.Li("Accelerate the pivot to mass-market binge genres by reallocating away from prestige/educational content toward high-growth Romantic and Action & Adventure titles."),
                    html.Li("Dominate high-loyalty niches by investing in low-cost genres like Anime Features, Classic & Cult TV, and Faith & Spirituality."),
                    html.Li("Use metadata-driven insights to surface emerging genre pairings and boost discovery through curated landing pages.")
                ], style={'color': 'var(--font-color)', 'fontSize': '1.1rem', 'lineHeight': '1.6'})
            ],
            style=card_style()
        ),
        html.Div(
            [
                html.H3("Audience Targeting Insights", style={'color': '#E50914'}),
                html.Ul([
                    html.Li("Defend the all-household position by maintaining a balanced 46.5% mature-content mix to serve adults while anchoring family utility."),
                    html.Li("Close the family-content gap with low-cost kids’ miniseries and strategic licensing of high-value family IP to reduce churn risk."),
                    html.Li("Target micro-genres to engineer emotional tone and enable marketing/production teams to act on emerging narrative trends."),
                    html.Li("Use NLP-driven subgenre signals to better brief creators and shape content toward feel-good or high-conflict demand clusters."),
                    html.Li("Adapt tagging strategy by market maturity with dense multi-tagging in emerging markets and precise minimal tagging in mature ones.")
                ], style={'color': 'var(--font-color)', 'fontSize': '1.1rem', 'lineHeight': '1.6'})
            ],
            style=card_style()
        ),
        html.Div(
            [
                html.H3("Competitive Positioning Recommendations", style={'color': '#E50914'}),
                html.Ul([
                    html.Li("Reinforce the middle-ground value position to sit between family-only and adult-premium competitors."),
                    html.Li("Execute a dual-flank (“pincer”) defense by expanding family content and strengthening high-quality adult differentiators."),
                    html.Li("Use curated themed hubs (e.g., Anime, Classics) to counter niche specialists with superior catalog depth and value.")
                ], style={'color': 'var(--font-color)', 'fontSize': '1.1rem', 'lineHeight': '1.6'})
            ],
            style=card_style()
        ),
        html.Div(
            [
                html.H3("Investment Allocation Suggestions", style={'color': '#E50914'}),
                html.Ul([
                    html.Li("INVEST/MAINTAIN high-growth, high-quality genres like Korean TV and Docuseries to reinforce clear portfolio winners."),
                    html.Li("UPGRADE/MAINTAIN high-growth, low-quality segments by improving talent and creative quality in Stand-Up Comedy & Talk and TV Horror."),
                    html.Li("OPTIMIZE/MILK low-growth, high-quality anchors such as Classic & Cult TV and select Sci-Fi with light, targeted investment."),
                    html.Li("DIVEST/EXIT low-growth, low-quality genres to free budget from underperforming Independent Movies and weak Horror segments."),
                    html.Li("Fund a Strategic Asset program and rebalance promo spend to retain high-impact creators and maximize marketing on mass-market binge genres while maintaining a prestige halo.")
                ], style={'color': 'var(--font-color)', 'fontSize': '1.1rem', 'lineHeight': '1.6'})
            ],
            style=card_style()
        )
    ],
    className='strat-recom',
    style={
        'backgroundColor': 'var(--background-color)',
        'minHeight': '100vh',
        'padding': '60px 20px',
        'fontFamily': 'Segoe UI, sans-serif',
        'overflow': 'scroll-y'
    }
)