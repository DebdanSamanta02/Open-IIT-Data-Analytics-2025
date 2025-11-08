from dash import Dash, html, dcc, dash_table, Output, Input, callback
import pandas as pd
import plotly.express as px
import numpy as np


# =====================================
# LOAD & PROCESS DATA
# =====================================

df = pd.read_csv('./data/netflix_titles.csv')

# Type counts
type_counts = df['type'].value_counts().reset_index()
type_counts.columns = ['Type', 'Count']

# Year counts
year_counts = df['release_year'].value_counts().sort_index().reset_index()
year_counts.columns = ['Release Year', 'Count']

# Country counts
country_list = df["country"].str.split(",").explode().str.strip()
country_counts = country_list.value_counts().head(20)
country_normalized_counts = country_list.value_counts(normalize=True)
shannon_index = -np.sum(country_normalized_counts * np.log2(country_normalized_counts))

# CAGR
titles = df['title'].count()
start_value = year_counts['Count'].iloc[0]
end_value = year_counts['Count'].iloc[-1]
num_years = len(year_counts) - 1
CAGR = ((end_value / start_value) ** (1 / num_years) - 1) * 100

# Language detection
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

lang_df = pd.DataFrame(list(language_counts.items()), columns=['Language', 'Count']).sort_values(by='Count', ascending=False)

# Rating category classification
rating_groups = {
    'Kids': ['TV-Y', 'TV-Y7', 'TV-G', 'G', 'TV-Y7-FV'],
    'Teens': ['TV-PG', 'PG', 'PG-13', 'TV-14'],
    'Adults': ['TV-MA', 'R', 'NC-17', 'NR', 'UR']
}
def cat(x):
    for k, v in rating_groups.items():
        if x['rating'] in v:
            return k
    return 'Unknown'
df['category'] = df.apply(lambda x: cat(x), axis=1)


# =====================================
# GEOGRAPHIC VISUALS
# =====================================

# WORLD MAP (Reds)
world_counts = country_list.value_counts().reset_index()
world_counts.columns = ['Country', 'Titles']
world_counts['Rank'] = world_counts['Titles'].rank(ascending=False).astype(int)

fig_world = px.choropleth(
    world_counts,
    locations='Country',
    locationmode='country names',
    color='Titles',
    color_continuous_scale='Reds',
    hover_name='Country',
    hover_data={'Titles': ':,', 'Rank': True, 'Country': False}
)
fig_world.update_layout(
    title={'text': 'Netflix Titles by Each Country', 'x': 0.5, 'xanchor': 'center', 'font': {'size': 22}},
    height=600,
    margin=dict(l=0, r=0, t=40, b=0),
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    geo=dict(
        showframe=False,
        showcoastlines=True,
        projection_type='natural earth',
        showocean=True,
        oceancolor='rgba(40,40,40,0.90)'
    )
)

# COUNTRY BAR (Reds)
fig_country_bar = px.bar(
    x=country_counts.values[::-1],
    y=country_counts.index[::-1],
    orientation='h',
    color=country_counts.values[::-1],
    color_continuous_scale='Reds',
    title='Country Comparison: Top 20 Producers'
)
fig_country_bar.update_layout(yaxis=dict(autorange='reversed'))

# =====================================
# REGIONAL DEEP DIVES
# =====================================

df_small = df[['country', 'listed_in']].dropna()
df_exploded = df_small.assign(
    country_list=df_small['country'].str.split(','),
    genres_list=df_small['listed_in'].str.split(',')
).explode('country_list').explode('genres_list')
df_exploded['country'] = df_exploded['country_list'].str.strip()
df_exploded['genre'] = df_exploded['genres_list'].str.strip()
df_agg = df_exploded.groupby(['country', 'genre']).size().reset_index(name='Count')

top_countries = df_agg.groupby('country')['Count'].sum().nlargest(15).index
top_genres = df_agg.groupby('genre')['Count'].sum().nlargest(15).index
df_filtered = df_agg[df_agg['country'].isin(top_countries) & df_agg['genre'].isin(top_genres)]
df_pivot = df_filtered.pivot(index='country', columns='genre', values='Count').fillna(0)

# HEATMAP (YlOrRd as before)
fig_heatmap = px.imshow(
    df_pivot,
    text_auto=True,
    aspect="auto",
    color_continuous_scale='YlOrRd',
    title="Genre Concentration by Country (Top 15 Countries & Genres)",
    labels={'color': 'Titles'}
)

# =====================================
# REGION-LEVEL VIEWS
# =====================================

region_map = {
    'North America': ['United States', 'Canada', 'Mexico'],
    'Europe': ['United Kingdom', 'France', 'Germany', 'Spain', 'Italy', 'Poland', 'Sweden'],
    'Asia-Pacific': ['India', 'Japan', 'South Korea', 'China', 'Thailand', 'Indonesia', 'Singapore'],
    'Latin America': ['Brazil', 'Argentina', 'Chile', 'Colombia', 'Peru'],
    'Middle East & Africa': ['United Arab Emirates', 'Egypt', 'South Africa', 'Nigeria', 'Turkey', 'Saudi Arabia']
}
country_to_region = {c: r for r, countries in region_map.items() for c in countries}

df_region = df.copy()
df_region['country_list'] = df_region['country'].dropna().str.split(',')
df_region = df_region.explode('country_list')
df_region['country_list'] = df_region['country_list'].str.strip()
df_region['region'] = df_region['country_list'].map(country_to_region)
df_region = df_region.dropna(subset=['region'])

# REGIONAL CONTENT MIX (Netflix reds)
region_mix = df_region.groupby(['region', 'type']).size().reset_index(name='Count')
fig_region_bar = px.bar(
    region_mix,
    x='region',
    y='Count',
    color='type',
    barmode='group',
    color_discrete_sequence=['#E50914', '#B20710'],
    title='Regional Content Mix: Movies vs. TV Shows'
)

# REGIONAL TREND (Reds sequence)
df_region['date_added'] = pd.to_datetime(df_region['date_added'], errors='coerce')
df_region['year_added'] = df_region['date_added'].dt.year
region_trend = df_region.groupby(['region', 'year_added']).size().reset_index(name='Count')
region_trend = region_trend[region_trend['year_added'] > 2010]

fig_region_trend = px.line(
    region_trend,
    x='year_added',
    y='Count',
    color='region',
    markers=True,
    title='Content Growth Over Time by Region',
    color_discrete_sequence=px.colors.sequential.Reds
)

# TOP GENRES PER REGION (Reds sequence)
df_genre = df_region.copy()
df_genre = df_genre.explode('listed_in')
df_genre['listed_in'] = df_genre['listed_in'].str.strip()
region_genre = df_genre.groupby(['region', 'listed_in']).size().reset_index(name='Count')
region_genre_top = region_genre.sort_values(['region', 'Count'], ascending=[True, False]).groupby('region').head(5)

fig_region_genre = px.bar(
    region_genre_top,
    x='region',
    y='Count',
    color='listed_in',
    title='Top 5 Genres per Region',
    barmode='stack',
    color_discrete_sequence=px.colors.sequential.Reds
)

# =====================================
# PRODUCTION HUBS (Reds)
# =====================================

df_prod = df.copy()
df_prod['country'] = df_prod['country'].dropna().str.split(',')
df_prod = df_prod.explode('country')
df_prod['country'] = df_prod['country'].str.strip()

top_hubs = df_prod['country'].value_counts().reset_index().head(15)
top_hubs.columns = ['Country', 'Total Titles']
fig_prod_hubs = px.bar(
    top_hubs[::-1],
    x='Total Titles',
    y='Country',
    orientation='h',
    color='Total Titles',
    color_continuous_scale='Reds',
    title='Top Production Hubs on Netflix'
)

# =====================================
# MARKET OPPORTUNITIES (Reds)
# =====================================

df_market = df.copy()
df_market['country'] = df_market['country'].dropna().str.split(',')
df_market = df_market.explode('country')
df_market['country'] = df_market['country'].str.strip()
df_market['date_added'] = pd.to_datetime(df_market['date_added'], errors='coerce')
df_market['year_added'] = df_market['date_added'].dt.year

country_year = df_market.groupby(['country', 'year_added']).size().reset_index(name='Count')
country_year = country_year[country_year['year_added'] >= 2015]

growth_data = []
for country, group in country_year.groupby('country'):
    group = group.sort_values('year_added')
    if len(group) > 3 and group['Count'].iloc[0] > 0:
        start_val = group['Count'].iloc[0]
        end_val = group['Count'].iloc[-1]
        years = group['year_added'].iloc[-1] - group['year_added'].iloc[0]
        if years > 0:
            cagr = ((end_val / start_val) ** (1 / years) - 1) * 100
            growth_data.append((country, cagr, end_val))

df_growth = pd.DataFrame(growth_data, columns=['Country', 'CAGR (%)', 'Recent Titles'])

fig_market = px.scatter(
    df_growth,
    x='Recent Titles',
    y='CAGR (%)',
    text='Country',
    color='CAGR (%)',
    color_continuous_scale='Reds',
    title='Market Opportunities: Growth vs. Current Presence'
)
fig_market.update_traces(textposition='top center', marker=dict(size=10, line=dict(width=1, color='white')))


# =====================================
# STYLES AND LAYOUT
# =====================================

def card_style():
    return {
        'flex': '1',
        'margin': '10px',
        'background': 'var(--graph-color)',
        'borderRadius': 'var(--card-radius)',
        'padding': '35px',
        'boxShadow': 'var(--card-shadow)',
        'minWidth': '40%',
        'transition': f'all var(--transition-speed) ease',
        'backdropFilter': 'blur(10px)',
        'WebkitBackdropFilter': 'blur(10px)'
    }

layout = html.Div(
    [
        html.H1(
            [html.Span("G", style={'color': '#E50914'}), "eographic ", html.Span("I", style={'color': '#E50914'}), "nsights"],
            style={
                'textAlign': 'center',
                'color': 'var(--font-color)',
                'fontWeight': 'var(--heading-weight)',
                'fontSize': '2.5rem',
                'letterSpacing': '-0.02em',
                'marginBottom': '15px',
                'transition': 'color var(--transition-speed) ease'
            }
        ),
        html.P(
            "Comparative overview of Netflix's performance across countries.",
            style={
                'textAlign': 'center',
                'color': 'var(--font-color)',
                'marginBottom': '45px',
                'fontSize': '1.1rem',
                'fontWeight': 'var(--subheading-weight)',
                'maxWidth': '800px',
                'margin': '0 auto 45px auto',
                'lineHeight': '1.5'
            }
        ),

        dcc.Graph(id='fig_world', figure=fig_world, config={'displayModeBar': False}, style=card_style()),
        dcc.Graph(id='fig_country_bar', figure=fig_country_bar, config={'displayModeBar': False}, style=card_style()),
        dcc.Graph(id='fig_heatmap', figure=fig_heatmap, config={'displayModeBar': False}, style=card_style()),
        dcc.Graph(id='fig_region_bar', figure=fig_region_bar, config={'displayModeBar': False}, style=card_style()),
        dcc.Graph(id='fig_region_trend', figure=fig_region_trend, config={'displayModeBar': False}, style=card_style()),
        dcc.Graph(id='fig_region_genre', figure=fig_region_genre, config={'displayModeBar': False}, style=card_style()),
        dcc.Graph(id='fig_prod_hubs', figure=fig_prod_hubs, config={'displayModeBar': False}, style=card_style()),
        dcc.Graph(id='fig_market', figure=fig_market, config={'displayModeBar': False}, style=card_style()),
    ],
    className='geo-insights',
    style={'backgroundColor': 'var(--background-color)', 'padding': '60px 20px'}
)


# =====================================
# CALLBACKS (Theme-safe: does not touch color sequences)
# =====================================

def _apply_theme(fig, theme):
    if theme == "light":
        text_color = "#0F0F0F"
        bg_color = "#FFFFFF"
        grid_color = "var(--grid-color-light)"
        ocean_color = "rgba(240,240,240,0.90)"
    else:
        text_color = "#FFFFFF"
        bg_color = "#0F0F0F"
        grid_color = "var(--grid-color-dark)"
        ocean_color = "rgba(40,40,40,0.90)"
    
    # Update ocean color for maps
    if 'geo' in fig.layout:
        fig.layout.geo.update(oceancolor=ocean_color)

    fig.update_layout(
        paper_bgcolor=bg_color,
        plot_bgcolor=bg_color,
        font=dict(color=text_color),
        title_font=dict(color=text_color),
        legend=dict(font=dict(color=text_color)),
        xaxis=dict(showgrid=True, gridcolor=grid_color, color=text_color),
        yaxis=dict(showgrid=True, gridcolor=grid_color, color=text_color)
    )
    return fig


@callback(
    [Output(id, "figure") for id in [
        "fig_world", "fig_country_bar", "fig_heatmap",
        "fig_region_bar", "fig_region_trend",
        "fig_region_genre", "fig_prod_hubs", "fig_market"
    ]],
    Input("current-theme", "data")
)
def update_all_themes(theme):
    figs = [fig_world, fig_country_bar, fig_heatmap, fig_region_bar,
            fig_region_trend, fig_region_genre, fig_prod_hubs, fig_market]
    return [_apply_theme(f, theme) for f in figs]
