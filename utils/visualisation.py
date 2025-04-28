import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import pandas as pd
from wordcloud import WordCloud
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import base64
from io import BytesIO
import numpy as np
from scipy.stats import gaussian_kde
import base64
from pathlib import Path

def generate_map_plot(df, selected_type):
    if selected_type != 'All':
        df = df[df['accident_type'] == selected_type]

    df = df.copy()
    df['hover_text'] = df.apply(lambda r: (
        f"Date: {r['accident_date'].strftime('%Y-%m-%d')}<br>"
        f"Make: {r.get('aircraft_make','Unknown')}<br>"
        f"City: {r.get('city','')}<br>"
        f"Fatalities: {int(r.get('fatalities',0))}<br>"
        f"Type: {r['accident_type']}<br>"
        f"Cause: {r.get('cause_category','')}"
    ), axis=1)

    fig = go.Figure()

    fig.update_geos(
        projection_type="orthographic",
        projection_rotation=dict(lat=40, lon=-97, roll=0),  # tilt a bit for drama
        showcountries=True, showcoastlines=True, showocean=True,
        landcolor="lightgray", oceancolor="lightblue",
        lakecolor="lightblue", coastlinecolor="grey",
        showlakes=True
    )

    fig.add_trace(go.Scattergeo(
        lon=df['longitude'],
        lat=df['latitude'],
        hovertext=df['hover_text'],
        hoverinfo="text",
        mode="markers",
        marker=dict(
            size=6,
            color="rgba(255,165,0,0.8)",      # warm orange
            line=dict(width=1, color="white"),
            opacity=0.9
        )
    ))

    # ——— beautiful hoverlabel styling ———
    fig.update_layout(
        hoverlabel=dict(
            bgcolor="rgba(0,0,0,0.7)",
            bordercolor="white",
            font_size=12,
            font_family="Poppins, sans-serif",
            font_color="white"
        )
    )

    # transparent paper so our CSS texture shows through
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)"
    )

    return pio.to_html(fig, full_html=False)

def generate_line_chart(df, selected_type):
    if selected_type != 'All':
        df = df[df['accident_type'] == selected_type]
    df = df.copy()
    df['year'] = df['accident_date'].dt.year
    yearly_counts = df['year'].value_counts().sort_index()

    # ---- load & encode your airplane image ----
    img_path = Path(__file__).parent.parent / "static" / "plane.png"
    img_b64 = base64.b64encode(img_path.read_bytes()).decode()

    # ---- build the line + markers ----
    fig = px.line(
        x=yearly_counts.index,
        y=yearly_counts.values,
        labels={'x': 'Year', 'y': 'Number of Accidents'},
        title=f"Accidents per Year ({selected_type})"
    )
    fig.update_traces(mode='lines+markers', marker=dict(size=8))

    # ---- drop the airplane into the background ----
    fig.update_layout(
        images=[dict(
            source=f"data:image/png;base64,{img_b64}",
            xref="paper", yref="paper",
            x=0.5, y=0.5,
            sizex=1, sizey=1,
            xanchor="center", yanchor="middle",
            sizing="stretch",
            opacity=0.2,     # make it super faint
            layer="below"
        )],
        margin={"r":0,"t":80,"l":50,"b":50},
        height=500
    )

    return pio.to_html(fig, full_html=False)


def generate_cause_fatalities_bar(df):
    
    # group by cause_category: total fatalities & count of records
    grouped = (
        df
        .groupby('cause_category')
        .agg(
            total_fatalities=('fatalities', 'sum'),
            count=('fatalities', 'count')
        )
        .reset_index()
    )

    # pick top 5 by total fatalities
    top5 = grouped.sort_values('total_fatalities', ascending=False).head(5)

    # build bar chart with Plotly Express
    fig = px.bar(
        top5,
        x='cause_category',
        y='total_fatalities',
        text='count',
        labels={
            'cause_category': 'Cause Category',
            'total_fatalities': 'Total Fatalities'
        }
    )
    fig.update_traces(textposition='inside')
    fig.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0},
        height=500
    )

    return pio.to_html(fig, full_html=False)


#----FOR PILOT------------


def generate_pilot_choropleth(df):
    # make a copy so we don’t clobber the original
    d = df.copy()

    # strip whitespace from cause_category
    d['cause_category'] = d['cause_category'].str.strip()

    # filter for exactly “Pilot Induced”
    pil_df = d[d['cause_category'] == 'Pilot Induced']

    # if there’s no data, return a friendly message
    if pil_df.empty:
        return "<div>No Pilot-Induced crashes found for the current filter.</div>"

    # count crashes by state
    state_counts = (
        pil_df
        .groupby('state')
        .size()
        .reset_index(name='crash_count')
    )

    # build a US-only choropleth
    fig = px.choropleth(
        state_counts,
        locations='state',
        locationmode='USA-states',
        color='crash_count',
        scope='usa',
        labels={'crash_count': 'Crash Count'},
        title='Pilot-Induced Crashes by State'
    )

    fig.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0},
        height=500
    )

    return pio.to_html(fig, full_html=False)


def generate_pilot_heatmap(df):
    # make a local copy & trim whitespace
    d = df.copy()
    d['cause_category'] = d['cause_category'].str.strip()

    # filter to only pilot-induced
    pil = d[d['cause_category'] == 'Pilot Induced']
    if pil.empty:
        return "<div>No Pilot-Induced crashes to display.</div>"

    # build a Mapbox density (heatmap)
    fig = px.density_mapbox(
        pil,
        lat='latitude',
        lon='longitude',
        radius=10,                        # adjust for “spread” of each point
        hover_data=['city','state','fatalities'],
        zoom=2,                           # starting zoom
        center={'lat': 37.8, 'lon': -96}, # center of the U.S.
        mapbox_style='carto-positron',
        title='Heatmap of Pilot-Induced Crashes'
    )
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, height=500)
    return pio.to_html(fig, full_html=False)

def generate_pilot_qualification_bar(df):
    # Filter for "Pilot Induced" causes
    filtered_df = df[df['cause_category'].str.strip().str.lower() == 'pilot induced'].copy()

    # Replace blank or NaN qualifications with "Unknown"
    filtered_df['pilot_qualification'] = filtered_df['pilot_qualification'].fillna('Unknown').replace(r'^\s*$', 'Unknown', regex=True)

    # Count qualificationsimport matplotlib
    qualification_counts = filtered_df['pilot_qualification'].value_counts().reset_index()
    qualification_counts.columns = ['Qualification', 'Crash Count']

    fig = px.bar(
        qualification_counts,
        x='Qualification',
        y='Crash Count',
        title='Crashes by Pilot Qualification (Pilot-Induced Only)',
        labels={'Crash Count': 'No. of Crashes'},
        text='Crash Count'
    )

    fig.update_layout(xaxis_tickangle=-45)
    return pio.to_html(fig, full_html=False)


def generate_day_night_maps(df):
    # Filter for pilot-induced accidents
    df = df[df['cause_category'].str.strip().str.lower() == 'pilot induced']
    # Separate data
    day_df = df[df['light_condition'].str.strip().str.lower() == 'day']
    night_df = df[df['light_condition'].str.strip().str.lower() == 'night']

    # Create subplot with 2 mapbox maps
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Daytime Accidents", "Nighttime Accidents"),
        specs=[[{"type": "mapbox"}, {"type": "mapbox"}]]
    )

    # Day map (light)
    fig.add_trace(
        go.Scattermapbox(
            lat=day_df['latitude'],
            lon=day_df['longitude'],
            mode='markers',
            marker=dict(size=6, color='orange'),
            name='Day',
            text=day_df['aircraft_model'],
            hoverinfo='text'
        ),
        row=1, col=1
    )

    # Night map (dark)
    fig.add_trace(
        go.Scattermapbox(
            lat=night_df['latitude'],
            lon=night_df['longitude'],
            mode='markers',
            marker=dict(size=6, color='aqua'),
            name='Night',
            text=night_df['aircraft_model'],
            hoverinfo='text'
        ),
        row=1, col=2
    )

    fig.update_layout(
        height=600,
        title_text="Pilot-Induced Crashes: Day vs Night",
        showlegend=False,
        margin=dict(r=0, t=40, l=0, b=0),
        mapbox=dict(
            style="carto-positron",  # light for day map
            center={'lat': 37.8, 'lon': -96},
            zoom=1
        ),
        mapbox2=dict(
            style="carto-darkmatter",  # dark for night map
            center={'lat': 37.8, 'lon': -96},
            zoom=1
        )
    )

    return pio.to_html(fig, full_html=False)


def generate_2gram_wordcloud(df):
    # Drop NaNs and convert to lowercase
    texts = df['accident_type'].dropna().str.lower().tolist()

    # Extract 2-grams using CountVectorizer
    vectorizer = CountVectorizer(ngram_range=(2, 2), stop_words='english')
    X = vectorizer.fit_transform(texts)
    sums = X.sum(axis=0)

    # Create frequency dictionary
    freqs = {word: int(sums[0, idx]) for word, idx in vectorizer.vocabulary_.items()}

    # Generate word cloud
    wc = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(freqs)

    # Convert to base64 image string
    buffer = BytesIO()
    plt.figure(figsize=(10, 5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close()

    return img_base64


def generate_pilot_age_histogram(df):
    df = df.dropna(subset=['pilot_age'])
    ages = df['pilot_age'].astype(float)

    # Basic stats
    n = len(ages)
    data_range = ages.max() - ages.min()

    # Sturges
    sturges_bins = int(np.ceil(np.log2(n) + 1))

    # Scott
    scott_bin_width = 3.5 * np.std(ages) / (n ** (1/3))
    scott_bins = int(np.ceil(data_range / scott_bin_width))

    # Freedman-Diaconis
    iqr = np.percentile(ages, 75) - np.percentile(ages, 25)
    fd_bin_width = 2 * iqr / (n ** (1/3))
    fd_bins = int(np.ceil(data_range / fd_bin_width))

    # Calculate KDE
    kde = gaussian_kde(ages)
    x_range = np.linspace(ages.min(), ages.max(), 500)
    kde_vals = kde(x_range)

    # Create figure
    fig = go.Figure()

    # Sturges histogram + KDE
    fig.add_trace(go.Histogram(x=ages, nbinsx=sturges_bins, name='Histogram (Sturges)', visible=True, opacity=0.6))
    fig.add_trace(go.Scatter(x=x_range, y=kde_vals * n * (data_range / sturges_bins), mode='lines', name='KDE (Sturges)', visible=True))

    # Scott histogram + KDE
    fig.add_trace(go.Histogram(x=ages, nbinsx=scott_bins, name='Histogram (Scott)', visible=False, opacity=0.6))
    fig.add_trace(go.Scatter(x=x_range, y=kde_vals * n * (data_range / scott_bins), mode='lines', name='KDE (Scott)', visible=False))

    # Freedman-Diaconis histogram + KDE
    fig.add_trace(go.Histogram(x=ages, nbinsx=fd_bins, name='Histogram (Freedman-Diaconis)', visible=False, opacity=0.6))
    fig.add_trace(go.Scatter(x=x_range, y=kde_vals * n * (data_range / fd_bins), mode='lines', name='KDE (Freedman-Diaconis)', visible=False))

    # Buttons
    fig.update_layout(
        updatemenus=[
            {
                "buttons": [
                    {"label": "Sturges", "method": "update",
                     "args": [{"visible": [True, True, False, False, False, False]},
                              {"title": "Pilot Age Histogram (Sturges)"}]},
                    {"label": "Scott", "method": "update",
                     "args": [{"visible": [False, False, True, True, False, False]},
                              {"title": "Pilot Age Histogram (Scott)"}]},
                    {"label": "Freedman-Diaconis", "method": "update",
                     "args": [{"visible": [False, False, False, False, True, True]},
                              {"title": "Pilot Age Histogram (Freedman-Diaconis)"}]},
                ],
                "direction": "down",
                "showactive": True,
                "x": 0.5,
                "xanchor": "center",
                "y": 1.2,
                "yanchor": "top"
            }
        ],
        title="Pilot Age Histogram",
        xaxis_title="Pilot Age",
        yaxis_title="Count",
        height=500,
        margin={"r": 0, "t": 100, "l": 50, "b": 50},
        bargap=0.05
    )

    return pio.to_html(fig, full_html=False)


#---------For Unknown----------------

# 1) Choropleth of Unknown-cause crashes by state
def generate_unknown_choropleth(df):
    d = df.copy()
    d['cause_category'] = d['cause_category'].str.strip().str.lower()
    unk = d[d['cause_category'] == 'unknown']
    if unk.empty:
        return "<div>No Unknown-cause crashes found.</div>"
    counts = unk.groupby('state').size().reset_index(name='crash_count')
    fig = px.choropleth(
        counts,
        locations='state',
        locationmode='USA-states',
        color='crash_count',
        scope='usa',
        labels={'crash_count':'Crash Count'}
    )
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, height=500)
    return pio.to_html(fig, full_html=False)

# 2) Hot-spot density map for Unknown causes
def generate_unknown_heatmap(df):
    d = df.copy()
    d['cause_category'] = d['cause_category'].str.strip().str.lower()
    unk = d[d['cause_category'] == 'unknown']
    if unk.empty:
        return "<div>No Unknown-cause crashes to display.</div>"
    fig = px.density_mapbox(
        unk,
        lat='latitude',
        lon='longitude',
        radius=10,
        hover_data=['city','fatalities'],
        zoom=2,
        center={'lat':37.8,'lon':-96},
        mapbox_style='carto-positron'
    )
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, height=500)
    return pio.to_html(fig, full_html=False)

# 3) “Qualification breakdown” doesn’t make sense for Unknown, so let’s show a breakdown by accident_type
def generate_unknown_type_breakdown(df):
    d = df.copy()
    d['cause_category'] = d['cause_category'].str.strip().str.lower()
    unk = d[d['cause_category'] == 'unknown']

    if unk.empty:
        return "<div>No Unknown-cause crashes to break down.</div>"

    # build your counts
    # build your counts (top 10 only)
    vc = unk['aircraft_make'].value_counts().head(10)
    type_counts = vc.reset_index()
    type_counts.columns = ['Aircraft Make', 'Count']

    # find max so we can set a clean y-range
    max_count = type_counts['Count'].max()
    y_max = max_count * 1.1  # 10% headroom
    tick_step = max(1, int(max_count / 5))  # 5 ticks

    fig = px.bar(
        type_counts,
        x='Aircraft Make',
        y='Count',
        title='Unknown-Cause Crashes by Aircraft Make',
        text='Count'
    )
    fig.update_traces(textposition='inside')
    fig.update_layout(
        xaxis_tickangle=-45,
        margin={"r":0, "t":40, "l":0, "b":0},
        height=500,
        # ---- add these ----
        yaxis=dict(
            range=[0, y_max],
            tick0=0,
            dtick=tick_step
        )
    )

    return pio.to_html(fig, full_html=False)


# 4) Day vs Night for Unknown causes
def generate_unknown_day_night_maps(df):
    d = df.copy()
   # Filter for pilot-induced accidents
    df = df[df['cause_category'].str.strip().str.lower() == 'unknown']
    # Separate data
    day_df = df[df['light_condition'].str.strip().str.lower() == 'day']
    night_df = df[df['light_condition'].str.strip().str.lower() == 'night']

    # Create subplot with 2 mapbox maps
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Daytime Accidents", "Nighttime Accidents"),
        specs=[[{"type": "mapbox"}, {"type": "mapbox"}]]
    )

    # Day map (light)
    fig.add_trace(
        go.Scattermapbox(
            lat=day_df['latitude'],
            lon=day_df['longitude'],
            mode='markers',
            marker=dict(size=6, color='orange'),
            name='Day',
            text=day_df['aircraft_model'],
            hoverinfo='text'
        ),
        row=1, col=1
    )

    # Night map (dark)
    fig.add_trace(
        go.Scattermapbox(
            lat=night_df['latitude'],
            lon=night_df['longitude'],
            mode='markers',
            marker=dict(size=6, color='aqua'),
            name='Night',
            text=night_df['aircraft_model'],
            hoverinfo='text'
        ),
        row=1, col=2
    )

    fig.update_layout(
        height=600,
        title_text="Unknown-Cause Crashes: Day vs Night",
        showlegend=False,
        margin=dict(r=0, t=40, l=0, b=0),
        mapbox=dict(
            style="carto-positron",  # light for day map
            center={'lat': 37.8, 'lon': -96},
            zoom=1
        ),
        mapbox2=dict(
            style="carto-darkmatter",  # dark for night map
            center={'lat': 37.8, 'lon': -96},
            zoom=1
        )
    )

    return pio.to_html(fig, full_html=False)

# 5) Age histogram for Unknown causes (if pilot_age exists, else it will be empty)
from scipy.stats import gaussian_kde
import numpy as np

import numpy as np
import plotly.graph_objects as go
import plotly.io as pio
from scipy.stats import gaussian_kde

def generate_unknown_age_histogram(df):
    d = df.copy()
    d['cause_category'] = d['cause_category'].str.strip().str.lower()
    unk = d[d['cause_category'] == 'unknown'].dropna(subset=['pilot_age'])
    if unk.empty:
        return "<div>No age data for Unknown-cause crashes.</div>"

    ages = unk['pilot_age'].astype(float)
    n = len(ages)
    data_range = ages.max() - ages.min()

    # Sturges
    sturges_bins = int(np.ceil(np.log2(n) + 1))

    # Scott
    scott_bw = 3.5 * np.std(ages, ddof=1) / (n ** (1/3))
    scott_bins = int(np.ceil(data_range / scott_bw))

    # Freedman–Diaconis
    iqr = np.percentile(ages, 75) - np.percentile(ages, 25)
    fd_bw = 2 * iqr / (n ** (1/3))
    fd_bins = int(np.ceil(data_range / fd_bw))

    # KDE
    kde = gaussian_kde(ages)
    x_range = np.linspace(ages.min(), ages.max(), 500)
    kde_vals = kde(x_range)

    fig = go.Figure()

    # Sturges
    fig.add_trace(go.Histogram(
        x=ages, nbinsx=sturges_bins,
        name='Histogram (Sturges)',
        visible=True, opacity=0.6
    ))
    fig.add_trace(go.Scatter(
        x=x_range,
        y=kde_vals * n * (data_range / sturges_bins),
        mode='lines',
        name='KDE (Sturges)',
        visible=True
    ))

    # Scott
    fig.add_trace(go.Histogram(
        x=ages, nbinsx=scott_bins,
        name='Histogram (Scott)',
        visible=False, opacity=0.6
    ))
    fig.add_trace(go.Scatter(
        x=x_range,
        y=kde_vals * n * (data_range / scott_bins),
        mode='lines',
        name='KDE (Scott)',
        visible=False
    ))

    # Freedman–Diaconis
    fig.add_trace(go.Histogram(
        x=ages, nbinsx=fd_bins,
        name='Histogram (Freedman–Diaconis)',
        visible=False, opacity=0.6
    ))
    fig.add_trace(go.Scatter(
        x=x_range,
        y=kde_vals * n * (data_range / fd_bins),
        mode='lines',
        name='KDE (Freedman–Diaconis)',
        visible=False
    ))

    # Dropdown menu
    fig.update_layout(
        updatemenus=[{
            "buttons": [
                {
                    "label": "Sturges",
                    "method": "update",
                    "args": [
                        {"visible": [True, True, False, False, False, False]},
                        {"title": "Age Distribution (Unknown) — Sturges"}
                    ]
                },
                {
                    "label": "Scott",
                    "method": "update",
                    "args": [
                        {"visible": [False, False, True, True, False, False]},
                        {"title": "Age Distribution (Unknown) — Scott"}
                    ]
                },
                {
                    "label": "Freedman–Diaconis",
                    "method": "update",
                    "args": [
                        {"visible": [False, False, False, False, True, True]},
                        {"title": "Age Distribution (Unknown) — Freedman–Diaconis"}
                    ]
                },
            ],
            "direction": "down",
            "showactive": True,
            "x": 0.5,
            "xanchor": "center",
            "y": 1.2,
            "yanchor": "top"
        }],
        title="Age Distribution (Unknown-Cause)",
        xaxis_title="Pilot Age",
        yaxis_title="Count",
        height=500,
        margin={"r": 0, "t": 100, "l": 50, "b": 50},
        bargap=0.05
    )

    return pio.to_html(fig, full_html=False)
