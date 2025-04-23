import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
import pandas as pd

def generate_map_plot(df, selected_type):
    # Filter by accident_type if needed
    if selected_type != 'All':
        df = df[df['accident_type'] == selected_type]

    # Create figure with orthographic projection
    fig = go.Figure()

    # Add base globe
    fig.update_geos(
        projection_type="orthographic",
        projection_rotation=dict(lat=40, lon=-97, roll=0),  # Center USA
        landcolor="lightgreen",
        oceancolor="lightblue",
        showcountries=True,
        showocean=True,
        showcoastlines=True,
        
        coastlinecolor="grey"
    )

    # Add scatter points for accidents
    fig.add_trace(go.Scattergeo(
        lon=df['longitude'],
        lat=df['latitude'],
        text=df['accident_type'],
        mode='markers',
        marker=dict(
            size=5,
            color='red',
            opacity=0.7,
            line=dict(width=0)
        )
    ))

    # Adjust layout
    fig.update_layout(
        height=600,
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        title_text='Global Accident Data'
    )

    return pio.to_html(fig, full_html=False)


def generate_line_chart(df, selected_type):
    if selected_type != 'All':
        df = df[df['accident_type'] == selected_type]

    df['year'] = df['accident_date'].dt.year
    yearly_counts = df['year'].value_counts().sort_index()

    fig = px.line(
        x=yearly_counts.index,
        y=yearly_counts.values,
        labels={'x': 'Year', 'y': 'Number of Accidents'},
        title=f"Accidents per Year ({selected_type})"
    )

    return pio.to_html(fig, full_html=False)


def generate_animated_map(df):
    df['accident_date'] = pd.to_datetime(df['accident_date'], errors='coerce')
    df['year'] = df['accident_date'].dt.year
    df = df.sort_values('year')

    fig = px.scatter_mapbox(
        df,
        lat='latitude',
        lon='longitude',
        animation_frame='year',
        hover_name='accident_type',
        hover_data={
            'accident_date': True,
            'city': True,
            'state': True,
            'cause_category': True,
            'fatalities': True
        },
        color_discrete_sequence=['blue'],
        zoom=3,
        height=600
    )

    fig.update_layout(
        mapbox=dict(
            style="carto-positron",
            center={"lat": 20, "lon": 0},
            zoom=1
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        title="Airline Crashes Animation with Detailed Hover Info"
    )

    return pio.to_html(fig, full_html=False)


def generate_cause_fatalities_bar(df, selected_type):
    # apply same filter as other charts
    if selected_type != 'All':
        df = df[df['accident_type'] == selected_type]

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
        },
        title=f"Top 5 Causes by Total Fatalities ({selected_type})"
    )
    fig.update_traces(textposition='inside')
    fig.update_layout(
        margin={"r":0,"t":40,"l":0,"b":0},
        height=500
    )

    return pio.to_html(fig, full_html=False)


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
        zoom=1,                           # starting zoom
        center={'lat': 37.8, 'lon': -96}, # center of the U.S.
        mapbox_style='carto-positron',
        title='Heatmap of Pilot-Induced Crashes'
    )
    fig.update_layout(margin={"r":0,"t":40,"l":0,"b":0}, height=500)
    return pio.to_html(fig, full_html=False)
