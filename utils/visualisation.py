import plotly.express as px
import plotly.io as pio
import pandas as pd
def generate_map_plot(df, selected_type):
    if selected_type != 'All':
        df = df[df['accident_type'] == selected_type]

    fig = px.scatter_mapbox(
        df,
        lat='latitude',
        lon='longitude',
        color_discrete_sequence=['red'],
        zoom=3,
        height=600
    )

    fig.update_layout(
        mapbox=dict(
            style="white-bg",  # Empty base to show only your custom tiles
            layers=[
                {
                    "sourcetype": "raster",
                    "source": [
                        "https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}"
                    ],
                    "below": "traces",

                }
            ],
            center={"lat": 20, "lon": 0},
            zoom=1
        ),
        # margin={"r": 0, "t": 30, "l": 0, "b": 0},
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        title="Airline Crashes on USGS Topographic Map"
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
    df.sort_values('year')
    fig = px.scatter_mapbox(
        df,
        lat='latitude',
        lon='longitude',
        animation_frame='year',
        color_discrete_sequence=['blue'],
        zoom=3,
        height=600
    )

    fig.update_layout(
        mapbox=dict(
            style="white-bg",  # Empty base to show only your custom tiles
            layers=[
                {
                    "sourcetype": "raster",
                    "source": [
                        "https://basemap.nationalmap.gov/arcgis/rest/services/USGSTopo/MapServer/tile/{z}/{y}/{x}"
                    ],
                    "below": "traces",

                }
            ],
            center={"lat": 20, "lon": 0},
            zoom=1
        ),
        margin={"r": 0, "t": 0, "l": 0, "b": 0},
        title="Airline Animation on USGS Topographic Map"
    )

    return pio.to_html(fig, full_html=False)