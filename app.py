# app.py
from flask import Flask, render_template, request
import pandas as pd
from utils.visualisation import *
from utils.data_loader import load_clean_data, get_accident_types

app = Flask(__name__)

df = load_clean_data()

@app.route('/', methods=['GET'])
def index():
    plot_html = generate_map_plot(df, 'All')

    return render_template(
        'index.html',
        plot_html=plot_html
    )


@app.route('/digdeep', methods=['GET'])
def digdeep():
    accident_types = get_accident_types(df)
    # for the line chart filter
    selected_type = request.args.get('type', 'All')
    # for the drill-down (when you click a bar)
    drill = request.args.get('drill')
    if drill:
         drill = drill.strip()
    
    # overview
    line_html = None if drill else generate_line_chart(df, selected_type)
    
    bar_html  = None if drill else generate_cause_fatalities_bar(df)

    wordcloud_img= None if drill else generate_2gram_wordcloud(df)

    if drill == 'Pilot Induced':
        choropleth_html = generate_pilot_choropleth(df)
        heatmap_html     = generate_pilot_heatmap(df)
        pilot_html=generate_pilot_qualification_bar(df)
        daynight_html=generate_day_night_maps(df)
        age_hist_html = generate_pilot_age_histogram(df)
        type_break_html = None
    
    elif drill == 'Unknown':
        choropleth_html = generate_unknown_choropleth(df)
        heatmap_html    = generate_unknown_heatmap(df)
        pilot_html = None
        type_break_html = generate_unknown_type_breakdown(df)
        daynight_html   = generate_unknown_day_night_maps(df)
        age_hist_html   = generate_unknown_age_histogram(df)
    else:
        choropleth_html = heatmap_html = pilot_html = type_break_html = daynight_html = age_hist_html = None

    return render_template(
        'digdeep.html',
        accident_types=accident_types,
        selected_type=selected_type,
        drill=drill,
        line_html=line_html,
        bar_html=bar_html,
        choropleth_html=choropleth_html,
        heatmap_html = heatmap_html,
        pilot_html=pilot_html,
        daynight_html=daynight_html,
        type_break_html=type_break_html,
        wordcloud_img=wordcloud_img,
        age_hist_html=age_hist_html
    )

if __name__ == '__main__':
    app.run(debug=True)
