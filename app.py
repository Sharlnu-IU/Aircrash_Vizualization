# app.py
from flask import Flask, render_template, request
import pandas as pd
from utils.visualisation import *
from utils.data_loader import load_clean_data, get_accident_types

app = Flask(__name__)

df = load_clean_data()

@app.route('/', methods=['GET'])
def index():
    accident_types = get_accident_types(df)
    selected_type = request.args.get('type', 'All')
    plot_html = generate_map_plot(df, selected_type)
    line_html = generate_line_chart(df, selected_type)
    animated_html=generate_animated_map(df)

    return render_template(
        'index.html',
        plot_html=plot_html,
        line_html=line_html,
        animated_html=animated_html,
        accident_types=accident_types,
        selected_type=selected_type
    )


@app.route('/digdeep', methods=['GET'])
def digdeep():
    accident_types = get_accident_types(df)
    selected_type = request.args.get('type', 'All')
    line_html    = generate_line_chart(df, selected_type)
    bar_html  = generate_cause_fatalities_bar(df, selected_type)
    choropleth_html = generate_pilot_choropleth(df)
    heatmap_html     = generate_pilot_heatmap(df)
    pilot_html=generate_pilot_qualification_bar(df)
    daynight_html=generate_day_night_maps(df)
    wordcloud_img=generate_2gram_wordcloud(df)
    age_hist_html = generate_pilot_age_histogram(df)
    return render_template(
        'digdeep.html',
        accident_types=accident_types,
        selected_type=selected_type,
        line_html=line_html,
        bar_html=bar_html,
        choropleth_html=choropleth_html,
        heatmap_html = heatmap_html,
        pilot_html=pilot_html,
        daynight_html=daynight_html,
        wordcloud_img=wordcloud_img,
        age_hist_html=age_hist_html
    )

if __name__ == '__main__':
    app.run(debug=True)
