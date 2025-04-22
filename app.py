# app.py
from flask import Flask, render_template, request
import pandas as pd
from utils.visualisation import generate_map_plot, generate_line_chart, generate_animated_map
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

if __name__ == '__main__':
    app.run(debug=True)
