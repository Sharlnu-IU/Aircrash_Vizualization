import pandas as pd

def load_clean_data():
    df = pd.read_csv('dataviz_data_final_with_extra1900.csv')
    if df.columns[0].lower().startswith('unnamed'):
        df = df.iloc[:, 1:]

    df['accident_date'] = pd.to_datetime(df['accident_date'], errors='coerce')
    df['accident_type'] = df['accident_type'].fillna("Unknown")
    return df

def get_accident_types(df):
    return sorted(df['accident_type'].dropna().unique())
