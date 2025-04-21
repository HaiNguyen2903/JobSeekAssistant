import pandas as pd

def extract_job_data(df_path):
    df = pd.read_csv(df_path)

    title_keywords = ['Data ', 'AI ', 'ML ', 'Machine Learning ']

    df = df[df['title'].str.contains('|'.join(title_keywords), case=False, na=False)]

    return df

def main():
    return

if __name__ == '__main__':
    main()