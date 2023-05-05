import pandas as pd 

file_paths = "src/dataset/"
file_name = "out.csv"

def generate_balaced_dataset():
    df = pd.read_csv(file_paths + file_name)

    df = df.dropna()
    
    df_positive = df[df['Sentiment']=='positivo']
    df_positive.shape

    df_negative = df[df['Sentiment']=='negativo']
    df_negative.shape

    # usamos downsampling para tener dataframes homogeneos
    df_ds_positive = df_positive.sample(df_negative.shape[0])
    df_ds_positive.shape

    df_balanced = pd.concat([df_ds_positive,df_negative ])
    df_balanced['Sentiment'].value_counts()

    # generamos una nueva columna al dataframe con nuestro valor numérico o "peso"
    df_balanced['Positive'] = df_balanced['Sentiment'].apply(lambda x: 1 if x=='positivo' else 0)
    df_balanced.sample(5)

    # guardamos el dataframe en un archivo csv
    df_balanced.to_csv(f'{file_paths}/balanced/balanced_out.csv', index=False)

    return df_balanced

