import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def train_model():
    print('Veri seti okunuyor...')
    df = pd.read_csv('data.csv', encoding='ISO-8859-1', nrows=20000)
    print('Veri temizleniyor...')
    df = df.dropna(subset=['CustomerID'])
    df = df[df['Quantity'] > 0]
    df['StockCode'] = df['StockCode'].astype(str)
    print('User-Item matrisi olu?turuluyor...')
    user_item_matrix = df.pivot_table(index='CustomerID', columns='StockCode', values='Quantity', aggfunc='sum').fillna(0)
    user_item_matrix = user_item_matrix.map(lambda x: 1 if x > 0 else 0)
    print('Benzerlik matrisi hesaplan?yor...')
    user_similarity = cosine_similarity(user_item_matrix)
    user_similarity_df = pd.DataFrame(user_similarity, index=user_item_matrix.index, columns=user_item_matrix.index)
    print('Model kaydediliyor...')
    with open('recommendation_model.pkl', 'wb') as f:
        pickle.dump({'user_similarity': user_similarity_df, 'user_item_matrix': user_item_matrix}, f)
    print('Tamamland?!')

if __name__ == '__main__':
    train_model()
