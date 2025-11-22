from fastapi import FastAPI, HTTPException
import pickle
import pandas as pd

app = FastAPI()

# Modeli yukle
print('Model yukleniyor...')
with open('recommendation_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

user_similarity_df = model_data['user_similarity']
user_item_matrix = model_data['user_item_matrix']
print('Model hazir!')

@app.get('/')
def home():
    return {'message': 'E-Commerce Recommendation API Calisiyor'}

@app.get('/recommend/{user_id}')
def recommend(user_id: int, num_recommendations: int = 5):
    # Kullanici ID veride var mi kontrol et
    if user_id not in user_item_matrix.index:
        raise HTTPException(status_code=404, detail='Kullanici bulunamadi')

    # Benzer kullanicilari bul
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)

    # En benzer kullaniciyi al (kendisi haric)
    # Gercek uygulamada birden fazla benzer kullanicinin ortalamasi alinir
    similar_user_id = similar_users.index[1]

    # O kullanicinin aldigi ama bizimkinin almadigi urunleri bul
    suggestions = []
    user_products = user_item_matrix.loc[user_id]
    similar_user_products = user_item_matrix.loc[similar_user_id]

    for product_id in user_item_matrix.columns:
        if user_products[product_id] == 0 and similar_user_products[product_id] == 1:
            suggestions.append(product_id)

        if len(suggestions) >= num_recommendations:
            break

    return {
        'user_id': user_id,
        'recommendations': suggestions
    }

