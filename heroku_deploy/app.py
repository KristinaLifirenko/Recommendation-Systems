import streamlit as st
import numpy as np
import pandas as pd
import nmslib
import pickle
import scipy.sparse as sparse
import time
import requests
from ast import literal_eval

@st.cache
def nearest_books_nms(itemid, index, n=10):
    """Функция для поиска ближайших соседей, возвращает построенный индекс"""
    itemid = int(itemid)
    nn = index.knnQuery(item_embeddings[itemid], k=n)
    return nn

def read_files(folder_name='data'):
    """
    Функция для чтения файлов + некоторых преобразований
    """
    items = pd.read_csv(folder_name+'/products.csv')
    items['itemid'] = items['itemid'].astype(str)
    items['image'] = items['image'].apply(lambda x: x[2:-2] if pd.notnull(x) else x)
    items['style'] = items['style'].fillna('No description')
#    items['style'] = items['style'].apply(lambda x: literal_eval(x) if x != 'No description' else x)
    return items

@st.cache
def load_embeddings():
    """
    Функция для загрузки векторных представлений
    """
    with open('item_embeddings_lfm.pickle', 'rb') as f:
        item_embeddings = pickle.load(f)
    nms_idx = nmslib.init(method='hnsw', space='cosinesimil')
    nms_idx.addDataPointBatch(item_embeddings)
    nms_idx.createIndex(print_progress=True)
    return item_embeddings,nms_idx

def product_description(option):
    """
    Функция для создания составляющих описания товара
    """
    choice = products[products['itemid'] == option]
    url = choice['image'].iloc[0]
    img = requests.get(url).content
    style = choice['style'].iloc[0]
    rating = choice['rating'].mean()
    rating_func = lambda r: {r<0.2 : 1, 0.2<r<0.4 : 2, 0.4<r<0.6 : 3, 0.6<r<0.8 : 4, r>0.8 : 5}[1]
    rating_star = rating_func(rating) 
    return choice, img, style, rating_star
    

st.title('Welcome to Recommendation System Prototype')

#Загружаем данные
products  = read_files(folder_name='data')  
item_embeddings,nms_idx = load_embeddings()

# вводим id или часть id интересующего нас товара
product = st.text_input('Search for product...', '')

# находим все подходящие товары в датасете
output = products[products['itemid'].str.contains(product) > 0]

# выбираем товар из списка
option = st.selectbox('Select product', output['itemid'].values)

# выводим информацию о выбранном товаре
st.header('Product info: ')
choice, img, style, rating_star = product_description(option)
st.image(img, width=150)  # картинку
if style != 'No description':  # описание
    style = literal_eval(style)
    for key, value in style.items():
        st.markdown(f'*{key}*')
        st.write(value)
else:
    st.text(style)
st.markdown(':star:'*rating_star) # рейтинг товара

# также можно посмотреть отзывы покупателей об этом товаре
if st.button("Show customers' reviews on the product"):
    rec = choice.drop_duplicates('reviewerName')
    for name, summary, review in zip(rec['reviewerName'][:5], rec['summary'][:5], rec['reviewText'][:5]): 
        st.subheader(name)
        st.markdown(f'----*{summary}*----')
        st.write(review)
        st.markdown('------')

# а тут рекомендации к товару
st.sidebar.header('Products, you may also like: ')
#Ищем рекомендации
val_index = output[output['itemid'].values == option]['itemid'].iloc[0]
index = nearest_books_nms(val_index, nms_idx, 5)

#Выводим рекомендации с краткой информацией о товаре
for idx in index[0][1:]:
    try:
        choice, img, style, rating_star = product_description(str(idx))
        st.sidebar.image(img, caption=f'Product ID: {idx}') # картинка
        if style != 'No description':  # описание
            style = literal_eval(style)
            for key, value in style.items():
                st.sidebar.text(key)
                st.sidebar.text(value)
        else:
             st.sidebar.text(style)
        st.sidebar.markdown(':star:'*rating_star) # рейтинг
        st.sidebar.markdown('-----')
    except:
        st.sidebar.text('')
