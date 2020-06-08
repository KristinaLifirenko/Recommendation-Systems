import streamlit as st
import numpy as np
import pandas as pd
import nmslib
import pickle
import scipy.sparse as sparse
import time
import requests
from ast import literal_eval

#@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_embeddings():

    """Функция для загрузки векторных представлений"""

    with open('item_embeddings_lfm.pickle', 'rb') as f:
        item_embeddings = pickle.load(f)
    nms_idx = nmslib.init(method='hnsw', space='cosinesimil')
    nms_idx.addDataPointBatch(item_embeddings)
    nms_idx.createIndex(print_progress=True)
    return item_embeddings,nms_idx


#@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def nearest_items_nms(itemid, index, n=10):

    """Функция для поиска n ближайших соседей, возвращает построенный индекс"""

    itemid = int(itemid)
    nn = index.knnQuery(item_embeddings[itemid], k=n)
    return nn

def read_files(folder_name='data'):
    """
    Функция для работы с продуктовым словарем
    """
    items = pd.read_csv('food_dict_prepared.csv')
    items['itemid'] = items['itemid'].astype(str)
    items['price'].fillna('Not Available',inplace=True)
    items['image'].fillna('No Image Available',inplace=True)
    items['image'] = items['image'].apply(lambda x: x[2:-2] if pd.notnull(x) else x)
    items['description'] = items['description'].apply(lambda x: x[2:-2] if pd.notnull(x) else x)

    return items

def read_reviews(folder_name='data'):
    """
    Функция для чтения файла с отзывами
    """
    items = pd.read_csv('image.csv')
    items['itemid'] = items['itemid'].astype(str)
    items['image'] = items['image'].apply(lambda x: x[2:-2] if pd.notnull(x) else x)
    items['style'] = items['style'].fillna('No description')
    return items

#@st.cache
def product_description(option):
    """
    Функция для создания составляющих описания товара
    """
    choice = products[products['itemid'] == option]
    try:
        url = choice['image'].iloc[0]
        img = requests.get(url).content
    except:
        img = 'No Image Available'
    price = choice['price'].iloc[0]
    title = choice['title'].iloc[0]
    brand = choice['brand'].iloc[0]
    desc = choice['description'].iloc[0]
    rank = choice['rank'].iloc[0]

    return choice, img, price, title, brand, desc, rank

st.title('Welcome to Recommendation System Prototype')

#Загружаем данные
products = read_files()
item_embeddings,nms_idx = load_embeddings()

# вводим id или часть id интересующего нас товара
product = st.text_input('Search for product...', '')

# находим все подходящие товары в датасете
output = products[products['itemid'].str.contains(product)]

# выбираем товар из списка
option = st.selectbox('Select product', output['itemid'].values)

# выводим информацию о выбранном товаре
st.header('Product info:')

#можно посмотреть отзывы покупателей об этом товаре
if st.checkbox("Show customers' reviews on the product"):
    reviews = read_reviews()
    ch = reviews[reviews['itemid'] == option]
    rec = ch.drop_duplicates('reviewerName')
    for name, summary, review in zip(rec['reviewerName'][:5], rec['summary'][:5], rec['reviewText'][:5]): 
        st.subheader(name)
        st.markdown(f'----*{summary}*----')
        st.write(review)
        st.markdown('------')

choice, img, price, title,brand,desc, rank = product_description(option)
st.image(img, width=150)  # картинка
st.markdown(f'**{price}**')
st.markdown(f'*{title}*')
st.markdown(f'*{brand}*')
st.markdown(desc)

# выводим на боковую панель рекомендации к товару
st.sidebar.header('Products, you may also like: ')

#Ищем рекомендации
output = products[products['itemid'] == option]
val_index = output[output['itemid'].values == option]['itemid'].iloc[0]
index = nearest_items_nms(val_index, nms_idx, 5)


#Выводим для каждой рекомендации инфо о товаре
for idx in index[0][1:]:
    try:
        choice, img, price, title,brand,desc, rank = product_description(str(idx))
        try:
# if img !='No Image Available':
            st.sidebar.image(img, caption=f'Product ID: {idx}') # картинка
#        else:
        except:
            st.sidebar.markdown('No Image')
            
        st.sidebar.markdown(f'*{title}*')
        st.sidebar.markdown(f'**{price}**')
        st.sidebar.markdown('-----')
    except:
        st.sidebar.text('')