# Recommendation-Systems

В рамках учебного соревнования Skillfactory (ссылка на соревнование: https://www.kaggle.com/c/recommendationsv4) необходимо было реализовать алгоритм построения рекомендательной системы на основе истории оценок пользователей для товаров, а также построить прототип работающего сервиса.

В качестве данных в работе над проектом было предусмотрено 3 файла: train и test с историями оценок пользователей, а также отзывами о товаре; и мета-словарь, содержащий справочную информацию о продуктах.

В качестве алгоритма для построения рекомендаций после ряда экспериментов был выбран LightFM, дающий наилучшее знаение целевой метрики. В качестве альтернатив были опробованы FastAi, NN, suprise (для примера приведен ноутбук с экспериментами Rec_sys_FastAI.ipynb).

Репозиторий содержит следующие файлы:

recommendations-lightfm.ipynb - ноутбук с обработкой данных и EDA, построениеv модели и сохранением получившихся эмбедингов.

Rec_sys_FastAI.ipynb - пример реализации рек. системы  помощью алгоритма FastAI.

/heroku_deploy - папка с файлами для деплоя проекта в heroku.

Датасет food_dict_prepared.csv, использованный для деплоя сервиса на heroku, доступен по ссылке: https://drive.google.com/file/d/1XcF6xdeq5DnYqi2MlrZAfxdN-8cJs4AR/view?usp=sharing

Демонстрация работы сервиса с помощью streamlit и heroku: https://lit-garden-31467.herokuapp.com/
