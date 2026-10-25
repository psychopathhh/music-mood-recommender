# Отчет по ML-проекту: Music Mood Recommender

## Цель проекта

Создать ML-модель и веб-приложение, которое определяет настроение песни и подбирает музыкальные рекомендации по признакам трека.

## Датасет

Используется `dataset.csv` со Spotify-подобными признаками: `energy`, `valence`, `danceability`, `acousticness`, `instrumentalness`, `speechiness`, `tempo`, `popularity`, `track_genre`, `artists`, `track_name`.

## EDA

В `train_model.py` выполняется первичный анализ данных:

- количество строк и столбцов;
- число уникальных треков, артистов и жанров;
- количество пропусков до очистки;
- распределение классов настроения;
- самые частые жанры;
- средние значения основных аудио-признаков.

Метрики сохраняются в `artifacts/eda_metrics.json` и отображаются на вкладке EDA в веб-приложении.

## Предобработка

- удаляются технические `Unnamed`-столбцы;
- удаляются строки с пропусками в ключевых признаках;
- удаляются дубликаты по `track_id`;
- категориальные признаки кодируются One-Hot Encoding;
- числовые признаки масштабируются StandardScaler;
- признаки `energy`, `valence`, `danceability`, `acousticness` используются для создания `mood`, но исключаются из обучения классификатора;
- целевой класс `mood` создается по правилам на основе `energy`, `valence`, `danceability`, `acousticness`.

## Классы настроения

- `happy`: высокая energy и высокая valence;
- `intense`: высокая energy и низкая valence;
- `calm`: низкая energy и высокая valence;
- `sad`: низкая energy и низкая valence;
- `dance`: высокая danceability и высокая energy;
- `chill`: высокая acousticness и низкая energy;
- `neutral`: промежуточные значения.

## Модель

Используется `RandomForestClassifier` в `Pipeline` вместе с предобработкой. Чтобы снизить leakage-like эффект, классификатор не получает признаки, по которым создан target: `energy`, `valence`, `danceability`, `acousticness`. Для оценки качества применяется train/test split и кросс-валидация. Гиперпараметры подбираются через `GridSearchCV`.

Подбираются параметры:

- `n_estimators`;
- `max_depth`;
- `min_samples_leaf`.

## Рекомендательная часть

Похожие треки находятся через `NearestNeighbors` по музыкальному вектору: danceability, energy, valence, tempo, acousticness, instrumentalness, speechiness, popularity.

Похожие артисты находятся по агрегированным средним признакам артиста: mean energy, mean danceability, mean valence, mean tempo, mean acousticness, mean popularity.

## Web-приложение

Сайт на Flask содержит вкладки:

- Recommend by Mood;
- Similar Tracks;
- Similar Artists;
- Playlist Generator;
- EDA & Model.

Пользователь выбирает настроение, жанр, энергию, танцевальность или режим прослушивания. Для плейлистов используется content-based recommendation: каждый режим имеет целевой музыкальный профиль, а треки ранжируются по cosine similarity к этому профилю.

## Запуск

```bash
pip install -r requirements.txt
python train_model.py
python app.py
```

После запуска открыть `http://127.0.0.1:5000`.

## Генерация плейлистов

Плейлисты реализованы как content-based recommender. Для режимов Workout, Study, Party, Sad evening, Chill и Road trip заданы целевые профили по признакам danceability, energy, valence, tempo, acousticness, instrumentalness, speechiness и popularity. После пользовательских фильтров система масштабирует признаки и считает cosine similarity между каждым треком и профилем режима. В интерфейсе показывается match score — процент совпадения трека с выбранным сценарием.

## Почему accuracy стала честнее

В датасете нет готовой колонки настроения, поэтому `mood` создается автоматически по музыкальным правилам из `energy`, `valence`, `danceability` и `acousticness`. Эти признаки исключены из входа классификатора, чтобы модель не повторяла правила напрямую. Рекомендательная часть не ограничивается этим правилом: похожие треки, артисты и плейлисты используют полный content-based audio-вектор, где эти признаки важны и уместны.
