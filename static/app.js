let state = { tracks: [], artists: [], genres: [], eda: {} };

const $ = (selector) => document.querySelector(selector);
const results = $('#results');
const sourceBox = $('#sourceBox');

function fmt(value, digits) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return 'n/a';
  return Number(value).toFixed(digits || 0);
}

function titleCase(text) {
  return String(text || '').replaceAll('_', ' ').replace(/\b\w/g, (m) => m.toUpperCase());
}

function pct(value) {
  return value ? (Math.round(Number(value) * 1000) / 10) + '%' : '...';
}

const moodNames = {
  happy: 'Весёлое',
  sad: 'Грустное',
  calm: 'Спокойное',
  energetic: 'Энергичное',
  dance: 'Танцевальное',
  chill: 'Чилл',
  intense: 'Интенсивное',
  neutral: 'Нейтральное'
};
const modeNames = {
  workout: 'Тренировка',
  study: 'Учёба',
  party: 'Вечеринка',
  sad_evening: 'Грустный вечер',
  chill: 'Чилл',
  road_trip: 'Дорога'
};
function ruName(value) {
  return moodNames[value] || modeNames[value] || titleCase(value);
}
function artLetter(text) {
  return String(text || '?').trim().charAt(0).toUpperCase() || '?';
}
function artClass(text) {
  const n = String(text || '').split('').reduce((sum, ch) => sum + ch.charCodeAt(0), 0);
  return 'art-' + (n % 6);
}

function post(url, payload) {
  return fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  }).then((res) => {
    if (!res.ok) return res.json().then((data) => Promise.reject(data));
    return res.json();
  });
}

function option(value, text) {
  const el = document.createElement('option');
  el.value = value;
  el.textContent = text;
  return el;
}

function fillGenres(selectId) {
  const select = $(selectId);
  select.innerHTML = '';
  select.appendChild(option('any', 'Любой жанр'));
  state.genres.forEach((genre) => select.appendChild(option(genre, genre)));
}

function fillHints() {
  const trackHints = $('#trackHints');
  const artistHints = $('#artistHints');
  trackHints.innerHTML = '';
  artistHints.innerHTML = '';
  state.tracks.slice(0, 80).forEach((track) => {
    const item = option(track.track_name + ' - ' + track.artists, track.track_name + ' - ' + track.artists);
    trackHints.appendChild(item);
  });
  state.artists.slice(0, 80).forEach((artist) => {
    const item = option(artist.artist_primary, artist.artist_primary);
    artistHints.appendChild(item);
  });
}

function stat(label, value) {
  const div = document.createElement('div');
  div.className = 'stat';
  div.innerHTML = '<strong>' + value + '</strong><span>' + label + '</span>';
  return div;
}

function renderStats() {
  const eda = state.eda;
  $('#cvAccuracy').textContent = pct(eda.cv_accuracy);
  const stats = $('#stats');
  stats.innerHTML = '';
  stats.appendChild(stat('треков', eda.unique_tracks || 0));
  stats.appendChild(stat('артистов', eda.unique_artists || 0));
  stats.appendChild(stat('жанров', eda.genres || 0));
  stats.appendChild(stat('точность', pct(eda.model_accuracy))); 

  const summary = $('#edaSummary');
  summary.innerHTML = '';
  const rows = [
    ['Строк после очистки', eda.rows],
    ['Столбцов', eda.columns],
    ['Пропусков до очистки', eda.missing_values_before_cleaning],
    ['Точность модели', pct(eda.model_accuracy)],
    ['Кросс-валидация', pct(eda.cv_accuracy)],
    ['Лучшая глубина', eda.best_params ? eda.best_params['classifier__max_depth'] : 'n/a'],
    ['Деревьев в лесу', eda.best_params ? eda.best_params['classifier__n_estimators'] : 'n/a'],
    ['Исключены из модели', eda.excluded_from_classifier ? eda.excluded_from_classifier.join(', ') : 'n/a']
  ];
  rows.forEach((row) => {
    const div = document.createElement('div');
    div.className = 'eda-row';
    div.innerHTML = '<span>' + row[0] + '</span><strong>' + row[1] + '</strong>';
    summary.appendChild(div);
  });
  renderMiniCharts();
}

function miniRows(targetId, data, limit, formatter) {
  const target = $('#' + targetId);
  if (!target) return;
  const entries = Object.entries(data || {}).slice(0, limit || 8);
  const max = Math.max(...entries.map(([, value]) => Number(value) || 0), 1);
  target.innerHTML = entries.map(([label, value]) => {
    const width = Math.max(4, (Number(value) || 0) / max * 100);
    const shown = formatter ? formatter(value) : value;
    return '<div class="mini-row"><div class="mini-row-head"><span>' + ruName(label) + '</span><strong>' + shown + '</strong></div><div class="mini-track"><div class="mini-fill" style="width:' + width + '%"></div></div></div>';
  }).join('');
}

function renderMiniCharts() {
  const eda = state.eda || {};
  miniRows('moodChart', eda.mood_distribution || {}, 8);
  miniRows('genreChart', eda.top_genres || {}, 8);
  const features = {};
  ['energy', 'valence', 'danceability', 'acousticness', 'instrumentalness', 'speechiness'].forEach((name) => {
    if (eda.feature_means && eda.feature_means[name] !== undefined) features[name] = eda.feature_means[name];
  });
  miniRows('featureChart', features, 8, (value) => fmt(value, 2));
}

function bar(label, value) {
  const pct = Math.max(0, Math.min(100, Number(value || 0) * 100));
  return '<div class="bar"><span>' + label + '</span><div class="bar-track"><div class="bar-fill" style="width:' + pct + '%"></div></div><strong>' + fmt(value, 2) + '</strong></div>';
}

function trackCard(track) {
  const div = document.createElement('article');
  div.className = 'track-card';
  const letter = artLetter(track.track_genre || track.track_name);
  const art = '<div class="cover-art ' + artClass(track.track_genre) + '"><span>' + letter + '</span></div>';
  div.innerHTML =
    '<div class="track-top">' + art + '<div class="track-main"><h4>' + track.track_name + '</h4><p>' + track.artists + '</p></div></div>' +
    '<div class="bars">' +
      bar('энергия', track.energy) +
      bar('позитив', track.valence) +
      bar('танец', track.danceability) +
    '</div>' +
    '<div class="tags">' +
      '<span class="tag">' + ruName(track.mood) + '</span>' +
      '<span class="tag">' + track.track_genre + '</span>' +
      '<span class="tag">популярность ' + track.popularity + '</span>' +
      (track.match_score !== undefined && track.match_score !== null ? '<span class="tag match-tag">совпадение ' + Math.round(Number(track.match_score) * 100) + '%</span>' : '') +
    '</div>';
  return div;
}
function artistCard(artist) {
  const div = document.createElement('article');
  div.className = 'artist-card';
  div.innerHTML =
    '<h4>' + artist.artist_primary + '</h4>' +
    '<p>' + artist.top_genre + ' · ' + artist.tracks_count + ' треков</p>' +
    '<div class="bars">' +
      bar('энергия', artist.mean_energy) +
      bar('позитив', artist.mean_valence) +
      bar('танец', artist.mean_danceability) +
    '</div>' +
    '<div class="tags"><span class="tag">популярность ' + fmt(artist.mean_popularity, 0) + '</span><span class="tag">акустика ' + fmt(artist.mean_acousticness, 2) + '</span></div>';
  return div;
}

function renderTracks(items, title, label) {
  sourceBox.classList.add('hidden');
  results.innerHTML = '';
  $('#resultTitle').textContent = title;
  $('#resultLabel').textContent = label || 'Рекомендации';
  $('#resultCount').textContent = items.length + ' треков';
  items.forEach((item) => results.appendChild(trackCard(item)));
  if (!items.length) results.innerHTML = '<p class="empty">Ничего не найдено</p>'; 
}

function renderArtists(items, title, label) {
  sourceBox.classList.add('hidden');
  results.innerHTML = '';
  $('#resultTitle').textContent = title;
  $('#resultLabel').textContent = label || 'Артисты';
  $('#resultCount').textContent = items.length + ' артистов';
  items.forEach((item) => results.appendChild(artistCard(item)));
  if (!items.length) results.innerHTML = '<p class="empty">Ничего не найдено</p>'; 
}

function renderSource(source, kind) {
  if (!source) {
    sourceBox.classList.add('hidden');
    return;
  }
  sourceBox.classList.remove('hidden');
  if (kind === 'artist') {
    sourceBox.innerHTML = '<strong>Исходный артист:</strong> ' + source.artist_primary + ' · ' + source.top_genre + ' · популярность ' + fmt(source.mean_popularity, 0);
  } else {
    const mood = source.predicted_mood || source.mood;
    sourceBox.innerHTML = '<strong>Исходный трек:</strong> ' + source.track_name + ' · ' + source.artists + ' · настроение ' + ruName(mood);
  }
}

function commonPayload(prefix) {
  return {
    genre: $('#' + prefix + 'Genre').value,
    energy: $('#' + prefix + 'Energy').value,
    danceability: $('#' + prefix + 'Dance').value
  };
}

function bindTabs() {
  document.querySelectorAll('.tab').forEach((tab) => {
    tab.addEventListener('click', () => {
      document.querySelectorAll('.tab').forEach((item) => item.classList.remove('active'));
      document.querySelectorAll('.tab-panel').forEach((item) => item.classList.remove('active'));
      tab.classList.add('active');
      $('#panel-' + tab.dataset.tab).classList.add('active');
    });
  });
}

function bindActions() {
  document.querySelectorAll('.mood-dot').forEach((dot) => {
    dot.addEventListener('click', () => {
      $('#moodSelect').value = dot.dataset.mood;
      $('#moodButton').click();
    });
  });

  $('#moodButton').addEventListener('click', () => {
    const payload = commonPayload('mood');
    payload.mood = $('#moodSelect').value;
    payload.limit = 12;
    post('/api/recommend/mood', payload).then((data) => renderTracks(data.items, ruName(payload.mood) + ': подборка', 'Рекомендации по настроению'));
  });

  $('#playlistButton').addEventListener('click', () => {
    const payload = commonPayload('playlist');
    payload.mode = $('#playlistMode').value;
    payload.limit = $('#playlistLimit').value;
    post('/api/playlist', payload).then((data) => renderTracks(data.items, ruName(payload.mode) + ': content-based плейлист', 'Ранжирование по похожести'));
  });

  $('#trackButton').addEventListener('click', () => {
    post('/api/similar-track', { query: $('#trackQuery').value }).then((data) => {
      renderTracks(data.items || [], 'Похожие треки', 'Поиск похожих')
      renderSource(data.source, 'track');
    });
  });

  $('#predictButton').addEventListener('click', () => {
    post('/api/predict', { query: $('#trackQuery').value }).then((data) => {
      renderTracks([data], 'Настроение: ' + ruName(data.predicted_mood), 'ML-классификатор');
    }).catch((err) => renderTracks([], err.error || 'Трек не найден', 'ML-классификатор'));
  });

  $('#artistButton').addEventListener('click', () => {
    post('/api/similar-artists', { query: $('#artistQuery').value }).then((data) => {
      renderArtists(data.items || [], 'Похожие артисты', 'Векторы артистов');
      renderSource(data.source, 'artist');
    });
  });
}

fetch('/api/initial')
  .then((res) => res.json())
  .then((data) => {
    state = data;
    fillGenres('#moodGenre');
    fillGenres('#playlistGenre');
    fillHints();
    renderStats();
    renderTracks(state.tracks.slice(0, 12), 'Популярные треки', 'Пример из датасета');
    bindTabs();
    bindActions();
  })
  .catch((err) => {
    results.innerHTML = '<p class="empty">Не удалось загрузить данные проекта.</p>'; 
    console.error(err);
  });
