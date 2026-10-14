# Centimo: Social Media & App Review Sentiment Analysis

Centimo is a Django web app that analyses public sentiment from **YouTube video comments**, **Reddit post comments** and **Google Play Store app reviews**. Paste a link, and Centimo fetches the comments or reviews, classifies each one as Positive, Neutral or Negative, and shows the results as charts, a word cloud and a plain-English summary. You can download the analysis as a PDF report.

Users can also sign up, link their YouTube channel and Reddit profile, and see a personal dashboard with their channel stats and live Play Store trends.

---

## Features

- **Link-based analysis**: auto-detects whether a URL is YouTube, Reddit or Google Play and routes it to the right pipeline
- **Sentiment classification** into Positive / Neutral / Negative, with counts and percentages
- **Visual insights**: sentiment pie chart and word cloud
- **Language detection**: breakdown of the languages used in the comments
- **Spam detection**: counts comments containing common spam phrases (YouTube and Play Store)
- **Auto-generated summary** that names the most frequent descriptive words
- **PDF report export** built with ReportLab
- **User accounts**: sign up, log in, log out, with hashed passwords and session-based auth
- **Personal dashboard**: YouTube subscribers, videos and views; Reddit karma, awards and account age; trending and rising Play Store apps (India)
- **Login activity tracking** and a **contact / feedback form**

---

## How the sentiment analysis works

| Source            | Data fetched                                | Method                                                    |
| ----------------- | ------------------------------------------- | --------------------------------------------------------- |
| YouTube           | Up to 600 top-level comments (YouTube Data API v3) | **VADER** compound score: `> 0.05` Positive, `< -0.05` Negative, otherwise Neutral |
| Reddit            | All comments on the post (PRAW)             | **VADER**, same thresholds                                |
| Google Play Store | Up to 600 English (US) reviews (`google-play-scraper`) | **TextBlob** polarity: `> 0.2` Positive, `< -0.2` Negative, otherwise Neutral |

Supporting analysis:

- **Keyword extraction**: TextBlob part-of-speech tagging keeps adjectives (JJ, JJR, JJS); the top 5 feed the written summary
- **Language detection**: `langdetect`, with codes mapped to language names through `pycountry`
- **Spam detection**: simple keyword matching ("subscribe", "buy now", "visit my channel", and so on)
- **Word cloud**: `wordcloud` rendered with matplotlib and embedded as a base64 image

> VADER and TextBlob are lexicon/rule-based models. They need no training, but they struggle with sarcasm, slang, code-mixed text and non-English comments.

---

## Tech stack

Django · SQLite · NLTK (VADER) · TextBlob · pandas · matplotlib · WordCloud · langdetect · pycountry · ReportLab · PRAW · Google API Client · google-play-scraper

---

## Project structure

```
Centimo-Sentiment-Analysis/
└── centimo/
    ├── manage.py
    ├── db.sqlite3
    ├── centimo/            # Django project (settings, urls, wsgi/asgi)
    ├── mocenti/            # Main app
    │   ├── models.py       # Signup, UserActivity, Feedback
    │   ├── views.py        # Fetching, sentiment analysis, PDF, auth, dashboard
    │   ├── urls.py
    │   └── migrations/
    ├── templates/          # index, login, signup, dashboard, url, result, About, contact
    └── static/             # CSS, JS, images
```

### Routes

| URL               | Purpose                                      |
| ----------------- | -------------------------------------------- |
| `/`               | Landing page                                 |
| `/signup`         | Create an account                            |
| `/login`, `/logout/` | Authentication                            |
| `/dashboard`      | Personal dashboard (login required)          |
| `/url`            | Paste a link to analyse                      |
| `/download-pdf/`  | Download the PDF analysis report             |
| `/contact/`       | Contact / feedback form                      |

---

## Getting started

### Prerequisites

- Python 3.10+
- A **YouTube Data API v3** key ([Google Cloud Console](https://console.cloud.google.com/))
- A **Reddit API** client ID and secret ([reddit.com/prefs/apps](https://www.reddit.com/prefs/apps)), created as a "script" app
- Internet access (the app fetches live data and downloads the VADER lexicon on startup)

### 1. Clone and install

```bash
git clone https://github.com/siddharth-bhairy/Centimo-Sentiment-Analysis.git
cd Centimo-Sentiment-Analysis/centimo

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install django nltk textblob pandas matplotlib wordcloud langdetect pycountry \
            reportlab praw google-api-python-client google-play-scraper
python -m textblob.download_corpora
```

### 2. Add your API keys

Open `mocenti/views.py` and fill in the placeholder constants near the top:

```python
YOUTUBE_API_KEY      = "your-youtube-api-key"
REDDIT_CLIENT_ID     = "your-reddit-client-id"
REDDIT_CLIENT_SECRET = "your-reddit-client-secret"
REDDIT_USER_AGENT    = "centimo by u/your_reddit_username"
```

> Do not commit real keys. A safer approach is reading them from environment variables, for example `YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "")`.

### 3. Check `ALLOWED_HOSTS`

In `centimo/settings.py` the `ALLOWED_HOSTS` line is missing a closing quote, which stops Django from starting. Change it to:

```python
ALLOWED_HOSTS = ["centimo-hrceevhwdeb4fecu.centralindia-01.azurewebsites.net", "localhost", "127.0.0.1"]
```

### 4. Set up the database and run

```bash
python manage.py migrate
python manage.py runserver
```

Open <http://127.0.0.1:8000>.

`DEBUG` is set to `False`, so Django will not serve static files (CSS, images) locally. For local development either set `DEBUG = True` in `settings.py` or run:

```bash
python manage.py runserver --insecure
```

---

## Usage

1. Go to **/signup** and create an account (optionally add your YouTube channel and Reddit profile links).
2. Log in to see your **dashboard**.
3. Open **/url** and paste one of the following:
   - A YouTube video: `https://www.youtube.com/watch?v=...` or `https://youtu.be/...`
   - A Reddit post: `https://www.reddit.com/r/<sub>/comments/...`
   - A Play Store app: `https://play.google.com/store/apps/details?id=<package>`
4. Read the results page, then click **Download PDF** to save the report.

---

## Limitations

- Sentiment is lexicon-based (VADER / TextBlob), not a trained model, so sarcasm and context are often missed.
- YouTube is capped at 600 top-level comments; replies are not analysed.
- Play Store reviews are fetched in English for the US store only.
- Spam detection is keyword-based and will miss many spam patterns.
- Language detection can be unreliable on very short comments.
- Subject to YouTube and Reddit API quotas and rate limits.

## Possible improvements

- Replace or combine lexicon scoring with a transformer model such as `cardiffnlp/twitter-roberta-base-sentiment-latest`
- Add multilingual sentiment support
- Cache analysis results to save API quota
- Move secrets to environment variables and use a production database
- Add a `requirements.txt` and automated tests

## License

No license has been specified yet. Add a `LICENSE` file before others reuse the code.
