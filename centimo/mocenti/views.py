import matplotlib
matplotlib.use("Agg")

import praw
from datetime import datetime, timezone
from django.shortcuts import render, redirect
from django.contrib import messages
from django.utils.timezone import now
from django.contrib.auth.hashers import make_password, check_password
from django.contrib.auth import logout as auth_logout
from mocenti.models import Signup, UserActivity, Feedback
from googleapiclient.discovery import build
import re
import pandas as pd
import nltk
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from langdetect import detect
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from io import BytesIO
import base64
import pycountry
from django.http import HttpResponse
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from urllib.parse import urlparse
from google_play_scraper import reviews
from textblob import TextBlob
from collections import Counter
import json
from reportlab.lib.colors import black, white, gray


nltk.download("vader_lexicon")

YOUTUBE_API_KEY = ""

# Extract video ID from YouTube URL
def extract_video_id(url):
    pattern = r"(?:https?://)?(?:www\.)?(?:youtube\.com/watch\?v=|youtu\.be/)([\w\-]+)"
    match = re.match(pattern, url)
    return match.group(1) if match else None

# Fetch comments from a YouTube video
def get_comments(video_url, api_key, max_comments=600):
    video_id = extract_video_id(video_url)
    if not video_id:
        return None

    youtube = build("youtube", "v3", developerKey=api_key)
    comments = []
    request = youtube.commentThreads().list(part="snippet", videoId=video_id, maxResults=100)
    response = request.execute()

    while response and len(comments) < max_comments:
        for item in response.get("items", []):
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)
            if len(comments) >= max_comments:
                break

        if "nextPageToken" in response and len(comments) < max_comments:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=response["nextPageToken"]
            )
            response = request.execute()
        else:
            break

    return comments


# Analyze sentiment of a text
def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)["compound"]
    return "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"

# Generate word cloud
def generate_word_cloud(comments):
    text = " ".join(comments)
    wordcloud = WordCloud(width=1600, height=800, background_color="white").generate(text)

    buf = BytesIO()
    plt.figure(figsize=(12, 6))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.savefig(buf, format="png", bbox_inches='tight')
    plt.close()
    buf.seek(0)

    return base64.b64encode(buf.getvalue()).decode()

# Detect languages of comments
def detect_languages(comments):
    lang_counts = {}
    for comment in comments:
        try:
            lang = detect(comment)
            lang_counts[lang] = lang_counts.get(lang, 0) + 1
        except:
            pass

    return {pycountry.languages.get(alpha_2=code).name if pycountry.languages.get(alpha_2=code) else code: count 
            for code, count in lang_counts.items()}

# Detect spam in comments
def detect_spam(comments):
    spam_keywords = ["subscribe", "buy now", "visit my channel", "check my video", "free money"]
    return sum(1 for comment in comments if any(keyword in comment.lower() for keyword in spam_keywords))

# Generate Pie Chart for Sentiment Analysis
def generate_pie_chart(sentiment_counts):
    labels, sizes = list(sentiment_counts.keys()), list(sentiment_counts.values())
    colors = ["#00BFFF", "#ADD8E6", "#87CEFA"]

    fig, ax = plt.subplots(figsize=(4, 6))
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140, colors=colors, textprops={"color": "black"})
    ax.set_title("Sentiment Distribution", fontsize=18)

    buf = BytesIO()
    plt.savefig(buf, format="png", bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)

    return base64.b64encode(buf.getvalue()).decode()

# Generate a textual summary based on sentiment analysis
def generate_summary(comments, sentiment_counts):
    if not comments:
        return "No comments found to analyze."

    keywords = extract_keywords(comments)  # Extract important words
    common_keywords = [word for word, count in Counter(keywords).most_common(5)]  # Get top 5 keywords
    
    word_list = ", ".join(common_keywords) if common_keywords else "no significant words found"

    positive = sentiment_counts.get("Positive", 0)
    negative = sentiment_counts.get("Negative", 0)

    if positive > negative:
        return f"Most comments were positive, but some users had concerns. Common positive words: {word_list}."
    elif negative > positive:
        return f"The video received mostly negative feedback. Frequent complaints include: {word_list}."
    else:
        return f"The comments were fairly balanced between positive and negative sentiments. Some commonly used words: {word_list}."

def extract_keywords(reviews):
    keywords = []
    pos_tags = ["JJ", "JJR", "JJS"]  # Adjectives

    for review in reviews:
        blob = TextBlob(str(review))
        keywords.extend([word.lower() for word, tag in blob.tags if tag in pos_tags])  # Extract relevant words

    return keywords


# Fetch Reddit profile details (Placeholder)
REDDIT_CLIENT_ID = ""
REDDIT_CLIENT_SECRET = ""
REDDIT_USER_AGENT = ""

# Initialize Reddit API client
reddit_api = praw.Reddit(
    client_id=REDDIT_CLIENT_ID,
    client_secret=REDDIT_CLIENT_SECRET,
    user_agent=REDDIT_USER_AGENT
)

# Contact Form Handling
def contact_view(request):
    if request.method == "POST":
        Feedback.objects.create(name=request.POST["name"], email=request.POST["email"], message=request.POST["comments"])
        messages.success(request, "Thank you for your feedback!")
        return redirect("contact")

    return render(request, "contact.html")

def index(request):
    return render(request, "index.html")


import re

def url(request):
    if request.method == "POST":
        user_url = request.POST.get("url").strip()

        result, error, source = None, None, None  

        if "youtube.com" in user_url or "youtu.be" in user_url:
            result, error = process_youtube(user_url)
            source = "YouTube"
        elif "reddit.com/r/" in user_url or "reddit.com/comments/" in user_url:
            result, error = process_reddit(user_url)
            source = "Reddit"
        elif re.search(r"play\.google\.com/.*/details\?", user_url):
            result, error = process_playstore(user_url)
            source = "Google Play Store"
        else:
            messages.error(request, "Unsupported URL type. Please enter a YouTube, Reddit, or Play Store link.")
            return redirect("url")

        if error:
            messages.error(request, error)
            return redirect("url")

        result["source"] = source
        result["url"] = user_url  

        return render(request, "result.html", result)

    return render(request, "url.html")




def extract_app_id(url):
    match = re.search(r"id=([a-zA-Z0-9._]+)", url)
    return match.group(1) if match else None

# ✅ Process YouTube URL
def process_youtube(video_url):
    comments = get_comments(video_url, YOUTUBE_API_KEY)
    if not comments:
        return None, "No comments found on this video."

    df = pd.DataFrame({"Comment": comments})
    df["Sentiment"] = df["Comment"].apply(analyze_sentiment)
    sentiment_counts = df["Sentiment"].value_counts().to_dict()

    total_comments = len(comments)
    sentiment_percentages = {
        "Positive": round((sentiment_counts.get("Positive", 0) / total_comments) * 100, 2),
        "Neutral": round((sentiment_counts.get("Neutral", 0) / total_comments) * 100, 2),
        "Negative": round((sentiment_counts.get("Negative", 0) / total_comments) * 100, 2),
    }

    wordcloud = generate_word_cloud(comments)
    lang_counts = detect_languages(comments)
    spam_count = detect_spam(comments)
    summary = generate_summary(comments, sentiment_counts)
    chart = generate_pie_chart(sentiment_counts)

    return {
        "video_url": video_url,
        "sentiment_counts": sentiment_counts,
        "sentiment_percentages": sentiment_percentages,
        "total_comments": total_comments,
        "chart": chart,
        "wordcloud": wordcloud,
        "lang_counts": lang_counts,
        "spam_count": spam_count,
        "summary": summary
    }, None

# ✅ Process Reddit Post
def process_reddit(post_url):
    try:
        submission = reddit_api.submission(url=post_url)
        submission.comments.replace_more(limit=None)
        comments = [comment.body for comment in submission.comments.list()]

        if not comments:
            return None, "No comments found."

        df = pd.DataFrame({"Comment": comments})
        df["Sentiment"] = df["Comment"].apply(analyze_sentiment)
        sentiment_counts = df["Sentiment"].value_counts().to_dict()

        total_comments = len(comments)
        sentiment_percentages = {
            "Positive": round((sentiment_counts.get("Positive", 0) / total_comments) * 100, 2),
            "Neutral": round((sentiment_counts.get("Neutral", 0) / total_comments) * 100, 2),
            "Negative": round((sentiment_counts.get("Negative", 0) / total_comments) * 100, 2),
        }

        wordcloud = generate_word_cloud(comments)
        chart = generate_pie_chart(sentiment_counts)
        lang_counts = detect_languages(comments)
        summary = generate_summary(comments, sentiment_counts)

        return {
            "post_url": post_url,
            "sentiment_counts": sentiment_counts,
            "sentiment_percentages": sentiment_percentages,
            "total_comments": total_comments,
            "wordcloud": wordcloud,
            "lang_counts": lang_counts,
            "summary": summary,
            "chart" : chart
        }, None
    except Exception as e:
        return None, f"Error processing Reddit post: {str(e)}"

# ✅ Process Google Play Store App
def process_playstore(app_url):
    app_id = extract_app_id(app_url)
    if not app_id:
        return None, "Invalid Google Play Store URL."

    try:
        # Fetch reviews from Google Play Store
        result, _ = reviews(app_id, lang="en", country="us", count=600)
        df = pd.DataFrame(result)[["userName", "content", "score"]].rename(columns={"content": "review"})

        if df.empty:
            return None, "No reviews found for this app."

        # Sentiment Analysis
        df["Sentiment"] = df["review"].apply(lambda r: TextBlob(str(r)).sentiment.polarity)
        df["Sentiment"] = df["Sentiment"].apply(lambda p: "Positive" if p > 0.2 else "Negative" if p < -0.2 else "Neutral")
        sentiment_counts = df["Sentiment"].value_counts().to_dict()

        # Calculate Sentiment Percentages
        total_reviews = len(df)
        sentiment_percentages = {
            "Positive": round((sentiment_counts.get("Positive", 0) / total_reviews) * 100, 2),
            "Neutral": round((sentiment_counts.get("Neutral", 0) / total_reviews) * 100, 2),
            "Negative": round((sentiment_counts.get("Negative", 0) / total_reviews) * 100, 2),
        }

        # Generate Additional Insights
        wordcloud = generate_word_cloud(df["review"].dropna().tolist())
        lang_counts = detect_languages(df["review"].dropna().tolist())
        spam_count = detect_spam(df["review"].dropna().tolist())
        summary = generate_summary(df["review"].dropna().tolist(), sentiment_counts)
        chart = generate_pie_chart(sentiment_counts)

        return {
            "app_url": app_url,
            "sentiment_counts": sentiment_counts,
            "sentiment_percentages": sentiment_percentages,
            "total_comments": total_reviews,
            "chart": chart,
            "wordcloud": wordcloud,
            "lang_counts": lang_counts,
            "spam_count": spam_count,
            "summary": summary
        }, None

    except Exception as e:
        return None, f"Error processing Play Store app: {str(e)}"


# Generate PDF Report

def download_pdf(request):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=letter)
    pdf.setTitle("Analysis Report")

    # Set Background to Black
    pdf.setFillColor(black)
    pdf.rect(0, 0, letter[0], letter[1], fill=True)

    # Function to Add Diagonal Watermark
    def add_watermark(pdf):
        pdf.setFillColor(gray)  # Light gray watermark
        pdf.setFont("Times-Italic", 20)  # Cursive-like font

        for x in range(-100, int(letter[0]), 200):  
            for y in range(-50, int(letter[1]), 100):  
                pdf.saveState()
                pdf.translate(x, y)
                pdf.rotate(45)  # Rotate watermark diagonally
                pdf.drawString(0, 0, "centimo - BG1")
                pdf.restoreState()

    # Add Watermark
    add_watermark(pdf)

    # Set Text Color to White
    pdf.setFillColor(white)

    # Title (Narrow Margin)
    pdf.setFont("Times-Italic", 22)
    pdf.drawString(50, 770, "Analysis Report")

    # Source & URL
    source = request.POST.get("source", "N/A")
    url = request.POST.get("url", "N/A")
    pdf.setFont("Times-Italic", 14)
    pdf.drawString(50, 740, f"Source: {source}")
    pdf.drawString(50, 720, f"URL: {url}")

    # Sentiment Analysis Data
    data_keys = ["positive", "neutral", "negative", "spam_count", "total_comments"]
    data = {key: request.POST.get(key, "0") for key in data_keys}
    summary = request.POST.get("summary", "No summary available.")

    y_position = 690
    pdf.setFont("Times-Italic", 16)
    pdf.drawString(50, y_position, "Sentiment Breakdown")

    pdf.setFont("Times-Italic", 14)
    for key, value in data.items():
        y_position -= 20
        pdf.drawString(50, y_position, f"{key.replace('_', ' ').title()}: {value}")

    # Language Breakdown (Fixed!)
    y_position -= 30
    pdf.setFont("Times-Italic", 16)
    pdf.drawString(50, y_position, "Detected Languages")
    pdf.setFont("Times-Italic", 14)

    # DEBUG: Check if lang_counts is coming correctly
    lang_counts_str = request.POST.get("lang_counts", "{}")
    print("Raw lang_counts received:", lang_counts_str)  # Debugging line

    try:
        lang_counts = json.loads(lang_counts_str)  # Fixed: Properly parsing the JSON input
        print("Parsed lang_counts:", lang_counts)  # Debugging line
    except json.JSONDecodeError:
        lang_counts = {}

    if lang_counts and isinstance(lang_counts, dict) and len(lang_counts) > 0:
        for lang, count in lang_counts.items():
            y_position -= 20
            pdf.drawString(50, y_position, f"{lang}: {count} comments")
    else:
        y_position -= 20
        pdf.drawString(50, y_position, "No languages detected.")  # This should now be fixed

    # Summary
    y_position -= 40
    pdf.setFont("Times-Italic", 16)
    pdf.drawString(50, y_position, "Summary")
    pdf.setFont("Times-Italic", 14)
    y_position -= 20
    pdf.drawString(50, y_position, summary)

    # Function to Add Images
    def add_image(base64_string, x, y, width, height):
        if base64_string:
            try:
                image_data = base64.b64decode(base64_string.split(",")[-1])
                image = ImageReader(BytesIO(image_data))
                pdf.drawImage(image, x, y, width, height, mask='auto')  # Preserve transparency
            except Exception as e:
                print("Error loading image:", e)

    # Add Pie Chart & Word Cloud (if available)
    y_position -= 130
    pdf.setFont("Times-Italic", 16)
    pdf.drawString(50, y_position, "Sentiment Pie Chart")
    add_image(request.POST.get("chart", ""), 50, y_position - 120, 200, 200)

    pdf.drawString(300, y_position, "Word Cloud")
    add_image(request.POST.get("wordcloud", ""), 300, y_position - 120, 250, 200)

    # Save PDF
    pdf.showPage()
    pdf.save()
    buffer.seek(0)

    response = HttpResponse(buffer, content_type="application/pdf")
    response["Content-Disposition"] = 'attachment; filename="analysis_report.pdf"'
    return response


from google_play_scraper import search

def get_playstore_trends():
    try:
        print("🔍 Fetching Play Store trends...")

        # Search for trending and rising apps
        trending_apps = search("top free apps", lang="en", country="in", n_hits=5)  # ✅ Fix: `num` instead of `n`
        rising_apps = search("new trending apps", lang="en", country="in", n_hits=5)

        # Fetch app details
        def get_app_details(app_list):
            return [
                {
                    "name": app_info["title"],
                    "icon": app_info["icon"],
                    "url": f"https://play.google.com/store/apps/details?id={app_info['appId']}"
                }
                for app_info in app_list
            ]

        return {
            "trending_apps": get_app_details(trending_apps),
            "rising_apps": get_app_details(rising_apps),
            "category": "Trending & New Apps"
        }
    except Exception as e:
        print(f"❌ Error fetching Play Store data: {e}")
        return {"error": f"Error fetching Play Store data: {str(e)}"}


# Dashboard view
def dashboard(request):
    user_id = request.session.get("user_id")
    if not user_id:
        return redirect("login")

    user = Signup.objects.get(id=user_id)
    last_login = UserActivity.objects.filter(user=user).order_by("-login_time").first()

    # 🔄 Fetch Real-Time Data
    youtube_data = fetch_youtube_data(user.youtube) or {}
    reddit_data = fetch_reddit_data(user.reddit) or {}
    playstore_data = get_playstore_trends() or {}

    # 🔄 Update Database with Fresh Data
    user.youtube_profile_pic = youtube_data.get("youtube_profile_pic", user.youtube_profile_pic)
    user.youtube_subscribers = youtube_data.get("youtube_subscribers", user.youtube_subscribers)
    user.youtube_total_videos = youtube_data.get("youtube_total_videos", user.youtube_total_videos)
    user.youtube_total_views = youtube_data.get("youtube_total_views", user.youtube_total_views)

    user.reddit_profile_pic = reddit_data.get("reddit_profile_pic", user.reddit_profile_pic)
    user.reddit_karma = reddit_data.get("reddit_karma", user.reddit_karma)
    user.reddit_awards_received = reddit_data.get("reddit_awards_received", user.reddit_awards_received)
    user.reddit_account_age = reddit_data.get("reddit_account_age", user.reddit_account_age)

    user.save()  # ✅ Save updated data in DB

    return render(request, "dashboard.html", {
        "user": user,
        "last_login": last_login.login_time if last_login else "Never Logged In",
        "playstore": playstore_data,
    })


def logout(request):
    request.session.flush()
    return redirect("index")


def extract_reddit_username(url):
    """Extracts Reddit username from the given profile URL"""
    parsed = urlparse(url)
    if parsed.netloc == "www.reddit.com":
        parts = parsed.path.strip("/").split("/")
        if len(parts) > 1 and parts[0] == "user":
            return parts[1]  # Returns the Reddit username
    return None

def extract_youtube_channel_id(youtube_url):
    """Extracts YouTube Channel ID using API"""
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        request = youtube.search().list(q=youtube_url, part="snippet", type="channel", maxResults=1)
        response = request.execute()

        if "items" not in response or not response["items"]:
            print("No channel found for this URL.")
            return None

        return response["items"][0]["id"]["channelId"]
    except Exception as e:
        print(f"Error fetching YouTube channel ID: {e}")
        return None

def signup_view(request):
    if request.method == "POST":
        name = request.POST.get("name")
        email = request.POST.get("email")
        youtube_url = request.POST.get("youtube")
        reddit_url = request.POST.get("reddit")
        password = request.POST.get("password")
        confirm_password = request.POST.get("confirm_password")

        # 🔴 Password Mismatch Check
        if password != confirm_password:
            messages.error(request, "Passwords do not match")
            return render(request, "signup.html")

        # 🔴 Check if Email Already Exists
        if Signup.objects.filter(email=email).exists():  #select user from tabel where email = "xy...";
            messages.error(request, "Email already registered")
            return render(request, "signup.html")

        # 🔴 Extract YouTube Channel ID
        youtube_channel_id = extract_youtube_channel_id(youtube_url)
        if not youtube_channel_id:
            messages.error(request, "Invalid YouTube link. Please enter a valid YouTube channel URL.")
            return render(request, "signup.html")

        # 🔴 Extract Reddit Username
        reddit_username = extract_reddit_username(reddit_url)
        if not reddit_username:
            messages.error(request, "Invalid Reddit link. Please enter a valid Reddit profile URL.")
            return render(request, "signup.html")

        # 🟢 Fetch Data from YouTube & Reddit APIs
        youtube_data = fetch_youtube_data(youtube_channel_id) or {}
        reddit_data = fetch_reddit_data(reddit_username) or {}

        # 🟢 Store Data with Safe Defaults
        user = Signup.objects.create(
            name=name,
            email=email,
            youtube=youtube_url,
            reddit=reddit_url,
            password=make_password(password),
            youtube_profile_pic=youtube_data.get("youtube_profile_pic", ""),
            youtube_subscribers=youtube_data.get("youtube_subscribers", 0),
            youtube_total_videos=youtube_data.get("youtube_total_videos", 0),
            youtube_total_views=youtube_data.get("youtube_total_views", 0),
            reddit_profile_pic=reddit_data.get("reddit_profile_pic", ""),
            reddit_karma=reddit_data.get("reddit_karma", 0),
            reddit_awards_received=reddit_data.get("reddit_awards_received", 0),
            reddit_account_age=reddit_data.get("reddit_account_age", 0),
        )

        # 🟢 Store User Session & Redirect to Dashboard
        request.session["user_id"] = user.id
        messages.success(request, "Signup successful! Redirecting to dashboard...")
        return redirect("dashboard")

    return render(request, "signup.html")


def login(request):
    if request.method == "POST":
        email, password = request.POST.get("email"), request.POST.get("password")

        try:
            user = Signup.objects.get(email=email)
            if check_password(password, user.password):
                UserActivity.objects.create(user=user, login_time=now())
                request.session["user_id"] = user.id

                # 🔄 Fetch Updated YouTube & Reddit Data
                new_youtube_data = fetch_youtube_data(extract_youtube_channel_id(user.youtube)) or {}
                new_reddit_data = fetch_reddit_data(extract_reddit_username(user.reddit)) or {}

                # 🔄 Update User Object Only if New Data Exists
                updated = False
                for key, value in {**new_youtube_data, **new_reddit_data}.items():
                    if value and getattr(user, key, None) != value:
                        setattr(user, key, value)
                        updated = True

                if updated:
                    user.save()  # ✅ Save only if data changed

                return redirect("dashboard")
            else:
                messages.error(request, "Invalid email or password")
        except Signup.DoesNotExist:
            messages.error(request, "Invalid email or password")
        except Exception as e:
            messages.error(request, "An error occurred while logging in. Please try again.")
            print(f"Login Error: {e}")  # ✅ Debugging

    return render(request, "login.html")


def fetch_reddit_data(username):
    try:
        user = reddit_api.redditor(username)
        account_age_days = (datetime.now(timezone.utc) - datetime.fromtimestamp(user.created_utc, timezone.utc)).days
        
        # Fetch recent post
        recent_post = None
        for submission in user.submissions.new(limit=1):  # Get latest post
            recent_post = submission
            break

        return {
            "reddit_username": user.name,
            "reddit_profile_pic": user.icon_img,
            "reddit_karma": user.comment_karma + user.link_karma,
            "reddit_awards_received": getattr(user, "awardee_karma", 0),
            "reddit_account_age": account_age_days,
            "reddit_recent_post_title": recent_post.title if recent_post else "No recent post",
            "reddit_recent_post_image": recent_post.url if recent_post else "",
            "reddit_recent_post_url": f"https://www.reddit.com{recent_post.permalink}" if recent_post else "",
        }
    except Exception as e:
        print(f"Error fetching Reddit data: {e}")
    return {}



def fetch_youtube_data(channel_id):
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

        # Fetch channel details
        request = youtube.channels().list(part="snippet,statistics,contentDetails", id=channel_id)
        response = request.execute()

        if "items" in response and response["items"]:
            data = response["items"][0]
            uploads_playlist_id = data["contentDetails"]["relatedPlaylists"]["uploads"]  # Get Uploads Playlist ID
            
            # Fetch latest video from the uploads playlist
            latest_video_request = youtube.playlistItems().list(
                part="snippet",
                playlistId=uploads_playlist_id,
                maxResults=1  # Only fetch the latest upload
            )
            latest_video_response = latest_video_request.execute()

            latest_video_data = latest_video_response["items"][0]["snippet"] if "items" in latest_video_response and latest_video_response["items"] else {}

            return {
                "youtube_channel_name": data["snippet"]["title"],
                "youtube_profile_pic": data["snippet"]["thumbnails"]["high"]["url"],
                "youtube_subscribers": int(data["statistics"].get("subscriberCount", 0)),
                "youtube_total_videos": int(data["statistics"].get("videoCount", 0)),
                "youtube_total_views": int(data["statistics"].get("viewCount", 0)),
                "youtube_latest_upload_url": f"https://www.youtube.com/watch?v={latest_video_data.get('resourceId', {}).get('videoId', '')}",
                "youtube_latest_upload_thumbnail": latest_video_data.get("thumbnails", {}).get("high", {}).get("url", ""),
                "youtube_latest_upload_title": latest_video_data.get("title", ""),
            }
    except Exception as e:
        print(f"Error fetching YouTube data: {e}")
    return {}

