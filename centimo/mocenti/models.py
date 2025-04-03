from django.db import models
from django.utils.timezone import now

class Signup(models.Model):
    name = models.CharField(max_length=122)
    email = models.CharField(max_length=122, unique=True)
    youtube = models.CharField(max_length=255, null=True, blank=True, default="No YouTube Link")
    reddit = models.CharField(max_length=255, null=True, blank=True, default="No Reddit Link")
    password = models.CharField(max_length=255)

    # YouTube Data Caching Fields   
    youtube_profile_pic = models.URLField(max_length=500, null=True, blank=True)
    youtube_channel_name = models.CharField(max_length=255, null=True, blank=True, default="No Username")
    youtube_subscribers = models.IntegerField(null=True, blank=True)
    youtube_total_videos = models.IntegerField(null=True, blank=True)
    youtube_total_views = models.BigIntegerField(null=True, blank=True)
    youtube_latest_upload_url = models.URLField(max_length=500, null=True, blank=True)  # 🔴 Recent Upload Video URL
    youtube_latest_upload_thumbnail = models.URLField(max_length=500, null=True, blank=True)  # 🔴 Recent Upload Thumbnail
    youtube_latest_upload_title = models.CharField(max_length=255, null=True, blank=True)  # 🔴 Recent Upload Title
    youtube_last_updated = models.DateTimeField(null=True, blank=True)

    # Reddit Data Caching Fields
    reddit_profile_pic = models.URLField(max_length=500, null=True, blank=True)
    reddit_username=models.CharField(max_length=255, null=True, blank=True, default="No Username")
    reddit_karma = models.IntegerField(null=True, blank=True)
    reddit_awards_received = models.IntegerField(null=True, blank=True)
    reddit_account_age = models.IntegerField(null=True, blank=True)  # Account age in days
    reddit_last_updated = models.DateTimeField(null=True, blank=True)
    reddit_recent_post_title = models.CharField(max_length=500, null=True, blank=True, default="No recent post")
    reddit_recent_post_image = models.URLField(max_length=500, null=True, blank=True)
    reddit_recent_post_url = models.URLField(max_length=500, null=True, blank=True)

    def __str__(self):
        return self.name

class UserActivity(models.Model):
    user = models.ForeignKey(Signup, on_delete=models.CASCADE)
    login_time = models.DateTimeField(auto_now_add=True)
    logout_time = models.DateTimeField(null=True, blank=True)

    def __str__(self):
        return f"{self.user.name} - {self.login_time}"

class Feedback(models.Model):
    name = models.CharField(max_length=122)
    email = models.EmailField(max_length=255)
    message = models.TextField()
    submitted_at = models.DateTimeField(default=now)

    def __str__(self):
        return f"{self.name} - {self.email} ({self.submitted_at})"
