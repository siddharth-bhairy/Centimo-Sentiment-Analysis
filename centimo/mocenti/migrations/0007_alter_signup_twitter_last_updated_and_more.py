# Generated by Django 5.1.5 on 2025-02-07 15:15

import django.utils.timezone
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('mocenti', '0006_signup_youtube_last_updated_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='signup',
            name='twitter_last_updated',
            field=models.DateTimeField(blank=True, default=django.utils.timezone.now, null=True),
        ),
        migrations.AlterField(
            model_name='signup',
            name='youtube_last_updated',
            field=models.DateTimeField(blank=True, default=django.utils.timezone.now, null=True),
        ),
    ]
