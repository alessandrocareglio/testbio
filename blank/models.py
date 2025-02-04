from django.db import models

class YourModelName(models.Model):
    # Existing model fields ...
    bioactivity = models.TextField(max_length=1024, blank=True)  # Adjust max_length as needed
# Create your models here.
