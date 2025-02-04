from django.contrib.sitemaps import Sitemap
from django.urls import reverse

class TestBioSitemap(Sitemap):
    changefreq = 'weekly'
    priority = 0.8

    def items(self):
        return ['index', 'process_smiles'] # Usa i nomi delle viste, non gli URL

    def location(self, item):
        return reverse(f'blank:{item}')  # Usa 'blank' (il nome corretto dell'app)
    
       
   