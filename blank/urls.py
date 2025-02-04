"""
from django.urls import path
from . import views

urlpatterns = [
    path('', views.index, name='index'), #url per la pagina index
    #path('process/', views.process_smiles, name='process_smiles'), #url per process smiles
    path('salva_smiles/', views.salva_smiles, name='salva_smiles'),
]
"""
#blank/urls

from django.urls import path
from . import views

app_name = 'blank'  # Aggiungi questa riga!

urlpatterns = [
    path('', views.index, name='index'),  # URL per il form
    path('results/', views.process_smiles, name='process_smiles'),  # URL per la view di processamento e risultati
]    







