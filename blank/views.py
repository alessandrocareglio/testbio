
  
#blank/views

import numpy as np
import os
import pandas as pd
from django.shortcuts import render
from django.http import HttpResponse
from .forms import SmilesForm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from rdkit import Chem
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from django.conf import settings
import base64
from io import BytesIO



############

def genera_grafico_base64(fig, format='png'):
    """Converte una figura matplotlib in base64."""
    buf = BytesIO()
    fig.savefig(buf, format=format)
    buf.seek(0)
    string = base64.b64encode(buf.read())
    return string.decode('utf-8')


def genera_scatterplot(df):
    try:
        if 'query_index' in df.columns and 'library_index' in df.columns:
            plt.figure(figsize=(10, 6))
            plt.scatter(df['query_index'], df['library_index'])
            plt.title('Scatterplot')
            plt.xlabel('Phytocomplex Index')
            plt.ylabel('Library Index')
            plt.tight_layout()

            fig = plt.gcf()  # Ottieni la figura corrente
            scatterplot_base64 = genera_grafico_base64(fig)  # Codifica in base64 PRIMA di chiudere la figura

            plt.close(fig)  # Chiudi la figura per liberare memoria
            return scatterplot_base64
        else:
            return "Colonne 'query_index' o 'library_index' non presenti."
    except Exception as e:
        return f"Errore nello scatterplot: {e}"







def genera_barplot(df):
    try:
        if 'target' in df.columns:
            target_counts = df['target'].value_counts()
            plt.figure(figsize=(12, 8))

            # Assegna x a hue e imposta legend=False
            sns.barplot(x=target_counts.index[:20], y=target_counts.values[:20], hue=target_counts.index[:20], palette="crest", legend=False)

            plt.title('Target Frequency (Top 20)')
            plt.xlabel('Target')
            plt.ylabel('Frequency')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            fig = plt.gcf()
            barplot_base64 = genera_grafico_base64(fig)
            plt.close(fig)
            return barplot_base64
        else:
            return "Colonna 'target' non presente."
    except Exception as e:
        return f"Errore nel barplot: {e}"

def genera_boxplot(df):
    try:
        if 'target' in df.columns and 'similarity' in df.columns:
            target_counts = df['target'].value_counts()
            plt.figure(figsize=(12, 8))
            sns.boxplot(x='target', y='similarity', data=df[df['target'].isin(target_counts.index[:10])])
            plt.title('Boxplot of Similarity by Target (Top 10)')
            plt.xlabel('Target')
            plt.ylabel('Tanimoto coefficient')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            fig = plt.gcf()
            boxplot_base64 = genera_grafico_base64(fig)
            plt.close(fig)
            return boxplot_base64
        else:
            return "Colonne 'target' o 'similarity' non presenti."
    except Exception as e:
        return f"Errore nel boxplot: {e}"



def genera_network_graph(df, tanimoto_threshold=0.3, font_size=12, node_size=100):
    try:
        if not df.empty and 'similarity' in df.columns and 'query_index' in df.columns and 'target' in df.columns:
            G = nx.Graph()
            for index, row in df.iterrows():
                tanimoto = row['similarity']
                if tanimoto is not None and tanimoto >= tanimoto_threshold:
                    G.add_edge(row['query_index'], row['target'], weight=tanimoto)

            if G.number_of_edges() > 0:
                plt.figure(figsize=(15, 10))

                # Layout a due colonne
                left_nodes = set(df['query_index'].unique())
                right_nodes = set(df['target'].unique())
                pos = {}

                # Posiziona i nodi sulla sinistra e sulla destra
                y_left = 0
                for node in left_nodes:
                    pos[node] = (-1, y_left)
                    y_left += 1

                y_right = 0
                for node in right_nodes:
                    pos[node] = (1, y_right)
                    y_right += 1

                nx.draw(G, pos, 
                        with_labels=True, 
                        node_size=node_size, 
                        font_size=font_size, 
                        width=[G[u][v]['weight'] * 3 if 'weight' in G[u][v] else 1 for u, v in G.edges()],
                        edge_color="gray", 
                        alpha=0.7,
                        node_color="skyblue")

                plt.title(f'Network graph (Relazioni tra Fitocomplessi e Target) - Tanimoto >= {tanimoto_threshold}')
                plt.tight_layout()
                fig = plt.gcf()
                network_graph_base64 = genera_grafico_base64(fig)
                plt.close(fig)
                return network_graph_base64
            else:
                return "Nessuna connessione trovata con la soglia di Tanimoto specificata."
        else:
            return "DataFrame vuoto o colonne necessarie mancanti. Impossibile generare il network graph."
    except Exception as e:
        return f"Errore nella generazione del network graph: {e}"



"""
def genera_network_graph(df, tanimoto_threshold=0.3, font_size=12, node_size=100):
    try:
        # ... (codice precedente)

        if G.number_of_edges() > 0:
            plt.figure(figsize=(20, 15))  # Aumenta le dimensioni del grafico
            pos = nx.spring_layout(G, k=0.3)  # Prova un layout diverso, ad esempio nx.kamada_kawai_layout
            nx.draw(G, pos, 
                    with_labels=True, 
                    node_size=node_size * 2,  # Aumenta la dimensione dei nodi
                    font_size=font_size, 
                    width=[G[u][v]['weight'] * 3 if 'weight' in G[u][v] else 1 for u, v in G.edges()],
                    edge_color="gray", 
                    alpha=0.7,
                    node_color="skyblue")  # Aggiungi colore ai nodi

            plt.title(f'Grafico Network (Relazioni tra Fitocomplessi e Target) - Tanimoto >= {tanimoto_threshold}')
            plt.tight_layout()
            fig = plt.gcf()
            network_graph_base64 = genera_grafico_base64(fig)
            plt.close(fig)
            return network_graph_base64
        else:
            return "Nessuna connessione trovata con la soglia di Tanimoto specificata."
        # ... (codice precedente)
    except Exception as e:
        return f"Errore nella generazione del network graph: {e}"







def genera_network_graph(df, tanimoto_threshold=0.3, font_size=12, node_size=100):
    try:
        if not df.empty and 'similarity' in df.columns and 'query_index' in df.columns and 'target' in df.columns:
            G = nx.Graph()
            for index, row in df.iterrows():
                tanimoto = row['similarity']
                if tanimoto is not None and tanimoto >= tanimoto_threshold:
                    G.add_edge(row['query_index'], row['target'], weight=tanimoto)

            if G.number_of_edges() > 0:
                plt.figure(figsize=(15, 10))
                pos = nx.spring_layout(G, k=0.3)
                nx.draw(G, pos, with_labels=True, node_size=node_size, font_size=font_size,
                        width=[G[u][v]['weight'] * 3 if 'weight' in G[u][v] else 1 for u, v in G.edges()],
                        edge_color="gray", alpha=0.7)
                plt.title(f'Grafico Network (Relazioni tra Fitocomplessi e Target) - Tanimoto >= {tanimoto_threshold}')
                plt.tight_layout()
                fig = plt.gcf()
                network_graph_base64 = genera_grafico_base64(fig)
                plt.close(fig)
                return network_graph_base64
            else:
                return "Nessuna connessione trovata con la soglia di Tanimoto specificata."
        else:
            return "DataFrame vuoto o colonne necessarie mancanti. Impossibile generare il network graph."
    except Exception as e:
        return f"Errore nella generazione del network graph: {e}"

"""


#############
def load_library_data(fingerprint_file, target_file, bioactivity_file):
    """Carica fingerprint e target da file."""
    try:
        with open(fingerprint_file, 'r') as f_fp, open(target_file, 'r') as f_t, open(bioactivity_file, 'r') as f_b:
            library_fingerprints = {}
            library_targets = {}
            library_bioactivities = {}  # Nuovo dizionario per le bioattività
            for i, (fp_line, target_line, bioactivity_line) in enumerate(zip(f_fp, f_t, f_b)):

                fp_line = fp_line.strip()
                target = target_line.strip()
                bioactivity = bioactivity_line.strip()
                library_bioactivities[i] = bioactivity  # Aggiungi bioattività
                if not fp_line:
                    continue
                fingerprint = [int(bit) for bit in fp_line.split(';')]
                library_fingerprints[i] = fingerprint
                library_targets[i] = target
        return library_fingerprints, library_targets, library_bioactivities
    except FileNotFoundError:
        return None, None
    except ValueError:
        return None, None


def salva_smiles(request):
    if request.method == 'POST':
        form = SmilesForm(request.POST)
        if form.is_valid():
            smiles_input = form.cleaned_data['smiles_input']
            smiles_list = smiles_input.strip().splitlines()

            fingerprints = calculate_fingerprints(smiles_list)  # Chiamata a calculate_fingerprints

            context = {
                'smiles_list': smiles_list,
                'fingerprints': fingerprints,
            }
            return render(request, 'results.html', context)
        else:
            return render(request, 'index.html', {'form': form})
    else:
        return render(request, 'index.html')




def calculate_fingerprints(smiles_list):
    """Calcola i fingerprint per una lista di SMILES."""
    fingerprints = []
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:  # Gestisci il caso in cui lo SMILES non è valido
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            arr = np.zeros((1,))
            DataStructs.ConvertToNumpyArray(fp, arr)
            fingerprints.append(arr.tolist()) #aggiunge il fingerprint convertito in lista alla lista dei fingerprint
        else:
            fingerprints.append(f"SMILES non valido: {smiles}")  # Aggiungi un messaggio di errore
    return fingerprints
############
# Modulo 1: Gestione degli SMILES
def parse_smiles(smiles_input):
    """Parsa l'input SMILES in una lista, gestendo spazi e righe vuote."""
    return [s.strip() for s in smiles_input.strip().splitlines() if s.strip()]

def smiles_to_mol(smiles):
    """Converte uno SMILES in un oggetto molecola RDKit, gestendo errori."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, f"SMILES non valido: {smiles}"
        return mol, None  # Nessun errore
    except Exception as e:
        return None, f"Errore nella conversione SMILES: {e}"

# Modulo 2: Calcolo dei Fingerprint
def calculate_fingerprint(mol):
    """Calcola il fingerprint per una molecola."""
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
    arr = np.zeros((1,))
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr.tolist()

def calculate_fingerprints(smiles_list):
    """Calcola i fingerprint per una lista di SMILES, gestendo errori."""
    fingerprints = []
    for smiles in smiles_list:
        mol, error = smiles_to_mol(smiles)
        if error:
            fingerprints.append(error)
        else:
            fingerprints.append(calculate_fingerprint(mol))
    return fingerprints

# Modulo 3: Caricamento Fingerprint da Libreria (Mantenuto, ma semplificato)
def load_library_fingerprints(file_path):
    """Carica i fingerprint da un file, gestendo errori di base."""
    try:
        with open(file_path, 'r') as f:
            library_fingerprints = {i: [int(bit) for bit in line.strip().split(';')] 
                                    for i, line in enumerate(f) if line.strip()}
        return library_fingerprints
    except (FileNotFoundError, ValueError, IndexError): #gestione di più errori in una volta sola
        return None

# Modulo 4: Views Django
def index(request):
    """View per mostrare il form."""
    form = SmilesForm()
    return render(request, 'index.html', {'form': form})





############
def tanimoto_similarity(fp1, fp2):
    """Calcola la similarità di Tanimoto tra due fingerprint."""
    if not isinstance(fp1, np.ndarray):
        fp1=np.array(fp1)
    if not isinstance(fp2, np.ndarray):
        fp2=np.array(fp2)
    intersection = np.sum(np.logical_and(fp1, fp2))
    union = np.sum(np.logical_or(fp1, fp2))
    if union == 0:
        return 0.0  # Gestione del caso in cui entrambi i fingerprint sono vuoti
    return float(intersection) / union





def confronta_fingerprints(query_fingerprints, library_fingerprints, library_targets, library_bioactivities):
    """Confronta i fingerprint e aggiunge i target."""
    results = []
    if library_fingerprints is None or library_targets is None or library_bioactivities is None:
        return "Errore nel caricamento dei dati della libreria."
    if isinstance(query_fingerprints, str): #gestione dell'errore se la query è una stringa
        return query_fingerprints
    for i, query_fp in enumerate(query_fingerprints):
        if isinstance(query_fp, str): #gestione dell'errore se un singolo fingerprint è una stringa
            results.append({"query_index": i, "library_index": "N/A", "target":"N/A","similarity": query_fp, "bioactivity": "N/A"})
            continue

        similarities = []
        for lib_index, lib_fp in library_fingerprints.items():
            try:
                similarity = tanimoto_similarity(np.array(query_fp), np.array(lib_fp))
                similarities.append((lib_index, library_targets[lib_index], similarity)) # Aggiunto il target
            except ValueError as e:
                return f"Errore nel calcolo della similarità per il fingerprint {i} della query: {e}"

        # Ordina per similarità decrescente
        top_matches = sorted(similarities, key=lambda x: x[2], reverse=True)[:5]

        for lib_index, target, similarity in top_matches:
            bioactivity = library_bioactivities.get(lib_index, "N/A")  # Valore predefinito se la chiave non esiste
            results.append({
                "query_index": i,
                "library_index": lib_index,
                "target": target,  # Aggiunto il target ai risultati
                "similarity": similarity,
                "bioactivity":bioactivity ,
            })
    return results


def process_smiles(request):
    if request.method == 'POST':
        form = SmilesForm(request.POST)
        if form.is_valid():
            smiles_input = form.cleaned_data['smiles_input']
            smiles_list = parse_smiles(smiles_input)
            query_fingerprints = calculate_fingerprints(smiles_list)

            fingerprint_file = os.path.join("data", "fingerprint_Approveddrugslibrary.txt")
            target_file = os.path.join("data", "targets.txt")
            bioactivity_file = os.path.join("data", "bioactivities.txt")  # Aggiungi il file
            
            
            library_fingerprints, library_targets, library_bioactivities = load_library_data(fingerprint_file, target_file, bioactivity_file)
           
            comparison_results = confronta_fingerprints(query_fingerprints, library_fingerprints, library_targets, library_bioactivities)

            if isinstance(comparison_results, str): #gestione dell'errore se comparison_results è una stringa
                context = {'error_message': comparison_results}
                return render(request, 'results.html', context)

            df_results = pd.DataFrame(comparison_results) #creazione dataframe per excell
            # Genera i grafici e ottieni i nomi dei file (o messaggi di errore)
            graph_filenames = {}
            graph_filenames["scatter"] = genera_scatterplot(df_results)
            graph_filenames["bar"] = genera_barplot(df_results)
            graph_filenames["box"] = genera_boxplot(df_results)
            network_graph_filename = genera_network_graph(df_results)

            
            
            excel_file_path = os.path.join("data", "confronto_risultati.xlsx") #salvataggio excell
            df_results.to_excel(excel_file_path, index=False) #salvataggio excell
            context = {
                'graph_filenames': graph_filenames, # Passa i nomi dei file al template
                'network_graph_filename':network_graph_filename,
                'comparison_results': comparison_results,
            }
            return render(request, 'results.html', context)
        else:
            return render(request, 'index.html', {'form': form})
    return render(request, 'index.html')


    ##############
    