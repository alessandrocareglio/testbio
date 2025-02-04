 # blank/forms.py


from django import forms

class SmilesForm(forms.Form):
    smiles_input = forms.CharField(widget=forms.Textarea(attrs={'rows': 15, 'cols': 80}), label="Insert SMILES (one per line):")