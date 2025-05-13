# Formalist IA

## Description
Formalist IA est une solution d'analyse de documents notariés (exemple : attestation de propriété) qui automatise l'extraction d'informations clés à partir de fichiers PDF en combinant :
- **Classification de tokens** pour identifier les entités linguistiques (noms, dates, montants, etc.).
- **Classification de séquences** pour catégoriser des segments de texte selon des attributs métiers.

## Fonctionnalités
- **Builder de dataset JSON** : conversion de documents PDF en jeu de données annotées pour l'entraînement citeturn0file3  
- **Entraînement IA** : fine-tuning de modèles pré-entraînés (DistilBERT, BERT, Mistral) pour la classification de tokens et de séquences citeturn0file2  
- **Pipeline d’inférence** : chargement de PDFs utilisateurs, extraction de textes, classification token & séquence, et export JSON citeturn0file0  
- **Apprentissage continu** : architecture modulaire permettant d’ajouter facilement de nouvelles données et de réentraîner les modèles citeturn0file4  

## Avancement du projet
- **Documents analysés** : 233 (coût de la fonction de perte finale : 1,5)  
- **Objectif** : atteindre au moins 5000 documents pour garantir la robustesse du modèle citeturn0file1  
- Implémentation des algorithmes de création de données, d'entraînement et d'inférence  
- Stratégies : prioriser les documents notariés courants, paralléliser création de données et validation, utiliser des agents IA, renforcer l'équipe.

## Architecture du projet
```
.
├── data/                     # PDFs bruts et jeux de données JSON
├── model/
│   ├── tok_clf_model/        # Modèle classification de tokens
│   └── seq_clf_model/        # Modèle classification de séquences
├── pdf_input/                # Entrée des fichiers PDF pour l’inférence
├── json_output/              # Sortie JSON post-inférence
├── main.py                   # Pipeline d’inférence (§ Inférence) citeturn0file0
├── JSON Dataset builder.py   # Génération du dataset JSON (§ Étiquetage) citeturn0file3
├── Training Model.py         # Entraînement des modèles (§ Apprentissage IA) citeturn0file2
└── README.md

```

## Prérequis
- Python 3.8 ou supérieur  
- Librairies : `transformers`, `datasets`, `pdfplumber`, `tqdm`, `matplotlib`, `numpy`, etc.  
- Clé d’API HuggingFace configurée via `.env`.

## Installation
```bash
git clone https://votre-repo/Formalist-IA.git
cd Formalist-IA
pip install -r requirements.txt
```

## Utilisation

### 1. Génération du dataset
```bash
python "JSON Dataset builder.py"
```

### 2. Entraînement des modèles
```bash
python "Training Model.py"
```

### 3. Inférence
```bash
python main.py
```

Les résultats JSON sont générés dans le dossier `json_output/`.

## Contribuer
Les contributions sont les bienvenues ! Merci d’ouvrir une issue ou de soumettre un pull request.

## Auteur
- CUZOU Alexandre

## Licence
MIT License
