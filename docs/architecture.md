# Architecture

## Principes

Le cœur dépend du contrat `ASRBackend`, pas de Whisper, Wav2Vec2 ou d'un
framework ML. Chaque backend expose son nom, ses langues, son cycle de vie et
retourne un `TranscriptionResult` commun. `ModelRegistry` associe un nom stable
à une fabrique ; il permet d'ajouter plus tard un backend ou une langue sans
modifier l'API, l'interface ou le schéma de résultat.

```text
FastAPI / Gradio
       |
 ModelRegistry
       |
   ASRBackend
       |
 Dummy (actuel) / Whisper (futur) / Wav2Vec2 (futur)
```

La configuration principale vient de `configs/project.yaml`. Les variables
`IVOIREVOICE_SECTION__CHAMP` la surchargent avant validation. Les
configurations de données, modèles et expériences ne déclenchent aucun
téléchargement.

## Pipeline dioula

```text
corpus brut (lecture seule)
          |
      discovery
          |
 clips parser ---- suppression immédiate des paramètres audioSrc
          |
 matching WAV limité au dossier du locuteur
          |
 soundfile.info + SHA-256 en flux
          |
 manifeste provisoire ---- audit agrégé ---- proposition de split
          |
 artefacts hors Git, chemins relatifs et identifiants pseudonymisés
```

- `settings.py` valide les racines fournies uniquement par variables
  d'environnement ;
- `discovery.py` inventorie la structure et infère les dossiers de locuteurs ;
- `clips.py` conserve le texte brut, produit NFC et normalise seulement les
  espaces ;
- `matching.py` préfère le WAV converti relié par `sequence`, puis applique nom
  exact, `sentence_id` et `clip_id` ;
- `audio.py` inspecte un WAV à la fois et peut calculer son SHA-256 en flux ;
- `manifest.py` construit le CSV canonique sans split définitif ;
- `audit.py` produit les statistiques et rapports ;
- `split.py` propose des groupes de locuteurs disjoints.

## Frontières

- `data/` : découverte, validation et contrats des corpus ;
- `models/` : contrat, registre et backends ;
- `evaluation/` : futures métriques reproductibles ;
- `api/` et `ui/` : adaptateurs d'entrée, sans logique ML spécifique.

Le stockage des données, checkpoints et sorties reste hors Git.
