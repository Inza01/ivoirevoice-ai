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
configurations de données, modèles et expériences décrivent les travaux futurs
sans déclencher de téléchargement.

## Frontières

- `data/` : futurs chargements et contrôles des données ;
- `models/` : contrat, registre et backends ;
- `evaluation/` : futures métriques reproductibles ;
- `api/` et `ui/` : adaptateurs d'entrée, sans logique ML spécifique.

Le stockage des données, checkpoints et sorties reste hors Git.

