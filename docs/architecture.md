# Architecture

## Principes

Le cœur dépend du contrat `ASRBackend`, pas de Whisper, Wav2Vec2 ou d'un
framework ML. Chaque backend expose son nom, ses langues, son cycle de vie et
retourne un `TranscriptionResult` commun. `ModelRegistry` associe un nom stable
à une fabrique ; il permet d'ajouter plus tard un backend ou une langue sans
modifier l'API, l'interface ou le schéma de résultat.

```text
FastAPI                    Gradio
   |                          |
 Dummy            ComparisonService
                              |
                    TranscriptionService
                              |
                       ModelRegistry
                              |
                         ASRBackend
                              |
              Whisper Tiny / Small / Tiny adapté pilote
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

La Phase 3A.1 ajoute :

- `curation.py` pour l'éligibilité, les variantes textuelles, la
  déduplication chemin/SHA-256 et la quarantaine des conflits ;
- `split_comparison.py` pour comparer les stratégies 17/2/2, 15/3/3 et la
  recherche orientée durée ;
- `recovery.py` pour une conversion ffmpeg doublement verrouillée, jamais
  exécutée implicitement.

```text
manifeste d'audit
       |
 éligibilité + variantes textuelles
       |
 déduplication audio_path
       |
 déduplication SHA-256 ---- conflits -> quarantaine
       |
 candidat unique, split vide
       |
comparaison A / B / C -> recommandation à valider humainement
```

La Phase 3A.2 ajoute `freeze.py`, qui applique l'affectation B validée, dérive
la cible MVP sans le marqueur `↘`, puis écrit un manifeste local immuable et
ses preuves de provenance.

```text
candidat unique + comparaison des splits
                 |
       stratégie B validée (15/3/3)
                 |
 conservation des textes + intonation_falling
                 |
 cible sans tons et sans ↘
                 |
 validations bloquantes (unicité, fuite, confidentialité)
                 |
 dataset dioula v0.1 local + métadonnées + rapports
```

Le validateur rouvre les sorties et les compare au candidat source, à
l'affectation des locuteurs, au hash de la configuration et aux métadonnées.
Un artefact v0.1 existant ne peut être remplacé que par un contenu strictement
identique.

## Baselines ASR dioula

La Phase 4A ajoute un backend Whisper chargé paresseusement et une chaîne
d'évaluation locale qui ne force pas de token de langue `dyu`.

```text
manifest v0.1 + métadonnées immuables
                  |
       contrôle du hash et de la gouvernance
                  |
      sélection test smoke / pilot / full
                  |
       Whisper épinglé + audio 16 kHz
                  |
 prédictions privées enregistrées progressivement
                  |
 WER / CER / locuteurs / latence / RTF
                  |
       synthèse agrégée sans transcription
```

`environment.py` décrit le runtime sans chemin personnel,
`compatibility.py` consigne les modèles retenus ou exclus, `baseline.py`
valide le split test et gère la reprise, `metrics.py` calcule les agrégats, et
`comparison.py` refuse de comparer des pilotes issus de sélections différentes.
La révision du modèle, le commit Git, l'état propre ou sale du dépôt et
l'empreinte SHA-256 des sources sont conservés avec chaque run.

Les trois niveaux sont verrouillés : six audios pour `smoke`, 150 pour
`pilot`, et le test entier uniquement avec une confirmation explicite. Les
tests unitaires injectent une pipeline factice et ne téléchargent aucun poids.

## Interface de démonstration

La Phase 5A place toute la logique hors de Gradio :

```text
Gradio Blocks
     |
ComparisonService  ---- isolement des erreurs modèle par modèle
     |
TranscriptionService ---- validation taille, durée, extension et audio_id
     |
ModelRegistry ---- fabriques paresseuses configurées en YAML
     |
ASRBackend ---- load / transcribe / unload séquentiels
     |
EvaluationService ---- normalisation, WER, CER et différences
     |
ExportService ---- JSON, CSV, TXT et aperçus temporaires anonymisés
```

`configs/ui/models.yaml` contient les deux baselines et l'entrée Tiny adaptée
pilote. Le checkpoint reste hors Git ; son emplacement est résolu uniquement
depuis `IVOIREVOICE_DIOULA_PILOT_MODEL_PATH`. Si cette variable manque, l'échec
du pilote reste isolé et les deux baselines demeurent utilisables.

Le benchmark lit exclusivement les rapports JSON agrégés et sépare la
validation pilote de 600 audios du pilote historique de 150 audios. L'analyse
d'erreurs lit les prédictions privées depuis `IVOIREVOICE_ARTIFACTS_DIR` et
n'affiche qu'un identifiant anonymisé. Un aperçu audio est copié vers un
fichier temporaire au nom anonymisé, puis supprimé. Le serveur écoute sur
l'interface loopback et le partage public Gradio est explicitement désactivé.

## Frontières

- `data/` : découverte, validation et contrats des corpus ;
- `models/` : contrat, registre et backends ;
- `evaluation/` : sélections, métriques, diagnostics et comparaisons ;
- `services/` : orchestration, métriques applicatives et exports ;
- `api/` et `ui/` : adaptateurs d'entrée, sans logique ML spécifique.

Le stockage des données, checkpoints et sorties reste hors Git.

## Évaluation finale one-time

La voie finale est isolée du comparateur historique :

```text
final_model_manifest + approval receipt + clean Git
                         |
              metadata-only preflight
                         |
                       SEALED
                         |
        exact confirmation + final refit model only
                         |
             EVALUATION_IN_PROGRESS
                         |
       streaming counters, no per-item persistence
                    /           \
             EVALUATED    FAILED_AFTER_ACCESS
```

`one_time_final_holdout.py` est le seul entry point officiel. Il prépare le
modèle avant la frontière d'accès, puis agrège directement les erreurs, durées,
losses et locuteurs en mémoire. Les états post-accès sont terminaux. La voie
historique à trois modèles reste bloquée avant le chargement du contexte.

## Dépendances autorisées

Les dépendances inter-domaines existantes sont des invariants vérifiés par
`scripts/check_harness.py` :

```text
api        -> config, exceptions, models, services
data       -> exceptions
models     -> exceptions
evaluation -> data, exceptions, models
training   -> data, evaluation, exceptions, models
services   -> data, evaluation, exceptions, models
ui         -> exceptions, services
```

Les imports internes à un domaine restent libres. Une nouvelle direction
nécessite une décision dans un plan d'exécution, une mise à jour de cette
architecture et du validateur dans le même changement. Les imports absolus et
relatifs sont contrôlés afin qu'un changement de syntaxe ne contourne pas la
frontière.
