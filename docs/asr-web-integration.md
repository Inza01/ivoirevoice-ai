# Intégration ASR de la plateforme web

## Statut

La Phase 2A raccorde la page web **Transcrire** à l'adaptateur FastAPI
versionné, puis au service ASR existant. Le parcours est réel, mais reste
`experimental` et strictement local : il ne constitue ni une API Internet
durcie, ni une nouvelle évaluation scientifique des modèles.

Cette intégration ne modifie pas les poids, les métriques, les splits ou le
protocole d'évaluation. Elle ne lit aucun corpus interne et ne possède aucune
voie vers le final holdout. Chaque requête traite uniquement le fichier fourni
explicitement par l'utilisateur.

## Flux d'exécution

```text
Navigateur
  |
  | multipart : audio + language + model
  v
Route Next.js same-origin /api/backend/*
  | allowlist de routes, origine privée, limites et timeout
  v
FastAPI /api/v1/transcriptions
  | validation + fichier temporaire aléatoire
  v
TranscriptionService (une inférence à la fois)
  |
  v
ModelRegistry -> WhisperBackend -> load / transcribe / unload
  |
  v
Réponse publique assainie, puis suppression immédiate de l'audio
```

Le navigateur ne reçoit jamais l'origine FastAPI interne, le chemin d'un
checkpoint, le cache de modèles, le matériel ou le nom de fichier temporaire.
Le serveur Next.js ne transmet ni cookie ni en-tête d'autorisation du
navigateur. FastAPI n'importe aucun module d'entraînement ou d'évaluation.

## Contrats HTTP

Les routes historiques restent disponibles pour compatibilité. Le nouveau
client utilise uniquement :

| Méthode | Route | Rôle |
|---|---|---|
| `GET` | `/api/health` | vivacité sans création ou chargement de modèle |
| `GET` | `/api/v1/languages` | codes `fr`, `en`, `dyu` et capacités publiques |
| `GET` | `/api/v1/models` | identifiants, libellés, statuts et langues compatibles |
| `POST` | `/api/v1/transcriptions` | transcription synchrone d'un upload temporaire |

Le `POST` utilise les champs multipart `audio`, `language` et `model`. La
réponse fournit un identifiant UUID aléatoire, l'état `completed`, la langue,
l'identifiant public du modèle, le texte, la durée audio, le temps de
traitement et le RTF. Elle ne réutilise jamais le hash de l'audio comme
identifiant public et n'expose jamais `TranscriptionResult.model_name`, car le
backend final peut y contenir un chemin local.

Les erreurs traversent les frontières sous la forme d'un code public issu
d'une allowlist. Les messages backend, exceptions, détails de validation,
tracebacks, chemins et contenus soumis sont supprimés. Le frontend traduit
uniquement ces codes contrôlés via ses catalogues FR/EN.

## Langues et modèles

Le registre runtime FastAPI est la source de vérité des compatibilités :

| Modèle public | `fr` | `en` | `dyu` | Statut |
|---|---:|---:|---:|---|
| Whisper Tiny — Baseline | oui | oui | oui | `experimental` |
| Whisper Small — Baseline | oui | oui | oui | `experimental` |
| Whisper Tiny — Dioula Final | non | non | oui | `experimental` |

Les baselines reçoivent explicitement la langue `fr` ou `en` pendant le
décodage. Pour `dyu`, aucun token de langue Whisper n'est forcé ; la tâche
reste toujours `transcribe`. Le modèle final est chargé exclusivement depuis
`IVOIREVOICE_DIOULA_FINAL_MODEL_PATH` et demeure inchangé et hors Git.

L'option automatique reste visible mais désactivée et marquée expérimentale.
La traduction reste `coming_soon` : aucun texte traduit n'est simulé.

## Politique d'upload et de rétention

Le lot accepte uniquement des conteneurs réellement décodables par la stack
SoundFile/libsndfile validée :

| Extension | MIME déclarés acceptés | Conteneur décodé attendu |
|---|---|---|
| `.wav` | `audio/wav`, `audio/x-wav`, `audio/wave` | WAV/WAVEX/RF64 |
| `.mp3` | `audio/mpeg`, `audio/mp3` | MP3 |
| `.flac` | `audio/flac`, `audio/x-flac` | FLAC |
| `.ogg` | `audio/ogg`, `application/ogg` | OGG |

M4A, MP4 audio et WebM ne sont pas annoncés : le décodeur actuel ne les
garantit pas. Les limites sont 25 Mio et 30 secondes. La limite précédente de
180 secondes n'est pas conservée pour l'API web, car la configuration Whisper
actuelle n'active ni chunking ni timestamps longs ; la dépasser demanderait un
changement de décodage distinct et évalué.

La validation serveur combine :

1. extension et MIME cohérents ;
2. taille bornée en lecture par blocs ;
3. signature binaire du conteneur ;
4. décodage SoundFile, durée finie et positive, fréquence, frames et canaux
   valides ;
5. cohérence entre extension et format réellement décodé.

Le nom client n'est jamais utilisé comme chemin. Un répertoire temporaire
privé et un nom UUID sont créés hors du dépôt. Le fichier et le répertoire sont
supprimés dans tous les cas, y compris validation invalide, échec de modèle ou
annulation. La seule politique autorisée est `delete_immediately`. L'audio et
la transcription ne servent jamais à l'entraînement.

## Cycle modèle et concurrence

`TranscriptionService` conserve le cycle paresseux existant et ajoute une
sérialisation par processus : création, chargement, transcription et
déchargement sont protégés par un verrou, avec `unload()` dans un `finally`.
FastAPI exécute cette section bloquante hors de l'event loop.

Le runtime GPU doit utiliser un seul worker Uvicorn. Plusieurs workers auraient
chacun leur verrou et pourraient charger plusieurs modèles simultanément. Les
endpoints de vivacité et de découverte restent metadata-only : ils ne créent
ni backend, ni contexte CUDA, ni téléchargement.

## Interface web

La page `/transcribe` charge les langues et modèles depuis les endpoints de
découverte, puis filtre le modèle par compatibilité. Elle couvre les états
chargement, vide, fichier sélectionné, traitement, succès et erreur. Elle
offre :

- sélection ou glisser-déposer accessible ;
- transcription avec prévention des doubles soumissions ;
- texte, langue, modèle public, durée, temps et RTF ;
- copie et exports TXT/JSON à champs explicitement autorisés ;
- effacement local ;
- action Traduire désactivée avec statut `coming_soon`.

Les exports ne contiennent ni nom de fichier, chemin, checkpoint, identifiant
participant ou métadonnée matérielle. Les URL Blob sont révoquées. Le
microphone reste désactivé dans ce lot, car sa capture et ses permissions
nécessitent une revue dédiée.

## Configuration locale

Les chemins restent fournis au runtime et ne sont jamais écrits dans le dépôt :

```bash
export IVOIREVOICE_MODEL_CACHE_DIR=/path/outside/repository/cache/models
export IVOIREVOICE_DIOULA_FINAL_MODEL_PATH=/path/outside/repository/checkpoint-002052
export IVOIREVOICE_API__MAX_UPLOAD_SIZE_MB=25
export IVOIREVOICE_API__AUDIO_RETENTION=delete_immediately
export IVOIREVOICE_API_INTERNAL_URL=http://127.0.0.1:8000
```

Lancer les deux processus dans deux terminaux :

```bash
make api
make web
```

FastAPI et Next.js écoutent sur loopback par défaut. `make demo` continue de
lancer Gradio sans changement.

## Tests et exploitation

Les tests utilisent uniquement des WAV synthétiques et des backends factices.
Ils ne téléchargent aucun poids et ne lisent aucun corpus, checkpoint ou
holdout. Ils couvrent les contrats, compatibilités, erreurs, signatures,
limites, nettoyage, sérialisation, journaux assainis, états UI, exports et
accessibilité structurelle.

Un smoke réel FR/EN exige un audio parlé synthétique ou sous licence explicite
et un cache local autorisé. Sans cet actif, les tests de contrat ne doivent pas
être présentés comme une validation linguistique. Pour le dioula, le chargement
et le contrat suffisent tant qu'aucun audio externe autorisé n'est disponible.

Le runner gardé refuse le dépôt et les racines privées connues. Une fois les
deux serveurs actifs et les fichiers externes préparés :

```bash
export IVOIREVOICE_WEB_ASR_SMOKE_FR_AUDIO_PATH=/safe/external/french.wav
export IVOIREVOICE_WEB_ASR_SMOKE_EN_AUDIO_PATH=/safe/external/english.wav
export IVOIREVOICE_WEB_ASR_SMOKE_CONFIRMATION=SAFE_EXTERNAL_OR_SYNTHETIC_AUDIO_NOT_FROM_PROJECT_DATA
make web-asr-smoke
```

`IVOIREVOICE_WEB_ASR_SMOKE_DYU_AUDIO_PATH` est facultative. En son absence, le
runner vérifie seulement que le modèle final annonce le contrat `dyu` et ne va
chercher aucun audio dans le projet. Il n'affiche jamais la transcription.

## Limites connues

- le traitement HTTP est synchrone ; une file persistante est une évolution
  future ;
- le proxy Next.js borne mais bufferise le multipart en mémoire ;
- il n'existe ni authentification, ni rate limiting, ni quota utilisateur ;
- aucun déploiement Internet n'est autorisé par ce lot ;
- les baselines FR/EN ne disposent pas d'un benchmark final publié ;
- l'accessibilité automatisée jsdom ne remplace pas une revue navigateur,
  clavier, zoom et lecteur d'écran ;
- l'intégrité gelée du checkpoint final reste contrôlée par le préflight local
  de démonstration, pas par la route de vivacité.

Avant toute exposition à des utilisateurs non fiables, une revue humaine doit
définir authentification, consentement, rétention, quotas, CORS, observabilité
privacy-safe, sécurité des en-têtes et capacité GPU.
