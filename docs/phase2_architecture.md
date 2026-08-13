# Architecture Phase 2 — plateforme linguistique IvoireVoice

## 1. Vision

La Phase 2 fait évoluer le démonstrateur ASR local vers une plateforme web
multilingue réunissant transcription, traduction, apprentissage et, plus tard,
communauté. Le produit reste **IvoireVoice** ; Transcribe, Translate, Learn et
Community désignent des espaces fonctionnels, pas des marques indépendantes.

Cette évolution est incrémentale. Elle utilise le moteur ASR validé au lieu de
le reconstruire. Le modèle Whisper Tiny Dioula Final `checkpoint-002052` reste
gelé, local et non publiable. Aucun entraînement, refit ou nouvel accès au
final holdout ne fait partie de cette architecture.

Les statuts fonctionnels ont une signification stricte :

- `available` : capacité effectivement raccordée et utilisable dans le périmètre
  annoncé ;
- `experimental` : capacité réelle mais limitée, sans promesse de robustesse
  générale ;
- `coming_soon` : interface ou contrat préparé, mais aucun moteur utilisable.

Une page présente un état `coming_soon` plutôt qu'un résultat simulé. Le
contenu pédagogique Dioula reste marqué `DEMO`, `PLACEHOLDER` ou
`À VALIDER PAR UN EXPERT LINGUISTIQUE` jusqu'à sa revue.

## 2. Architecture logique et audit de l'existant

La frontière cible est la suivante :

```text
Navigateur
   |
   v
Next.js / React — présentation, navigation, formulaires, accessibilité
   |
   v
FastAPI /api/v1 — validation, contrats HTTP, orchestration
   |
   v
Services applicatifs
   |-- ASR service
   |-- Translation service
   |-- Learning service
   |-- Exercise service
   |-- User service
   `-- Community service
   |
   +-- ModelRegistry / ASRBackend / TranslationProvider
   +-- PostgreSQL
   `-- ObjectStorage (local, puis S3/MinIO)
```

Le navigateur et Next.js ne chargent jamais PyTorch, Transformers ou un
checkpoint. FastAPI ne doit pas importer `training/`, lire un manifeste ML ou
accéder à un split. Les services applicatifs constituent la seule voie vers les
adaptateurs ML, la base et le stockage.

### What can be reused

- `src/ivoirevoice/models/base.py` : contrat `ASRBackend` et résultat
  `TranscriptionResult` ;
- `src/ivoirevoice/models/registry.py` : registre injectable à fabriques
  paresseuses ;
- `src/ivoirevoice/models/whisper.py` : backend Whisper local, épinglé et
  chargeable/libérable à la demande ;
- `src/ivoirevoice/services/transcription_service.py` : catalogue, validation
  audio et orchestration séquentielle ;
- `src/ivoirevoice/services/comparison_service.py` : isolement des échecs d'un
  modèle sans perdre les autres sorties ;
- `src/ivoirevoice/services/evaluation_service.py` : normalisation et métriques
  lorsqu'une référence est fournie explicitement ;
- `src/ivoirevoice/api/schemas.py`, l'enveloppe `APIError` et la fabrique
  FastAPI injectable comme bases de contrats HTTP ;
- `DummyBackend`, les WAV synthétiques et les tests ASGI pour tester sans
  télécharger de modèle ni lire le corpus ;
- le harness, l'audit de publication et les variables d'environnement validées.

### What should be isolated

- l'interface Gradio et ses loaders d'artefacts privés restent une application
  legacy locale ;
- les uploads utilisateur utilisent un espace temporaire dédié, jamais les
  racines du corpus, des artefacts d'expérimentation ou du final holdout ;
- l'inférence GPU passe par une file bornée à un modèle actif par worker ;
- traduction, pédagogie, comptes et communauté sont distincts de l'ASR ;
- le stockage persistant est derrière un port applicatif, sans chemin local
  exposé dans les réponses ;
- les dépendances Node restent dans `web/` et ne deviennent pas des dépendances
  du package Python.

### What should be replaced progressively

- les routes de démonstration non versionnées `/health`, `/models` et
  `/transcribe` par des routes `/api/v1`, sans les supprimer avant migration ;
- le raccordement FastAPI exclusif à `DummyBackend` par une orchestration via
  les services et le catalogue, avec injection de doubles en test ;
- les langues codées en dur dans les routes et composants par un registre
  central de capacités ;
- l'inférence bloquante dans une route `async` par un job borné ou une
  exécution hors event loop ;
- les identifiants dérivés du hash audio par des identifiants publics UUID
  non prédictibles ;
- la validation du seul `Content-Type` déclaré par une validation combinant
  taille, extension, signature MIME et décodage.

### What should not be touched

- `checkpoint-002052`, ses poids, son hash et ses paramètres ;
- `src/ivoirevoice/training/`, les configurations d'expériences et les états
  de reprise ;
- le protocole et le reçu terminal de l'unique final-holdout ;
- les métriques finales publiées ;
- le corpus, les manifests complets, les prédictions individuelles et les
  artefacts privés ;
- les révisions Whisper épinglées et la politique sans token `dyu` forcé ;
- l'application Gradio actuelle tant que la nouvelle surface n'a pas atteint
  la parité explicitement approuvée.

Les dépendances Python existantes restent contrôlées par
`scripts/check_harness.py`. Tout nouveau domaine Python exige une décision
documentée et une mise à jour simultanée de `docs/architecture.md`, du harness
et de `docs/knowledge-map.yaml`.

## 3. Architecture frontend

Le frontend cible réside dans `web/` et utilise Next.js, React, TypeScript en
mode strict et Tailwind CSS. Des primitives accessibles peuvent être inspirées
de shadcn/ui, sans rendre le produit dépendant d'un style tiers ni copier une
identité existante.

Structure proposée :

```text
web/
  src/
   app/
    page.tsx
    transcribe/page.tsx
    translate/page.tsx
    learn/page.tsx
    learn/courses/page.tsx
    learn/courses/[courseId]/page.tsx
    learn/lessons/[lessonId]/page.tsx
    practice/page.tsx
    community/page.tsx
    profile/page.tsx
    about/page.tsx
   components/
    layout/
    ui/
    transcription/
    learning/
   lib/
    api/
    languages/
   i18n/
    messages.ts
    provider.tsx
```

Les composants de fondation sont `Navbar`, `Footer`, `Hero`, `FeatureCard`,
`LanguageBadge`, `CourseCard`, `ProgressCard`, `AudioUploader`,
`LanguageSelector`, `PrimaryButton`, `SecondaryButton`, `EmptyState` et
`StatusBadge`. Ils partagent des tokens de couleur, typographie, espacement,
focus et mouvement. La Foundation rend son shell et ses pages comme composants
clients pour permettre le changement de locale FR/EN sans route localisée. Ce
choix reste cantonné à la présentation ; les futurs chargements de données
privilégieront des composants serveur et des îlots clients ciblés.

`lib/api` expose un client typé et configurable. Les composants ne construisent
pas d'URL FastAPI et ne connaissent pas les variables de checkpoint. Une route
Next.js côté serveur, limitée à une allowlist de contrats, conserve une seule
origine et ne transmet ni cookie ni autorisation du navigateur. Toute future
connexion directe à FastAPI exigera une allowlist CORS explicite.

Le Foundation MVP fournit un shell et des états honnêtes. Il ne contient ni
traduction fictive, ni cours Dioula présenté comme validé, ni scoring de
prononciation simulé.

## 4. Architecture backend

FastAPI reste l'adaptateur HTTP du package Python. La cible introduit des
routeurs versionnés et injecte les services :

```text
api/routes -> services -> ports de domaine -> adaptateurs ML/DB/stockage
```

Les routeurs traduisent HTTP vers des commandes applicatives. Ils ne chargent
pas un modèle, n'ouvrent pas directement un fichier de corpus et ne calculent
pas une métrique. Les services ne dépendent pas de FastAPI. Les adaptateurs de
base, stockage et providers implémentent des protocoles injectables afin de
rester remplaçables et testables.

L'ASR GPU est un composant à concurrence bornée. Pour le premier déploiement
local, une file en mémoire et un seul worker d'inférence suffisent. Un
démarrage multi-processus ne doit pas créer un exemplaire du modèle par worker
sans budget VRAM explicite. Une file distribuée éventuelle est un choix
d'exploitation futur, pas une dépendance du Foundation MVP.

Les anciens endpoints peuvent rester disponibles pendant une période de
compatibilité, avec documentation de dépréciation. Ils ne deviennent pas la
base du client Next.js.

## 5. Modèle de données

PostgreSQL est la cible persistante. SQLAlchemy avec migrations Alembic est une
option cohérente pour le backend Python, à confirmer dans un lot dédié ; aucune
base n'est requise pour afficher le Foundation statique.

Modèle minimal :

| Entité | Responsabilité et champs essentiels |
|---|---|
| `User` | UUID, email normalisé, nom d'affichage, locale, état, timestamps |
| `Language` | code canonique, noms localisés, statut ASR/traduction/apprentissage |
| `Course` | langue étudiée, titre localisé, niveau interne, statut de validation |
| `Module` | cours, ordre, titre, statut |
| `Lesson` | module, ordre, objectif et contenu révisable |
| `Exercise` | leçon, type, consigne, réponse attendue, statut de validation |
| `ExerciseChoice` | exercice, ordre, libellé, correction |
| `ExerciseAttempt` | utilisateur, exercice, réponse, résultat, date |
| `CourseEnrollment` | utilisateur, cours, état, inscription |
| `Progress` | inscription, dernière leçon, compteurs agrégés, mise à jour |
| `Transcription` | UUID, utilisateur facultatif, langue, modèle, état, consentement, expiration |
| `Translation` | UUID, langues source/cible, provider, état, consentement, expiration |
| `Discussion` | auteur, catégorie, titre, contenu, état de modération |
| `Comment` | discussion, auteur, parent facultatif, contenu, état de modération |

Les relations imposent des clés étrangères et des contraintes d'unicité sur
les ordres pédagogiques. Les suppressions de comptes et contenus sont
explicites. Les niveaux A1 à B2 restent des repères internes tant qu'aucune
correspondance CECR Dioula n'est scientifiquement établie.

Audio, transcription et traduction sont éphémères par défaut. Une conservation
requiert un consentement explicite et une date d'expiration. Le chemin de
stockage n'apparaît jamais dans une ressource publique.

## 6. API

Socle public proposé :

| Méthode et route | Rôle |
|---|---|
| `GET /api/health` | vivacité du service, sans charger de modèle |
| `GET /api/v1/models` | modèles publics, langues et états agrégés |
| `GET /api/v1/languages` | registre central et capacités par langue |
| `POST /api/v1/transcriptions` | valider un upload et créer une transcription |
| `GET /api/v1/transcriptions/{id}` | lire l'état et le résultat autorisé |
| `DELETE /api/v1/transcriptions/{id}` | supprimer résultat et audio associés |
| `POST /api/v1/translations` | demander une traduction à un provider disponible |
| `GET /api/v1/translations/{id}` | lire l'état et le résultat autorisé |
| `GET /api/v1/courses` | lister les cours publiés et validés |
| `GET /api/v1/courses/{id}` | lire la structure publique d'un cours |
| `GET /api/v1/lessons/{id}` | lire une leçon autorisée |
| `POST /api/v1/exercise-attempts` | enregistrer et corriger une tentative |
| `GET /api/v1/me/progress` | progression de l'utilisateur authentifié |

`POST /transcriptions` accepte `audio`, `language` et `model`. Une réponse
`202` fournit un UUID et un état `queued`, `processing`, `succeeded`, `failed`
ou `expired`. Une optimisation synchrone locale peut retourner immédiatement
un résultat, mais conserve le même schéma de ressource.

Les réponses d'erreur utilisent une enveloppe stable avec `code`, `message`,
`details` assainis et `request_id`. Une paire de traduction sans provider
retourne une erreur de capacité explicite ; elle ne fabrique jamais un texte.
Les endpoints modèles/langues n'exposent ni nom de classe interne, ni chemin,
ni contenu privé.

## 7. Services IA

### ASR

Le service ASR réutilise `TranscriptionService`, `ModelRegistry` et
`WhisperBackend`. Capacités actuelles :

| Langue | Statut | Preuve et limite |
|---|---|---|
| Français (`fr`) | `experimental` | baselines configurées, sans évaluation française finale publiée |
| English (`en`) | `coming_soon` | aucun backend anglais raccordé dans le catalogue actuel |
| Dioula (`dyu`) | `experimental` | modèle final local gelé ; résultats limités au corpus observé |

`auto` est une option d'expérience utilisateur, pas une langue. Elle reste
`coming_soon` jusqu'à ce qu'une politique de détection et de confiance soit
testée. Le modèle Dioula final accepte uniquement `dyu` et reste fourni par
`IVOIREVOICE_DIOULA_FINAL_MODEL_PATH`.

### Traduction

Un port indépendant est prévu :

```python
class TranslationProvider(Protocol):
    def translate(
        self,
        text: str,
        source_language: str,
        target_language: str,
    ) -> TranslationResult: ...
```

Il pourra être implémenté par NMT, modèle multilingue, LLM, API, dictionnaire
spécialisé ou moteur hybride. Aucun provider n'est actuellement validé :
FR↔DYU, EN↔DYU et FR↔EN restent `coming_soon` et non implémentés. Une simple
copie du texte ou un dictionnaire fictif ne constitue pas une traduction.

La synthèse vocale et le scoring phonétique suivent des ports séparés. Ils
restent `coming_soon`, notamment pour le Dioula, tant qu'une méthode fiable et
évaluée n'existe pas.

## 8. Internationalisation et registre des langues

Les locales d'interface initiales sont `fr` et `en`. Les messages sont
externalisés dans des catalogues et jamais dispersés dans les composants. Une
locale `dyu` pourra être ajoutée après validation linguistique.

Le code canonique interne du Dioula est `dyu`, cohérent avec le dépôt et
l'ISO 639-3. `dy` ne doit pas devenir un deuxième identifiant interne. Les
libellés affichés sont localisés séparément du code.

Le registre central expose au minimum :

```yaml
languages:
  fr:
    display_names: {fr: Français, en: French}
    interface: available
    asr: experimental
    translation: coming_soon
  en:
    display_names: {fr: Anglais, en: English}
    interface: available
    asr: coming_soon
    translation: coming_soon
  dyu:
    display_names: {fr: Dioula, en: Dioula}
    interface: coming_soon
    asr: experimental
    translation: coming_soon
```

Baoulé, Bété, Sénoufo et Agni sont enregistrables sans condition dispersée,
mais restent `coming_soon`. À terme, `GET /api/v1/languages` est la source de
vérité runtime ; une copie TypeScript de fondation doit être couverte par un
test de contrat pour éviter la dérive.

## 9. Sécurité

- limiter taille et durée avant inférence ;
- vérifier extension, type MIME déclaré, signature de fichier et décodage ;
- refuser les archives, chemins et noms de fichiers utilisés comme chemins
  serveur ;
- générer des UUID non prédictibles et appliquer une autorisation par
  ressource ;
- exécuter le traitement audio avec privilèges minimaux et répertoires
  temporaires dédiés ;
- borner file, concurrence, durée et mémoire des traitements ;
- ne jamais retourner stack trace, exception provider, chemin local ou secret ;
- journaliser `request_id`, état, latence et code d'erreur, sans audio ni texte ;
- appliquer une allowlist CORS ou un proxy same-origin ;
- conserver FastAPI, le modèle et la base sur un réseau non public par défaut ;
- préparer rate limiting, protection CSRF selon le mode d'authentification et
  headers de sécurité avant exposition Internet.

Email/mot de passe peut être ajouté après définition du stockage sécurisé des
mots de passe, des sessions, de la vérification d'email et de la récupération.
Google, Microsoft et OTP restent hors du Foundation MVP.

## 10. Privacy et gouvernance

Un audio utilisateur n'est jamais présumé réutilisable. Avant traitement,
l'interface explique la finalité, la rétention et la suppression. Par défaut :

1. upload vers un espace temporaire isolé ;
2. transcription ou échec ;
3. restitution à l'utilisateur ;
4. suppression automatique de l'audio et expiration du résultat.

Toute conservation pour historique ou amélioration future nécessite un choix
distinct, révocable et documenté. Ce consentement ne vaut jamais consentement
à l'entraînement. Les journaux excluent transcription, traduction, référence,
nom original, adresse de stockage et métadonnée participant.

Les corpus train/validation/test, le final holdout, les prédictions privées,
les checkpoints et les audios internes restent hors Git et hors dépendances de
la plateforme. Le final holdout ne peut être lu par aucune route, tâche ou
service Phase 2. Les rapports publics restent agrégés.

La licence du code et le consentement de redistribution du corpus/checkpoint
restent des décisions humaines bloquantes avant diffusion plus large. Les
politiques de rétention, export et suppression de compte doivent être validées
avant une ouverture à des utilisateurs réels.

## 11. Architecture pédagogique

La hiérarchie est :

```text
Course
  `-- Module
        |-- Lesson
        |     |-- Content
        |     |-- Vocabulary
        |     |-- Audio
        |     |-- Example
        |     `-- Exercise
        `-- Module quiz
```

Les exercices prévus sont QCM, texte à compléter, association,
compréhension orale, transcription, traduction dans les deux sens et pratique
de prononciation. Chaque type possède un contrat de réponse et de correction.
La correction présente état, bonne réponse, explication et exemple ; elle ne
se limite pas à « faux ».

La progression agrège cours, modules, leçons, tentatives et dernière activité
sans transformer l'espace apprenant en tableau analytique complexe. Les
recommandations sont déterministes et explicables au MVP ; aucune IA de
recommandation n'est prétendue.

La publication d'un contenu suit les états `draft`, `linguist_review`,
`approved`, `published`, `archived`. Seul `published` apparaît comme cours
fiable. Les scaffolds sont visiblement marqués `DEMO` ou `À VALIDER PAR UN
EXPERT LINGUISTIQUE`.

## 12. Stratégie de migration

1. Conserver `make ui` et `make demo` comme Legacy Demo UI.
2. Ajouter le frontend sous `web/`, sans modifier le moteur ASR.
3. Introduire le registre de langues et le client API abstrait avec doubles de
   données uniquement. **Foundation terminée.**
4. Ajouter les routeurs `/api/v1` à côté des routes FastAPI historiques.
5. Raccorder d'abord l'état de santé, les langues et les modèles.
6. Raccorder une transcription utilisateur au service ASR existant, avec
   upload temporaire, concurrence bornée et tests synthétiques.
7. Ajouter le port de traduction sans provider et afficher `coming_soon`.
8. Introduire PostgreSQL et le stockage seulement avec politiques de migration,
   sauvegarde, rétention et suppression.
9. Ajouter les domaines pédagogiques avec contenu placeholder validé comme tel.
10. Déprécier Gradio uniquement après parité, tests, revue humaine et décision
    explicite ; ne pas le supprimer dans cette phase.

Chaque lot conserve `make verify`. Les gates frontend rejoignent le Makefile et
la CI : installation verrouillée, lint, typecheck, tests et build. Le harness
et la documentation sont mis à jour dans le même changement qu'une nouvelle
frontière.

## 13. Déploiement

Le premier environnement reste local et privé :

```text
Next.js :3000
   |
FastAPI :8000
   |
worker ASR unique sur GPU ---- checkpoint monté hors image et hors Git
   |
PostgreSQL + stockage temporaire local
```

Next.js et FastAPI ont des health checks distincts. La disponibilité d'un
modèle est une readiness dégradée et ne doit pas transformer la liveness en
chargement coûteux. Le checkpoint est monté en lecture seule et fourni par
variable d'environnement ; il n'est jamais copié dans une image, un bundle web
ou un volume public.

Une évolution vers S3/MinIO utilise le même port `ObjectStorage`, avec chiffrement,
expiration et suppression vérifiable. PostgreSQL reçoit migrations, sauvegardes
et compte applicatif à privilèges minimaux. Les secrets viennent du gestionnaire
du déploiement, jamais d'un fichier versionné.

Un déploiement Internet n'est pas autorisé par cette architecture seule. Il
exige au minimum authentification, autorisation, rate limiting, politiques de
consentement/rétention, observabilité assainie, revue de licence, tests de
charge et revue de sécurité.

## 14. Roadmap

1. **Foundation / design system — terminé** : tokens, composants accessibles,
   registre TypeScript et client API abstrait.
2. **Frontend shell / navigation — terminé** : layout partagé, routes,
   responsive, catalogues `fr` et `en`.
3. **Transcription UI** : upload/microphone, états, résultat et actions ; aucun
   raccordement ML direct au navigateur.
4. **Translation abstraction + UI** : deux panneaux et contrat provider ; toutes
   les paires restent `coming_soon` sans provider validé.
5. **Learning domain** : modèle Course/Module/Lesson et workflow de validation.
6. **Courses / lessons** : navigation et rendu de contenus explicitement DEMO.
7. **Exercises** : types, correction pédagogique et tests d'accessibilité.
8. **Progress tracking** : inscriptions, tentatives et vues synthétiques.
9. **Authentication** : comptes locaux et sessions après revue de sécurité.
10. **Community** : discussions/commentaires minimaux avec modération.
11. **Quality and deployment** : contrats API, tests E2E, accessibilité,
    sécurité, rétention, observabilité et préparation d'exploitation.

Le prochain lot recommandé est le raccordement progressif des contrats publics
FastAPI, en commençant par health/langues/modèles puis une transcription
protégée. Une traduction, une base persistante, l'authentification et tout
contenu pédagogique définitif nécessitent toujours des lots et validations
séparés.
