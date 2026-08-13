# IvoireVoice — contrat produit de la plateforme linguistique

## Statut et portée

Ce document définit le produit cible de la Phase 2 et le périmètre de sa
première fondation web. Il complète, sans le remplacer, le
[contrat du démonstrateur ASR local](mvp.md). Le démonstrateur Gradio reste la
seule interface reliée au modèle Dioula final tant que l'intégration FastAPI de
production n'est pas terminée.

La Phase 2 fait évoluer IvoireVoice vers une plateforme multilingue réunissant
transcription, traduction et apprentissage. Elle ne modifie ni le modèle gelé
`checkpoint-002052`, ni les données, ni les résultats expérimentaux. Elle ne
constitue pas une autorisation de déployer publiquement le checkpoint ou le
corpus privés.

Les termes suivants ne sont pas interchangeables :

- **ASR / transcription** : convertir un audio en texte dans la même langue ;
- **traduction** : convertir un texte source en texte dans une autre langue ;
- **transcription puis traduction** : deux opérations explicites et
  indépendantes, avec un résultat intermédiaire visible et modifiable ;
- **contenu pédagogique validé** : contenu relu par un locuteur ou un expert
  linguistique identifié dans le processus éditorial.

## Promesse produit

IvoireVoice aide une personne à écouter, transcrire et apprendre des langues
africaines dans une expérience simple, chaleureuse et accessible. Le produit
principal reste **IvoireVoice**. Les libellés Transcrire, Traduire, Apprendre et
Communauté décrivent des espaces fonctionnels, pas des marques indépendantes.

Principes produit :

- une action principale claire par écran ;
- un vocabulaire compréhensible par une personne non technique ;
- une conception mobile utilisable avec une connexion intermittente ;
- aucune fonction annoncée comme disponible avant un fonctionnement bout en
  bout ;
- aucun contenu dioula présenté comme fiable avant validation linguistique ;
- aucune donnée audio réutilisée ou conservée sans choix explicite ;
- aucune métrique scientifique extrapolée au-delà du protocole qui l'a produite.

## Statuts de capacité

Chaque capacité exposée par l'interface possède l'un des statuts suivants :

- `available` : parcours bout en bout utilisable dans le périmètre annoncé ;
- `experimental` : parcours utilisable, mais avec une qualité, une couverture
  ou une validation encore limitée ;
- `coming_soon` : parcours non exécutable ; l'interface peut l'expliquer, mais
  ne doit produire aucun faux résultat.

Le statut est attaché à une **capacité**, pas seulement à une langue. Une langue
peut par exemple avoir un ASR expérimental et aucune traduction disponible.

### Matrice honnête des capacités

| Capacité | Preuve actuelle | Statut dans la nouvelle plateforme avant intégration | Présentation autorisée |
|---|---|---|---|
| ASR dioula (`dyu`) | Modèle final gelé utilisable localement dans Gradio | `coming_soon` | « Transcription dioula en cours d'intégration » ; le modèle reste qualifié d'expérimental |
| ASR français (`fr`) | Baselines acceptant `fr`, sans évaluation française achevée | `coming_soon` | Ne pas promettre une qualité validée |
| ASR anglais (`en`) | Non configuré dans l'application actuelle | `coming_soon` | Ne pas afficher de résultat simulé |
| Traduction français → dioula | Aucun fournisseur implémenté ou validé | `coming_soon` | Architecture seulement |
| Traduction dioula → français | Aucun fournisseur implémenté ou validé | `coming_soon` | Architecture seulement |
| Traduction anglais → dioula | Aucun fournisseur implémenté ou validé | `coming_soon` | Architecture seulement |
| Traduction dioula → anglais | Aucun fournisseur implémenté ou validé | `coming_soon` | Architecture seulement |
| Traduction français ↔ anglais | Aucun fournisseur intégré au projet | `coming_soon` | Ne pas déduire une disponibilité de capacités externes |
| Cours et leçons | Aucun contenu expertisé livré | `coming_soon` | Cartes `DEMO — À VALIDER PAR UN EXPERT LINGUISTIQUE` |
| Exercices et progression | Architecture à construire | `coming_soon` | Démonstration d'interface sans score persistant |
| Prononciation | Aucun scoring phonétique validé | `coming_soon` | Ne jamais afficher une note scientifique fictive |
| Communauté | Aucun compte, modération ou stockage | `coming_soon` | Page informative uniquement |

Une capacité passe à `experimental` ou `available` uniquement avec le service,
les tests et les garde-fous correspondants. Une simple page ou un composant ne
suffit pas.

## Registre des langues

Le produit sépare trois notions :

1. la langue de l'interface ;
2. la langue de l'audio ou du texte ;
3. la langue étudiée.

Les locales d'interface initiales sont `fr` et `en`. Le dioula pourra devenir
une locale d'interface après traduction et validation des messages. Le code
canonique du dioula dans les contrats techniques reste `dyu` ; le produit ne
doit pas créer un second identifiant `dy`.

Le français, l'anglais et le dioula sont déclarés dans un registre central.
Les futures langues africaines sont ajoutées par configuration et validation,
sans conditions dispersées dans les composants. Aucun code ou nom localisé de
langue future ne doit être inventé avant vérification.

## Personas

Les noms ci-dessous sont fictifs et ne correspondent à aucune personne du
corpus.

### Awa — apprenante débutante

- utilise principalement un smartphone ;
- comprend le français et souhaite apprendre le dioula pour la vie quotidienne ;
- veut reprendre une leçon sans devoir comprendre le fonctionnement de l'IA ;
- peut disposer d'une connexion lente ou intermittente ;
- attend des corrections encourageantes et explicatives.

### Karim — utilisateur de transcription

- possède un enregistrement qu'il est autorisé à traiter ;
- veut obtenir rapidement un texte copiable ou téléchargeable ;
- souhaite ensuite traduire ce texte sans perdre la transcription originale ;
- doit comprendre lorsqu'un modèle ou une langue n'est pas disponible ;
- ne souhaite pas que son audio soit conservé ou utilisé pour entraîner un
  modèle par défaut.

### Mariam — enseignante ou experte linguistique

- révise vocabulaire, orthographe, tons, traductions et exemples ;
- distingue un brouillon, un contenu relu et un contenu publié ;
- souhaite identifier l'origine et la version d'une leçon ;
- ne veut pas qu'un contenu généré automatiquement soit présenté comme validé.

### Yao — membre de la communauté

- pose une question de vocabulaire ou d'usage culturel ;
- veut distinguer les réponses de la communauté des contenus éditoriaux ;
- doit pouvoir signaler un contenu inapproprié ;
- ne doit pas voir d'information privée provenant des données ML.

### Besoins inclusifs transversaux

Le produit doit aussi fonctionner pour une personne naviguant au clavier,
utilisant un lecteur d'écran, zoomant le texte, ayant une perception limitée
des couleurs ou ne pouvant pas utiliser le microphone. Les parcours principaux
ne dépendent donc ni d'une couleur seule, ni d'un geste précis, ni d'un canal
audio unique.

## Parcours utilisateurs

### Transcrire un audio

1. L'utilisateur ouvre **Transcrire**.
2. Il choisit un fichier autorisé ou lance un enregistrement au microphone.
3. Il lit l'information de traitement et confirme l'envoi.
4. Il choisit la langue disponible ou le mode automatique lorsque celui-ci est
   réellement pris en charge.
5. Le produit valide le fichier avant traitement.
6. Un état explicite annonce l'attente puis le traitement.
7. Le texte apparaît avec la langue, la durée et le temps de traitement
   disponibles.
8. L'utilisateur copie ou télécharge son résultat.
9. Il peut demander une traduction uniquement si la paire est disponible.
10. L'audio temporaire est supprimé selon la politique annoncée.

### Transcrire puis traduire

1. La transcription est terminée et reste visible.
2. L'utilisateur sélectionne **Traduire cette transcription**.
3. Le texte source est copié dans Traduire ; il peut être corrigé avant envoi.
4. La langue source et la langue cible sont confirmées séparément.
5. Si la paire est `coming_soon`, le produit l'indique sans générer de texte.
6. Si un fournisseur est disponible, la traduction est affichée séparément de
   la transcription et conserve sa propre provenance et son propre statut.

### Commencer à apprendre

1. L'utilisateur ouvre **Apprendre**.
2. Il choisit un cours adapté à son objectif.
3. Une fiche décrit les objectifs, le niveau interne et le statut éditorial.
4. Il ouvre une leçon, lit ou écoute un exemple, puis réalise un exercice bref.
5. Il reçoit « Correct » ou « À revoir », la réponse attendue et une explication.
6. Avec un compte et lorsque la progression sera disponible, il peut marquer la
   leçon terminée et reprendre plus tard.

Les niveaux A1, A2, B1 et B2 peuvent servir d'échelle interne descriptive. Ils
ne constituent pas une certification CECR officielle du dioula.

### Pratiquer

1. L'utilisateur choisit un type d'exercice disponible.
2. Les consignes précisent la langue, l'objectif et le nombre de questions.
3. Chaque réponse reçoit une correction textuelle, jamais un simple code couleur.
4. Un exercice d'écoute fournit une alternative accessible après la tentative
   ou via un mode d'accommodation.
5. Un exercice de prononciation ne produit aucune note phonétique tant qu'un
   protocole scientifique n'est pas validé.

### Consulter sa progression

1. Un utilisateur authentifié voit ses cours en cours et sa dernière activité.
2. La progression globale reste synthétique ; elle n'est pas un tableau
   analytique complexe.
3. Les scores indiquent leur périmètre et ne sont pas comparés à d'autres
   utilisateurs par défaut.
4. L'utilisateur peut supprimer son historique selon la politique applicable.

### Participer à la communauté

Le MVP Foundation présente seulement l'intention et le statut `coming_soon`.
Les discussions ne deviennent disponibles qu'après authentification,
modération, signalement, règles de conduite et politique de conservation.

## Architecture pédagogique fonctionnelle

```text
Course
└── Module
    ├── Lesson
    │   ├── Content
    │   ├── Vocabulary
    │   ├── Audio
    │   ├── Example
    │   └── Exercise
    └── Module quiz
```

Types d'exercices prévus :

1. QCM ;
2. texte à compléter ;
3. association mot–traduction ;
4. écoute puis choix ;
5. écoute puis transcription ;
6. français vers dioula ;
7. dioula vers français ;
8. pratique de prononciation, sans scoring non validé.

Tout exemple linguistique de la Foundation porte visiblement la mention :

> **DEMO — À VALIDER PAR UN EXPERT LINGUISTIQUE**

Les contenus de démonstration ne doivent pas être repris du corpus privé.

## Pages et exigences fonctionnelles

### Navigation globale

La navigation principale contient Accueil, Transcrire, Traduire, Apprendre,
S'exercer, Communauté et Mon espace. À propos reste accessible depuis le pied
de page et peut apparaître dans le menu mobile. Le lien actif est identifiable
autrement que par la couleur.

### Accueil

- promesse produit et deux actions principales ;
- trois cartes Transcrire, Traduire et Apprendre avec leur statut réel ;
- langues actuellement prévues ;
- aperçu de cours de démonstration ;
- explication simple de la technologie et des limites ;
- aucune statistique ou capacité non prouvée.

### Transcrire

- upload et microphone comme alternatives équivalentes ;
- langue audio distincte de la locale d'interface ;
- validation du format, de la taille et de la durée ;
- résultat large et lisible ;
- langue détectée, durée et temps seulement lorsqu'ils sont disponibles ;
- copier, TXT et JSON ;
- action Traduire conditionnée par la matrice de capacités ;
- information claire sur la conservation de l'audio.

### Traduire

- deux panneaux source et cible ;
- saisie manuelle ou reprise explicite d'une transcription ;
- sélection des langues et inversion seulement pour une paire autorisée ;
- copier, effacer et écouter uniquement si la synthèse vocale existe ;
- état `coming_soon` sans traduction factice ;
- provenance du fournisseur et avertissement expérimental lorsque pertinent.

### Apprendre, cours et leçon

- reprise, cours recommandés et niveau interne ;
- catalogue filtrable sans surcharge ;
- cours → modules → leçons ;
- objectif, contenu, exemples, traductions validées, audio et vocabulaire ;
- navigation précédente/suivante ;
- progression persistante uniquement après authentification.

### S'exercer

- choix du type d'exercice ;
- consigne, progression de session et correction pédagogique ;
- aucun classement public dans le MVP ;
- aucune note de prononciation non validée.

### Communauté

- présentation des futures catégories : apprentissage, prononciation,
  vocabulaire, culture et traduction ;
- statut `coming_soon` dans la Foundation ;
- aucune fausse discussion ou identité utilisateur.

### Profil et progression

- état visiteur expliquant l'utilité future d'un compte ;
- langue d'interface et langue étudiée ;
- niveau, progression, cours et activité seulement lorsque les données existent ;
- préférences de confidentialité et suppression accessibles ;
- aucune authentification simulée.

### À propos

- mission et périmètre de recherche ;
- différence entre produit cible et capacités disponibles ;
- résultats ASR finaux agrégés sans nouvelle évaluation ;
- limites, gouvernance des données et statut du checkpoint ;
- contact et documentation, sans chemin local.

## Wireframes textuels

### `/` — Accueil

```text
[Lien d'évitement]
[Navbar : logo | navigation | locale | Mon espace]
[Hero : promesse | Transcrire un audio | Apprendre le Dioula]
[3 cartes : Transcrire | Traduire (statut) | Apprendre (statut)]
[Langues : Français | English | Dioula | futures langues]
[Cours DEMO : cartes À VALIDER]
[Technologie et confidentialité en langage simple]
[Footer : À propos | Confidentialité | Accessibilité]
```

### `/transcribe` — Transcrire

```text
[Header : titre | explication de confidentialité]
[Choix Importer un audio OU Enregistrer]
[Zone de dépôt / contrôle microphone]
[Langue de l'audio | statut des langues]
[Bouton Transcrire]
[État : vide / validation / traitement / erreur]
[Résultat : texte | langue | durée | temps]
[Copier | Télécharger TXT | Télécharger JSON | Traduire]
[Information de suppression de l'audio]
```

### `/translate` — Traduire

```text
[Header : titre | statut expérimental/coming soon]
[Panneau source : langue | texte | utiliser une transcription]
[Inverser les langues si paire autorisée]
[Panneau cible : langue | résultat]
[Traduire]
[Copier | Écouter si disponible | Effacer]
[Provenance, limites et message d'erreur]
```

### `/learn` — Tableau d'apprentissage

```text
[Header : Bonjour / invitation à se connecter]
[Continuer votre apprentissage]
[Progression globale ou état visiteur]
[Cours en cours]
[Cours recommandés DEMO — À VALIDER]
[Niveau interne et explication]
```

### `/learn/courses` — Catalogue

```text
[Titre et objectif]
[Filtres : niveau | thème | langue]
[Grille de CourseCard]
[Chaque carte : titre | niveau interne | progression | statut éditorial]
[État vide / chargement / erreur]
```

### `/learn/courses/[courseId]` — Fiche cours

```text
[Fil d'Ariane]
[Titre | description | niveau | statut éditorial]
[Objectifs]
[Progression ou invitation à se connecter]
[Liste des modules]
  [Module]
    [Leçons | durées | états]
[Commencer / Continuer]
```

### `/learn/lessons/[lessonId]` — Leçon

```text
[Fil d'Ariane et progression du module]
[Titre | objectif pédagogique]
[Contenu validé ou bannière DEMO]
[Exemple dioula | traduction française | anglais si validé]
[Lecteur audio accessible]
[Vocabulaire]
[Exercice rapide]
[Leçon précédente | Marquer comme terminée | Leçon suivante]
```

### `/practice` — S'exercer

```text
[Titre | objectif]
[Types d'exercices disponibles et statuts]
[Zone d'exercice]
[Consigne | question | contrôle audio si nécessaire]
[Valider]
[Feedback : Correct / À revoir | réponse | explication | exemple]
[Progression de la session]
```

### `/community` — Communauté

```text
[Titre]
[Bannière Coming soon]
[Description des règles et de la modération prévue]
[Aperçu non interactif des catégories]
[Aucune discussion fictive]
```

### `/profile` — Mon espace

```text
[État visiteur : se connecter, sans faux compte]
OU
[Avatar | nom choisi | préférences]
[Langue d'interface | langue étudiée | niveau interne]
[Progression | cours | activité récente]
[Confidentialité | exporter/supprimer mes données]
```

### `/about` — À propos

```text
[Mission]
[Ce qui est disponible / expérimental / à venir]
[Architecture simplifiée]
[Résultats ASR agrégés et limites]
[Confidentialité et gouvernance]
[Perspectives pour d'autres langues africaines]
```

## États et erreurs

Toutes les pages asynchrones prévoient les états suivants :

- initial et vide ;
- validation locale ;
- en attente ;
- traitement en cours avec libellé textuel ;
- succès ;
- succès partiel, par exemple transcription réussie mais traduction indisponible ;
- erreur récupérable avec action proposée ;
- indisponible ou `coming_soon` ;
- hors ligne ;
- accès refusé ou session expirée lorsque l'authentification existera.

Erreurs à traiter explicitement :

- fichier vide, trop volumineux, trop long, corrompu ou de type non autorisé ;
- permission microphone refusée ou enregistrement interrompu ;
- langue ou paire de traduction non prise en charge ;
- modèle non configuré, occupé ou en échec ;
- réseau interrompu, délai dépassé ou réponse API invalide ;
- cours, leçon ou exercice introuvable ;
- sauvegarde de progression impossible ;
- contenu communautaire supprimé ou non autorisé.

Un message utilisateur ne contient ni stack trace, ni nom de classe, ni chemin,
ni texte privé journalisé. Il explique ce qui s'est passé, ce qui a été
préservé et l'action suivante. Un échec d'un service ne doit pas effacer un
résultat déjà obtenu par un autre service.

## Responsive design

La conception commence par le contenu et doit fonctionner au minimum à 320 px
de large et à un zoom de 200 %.

### Mobile

- navigation condensée avec ordre de tabulation cohérent ;
- une seule colonne ;
- upload et microphone immédiatement accessibles ;
- panneaux de traduction empilés, source avant cible ;
- boutons principaux pleine largeur si nécessaire ;
- cartes sans défilement horizontal ;
- actions secondaires regroupées sans masquer leur libellé.

### Tablette

- grille de deux colonnes lorsque l'espace le permet ;
- navigation compacte ;
- traduction côte à côte seulement si chaque panneau reste lisible.

### Desktop

- largeur de lecture limitée ;
- espaces généreux ;
- deux panneaux de traduction et grilles de cours ;
- aucune information essentielle dépend d'un survol.

## Accessibilité — objectif WCAG 2.2 AA

La conformité ne repose pas sur une vérification visuelle seule. Les critères
d'acceptation incluent :

- HTML sémantique, landmarks et lien d'évitement ;
- hiérarchie de titres sans saut arbitraire ;
- navigation et activation intégrales au clavier ;
- focus visible avec contraste suffisant ;
- nom accessible pour chaque champ, bouton et contrôle audio ;
- erreurs reliées à leur champ et résumé d'erreurs focalisable ;
- annonces `aria-live` pour enregistrement, upload et traitement ;
- information jamais portée uniquement par la couleur, une icône ou le son ;
- contraste AA pour le texte et 3:1 pour les composants essentiels ;
- cibles tactiles d'au moins 44 × 44 px dans le design IvoireVoice ;
- prise en charge de `prefers-reduced-motion` ;
- textes redimensionnables sans perte de contenu ;
- attribut `lang` correct pour l'interface et les passages multilingues ;
- alternatives aux actions microphone et aux contenus audio ;
- tests automatisés axe, tests clavier et revue manuelle avec lecteur d'écran.

Les scores automatiques sont des signaux, pas une preuve suffisante de
conformité.

## Internationalisation

- aucun texte d'interface réutilisable n'est codé directement dans un composant ;
- les catalogues `fr` et `en` partagent les mêmes clés typées ;
- les nombres, dates, durées et pourcentages utilisent les API `Intl` ;
- changer la locale ne modifie pas la langue source d'un audio ou d'un cours ;
- les messages API reposent sur des codes stables, puis sont localisés dans le
  frontend ;
- les noms de langues sont localisés depuis le registre central ;
- les contenus de cours disposent de versions éditoriales distinctes et ne
  sont pas traduits automatiquement par le catalogue d'interface ;
- le CSS utilise des propriétés logiques afin de ne pas bloquer une future
  écriture de droite à gauche, sans prétendre qu'elle est déjà supportée.

## Confidentialité, consentement et rétention

### Audio et texte utilisateur

- l'utilisateur doit être autorisé à traiter l'audio qu'il fournit ;
- le traitement nécessaire à la requête est expliqué avant l'envoi ;
- l'audio est éphémère par défaut dans le MVP et supprimé après traitement ;
- aucune conservation facultative n'est pré-cochée ;
- le consentement au traitement, la sauvegarde dans un compte et une éventuelle
  contribution à l'amélioration ML sont trois choix séparés ;
- aucun audio, texte ou feedback utilisateur n'alimente l'entraînement par
  défaut ;
- une politique de rétention chiffrée doit être décidée avant toute persistance
  de production ; la Foundation ne doit pas inventer de durée ;
- l'utilisateur authentifié doit pouvoir consulter et supprimer ses données
  persistées lorsque cette capacité sera introduite.

### Journaux et observabilité

Les événements peuvent contenir un identifiant de requête non prédictible, le
statut, des tailles agrégées, la durée, la latence et une clé publique de
capacité. Ils ne contiennent pas :

- audio ou transcription ;
- texte à traduire ou résultat traduit ;
- nom de fichier original ;
- chemin local ;
- checkpoint ;
- secret ou jeton ;
- contenu de cours privé ;
- message communautaire complet dans les traces techniques.

### Données ML existantes

La plateforme ne dépend jamais directement des dossiers train, validation,
pilote historique ou final holdout. Le checkpoint privé est configuré
uniquement côté serveur. Les rapports publics restent agrégés. Aucun parcours
Phase 2 ne peut relancer un entraînement, un refit ou une évaluation du holdout.

### Déploiement et communauté

Le contrat de sécurité actuel couvre un opérateur local de confiance. Une
exposition publique exige au préalable authentification, autorisation, CORS
restrictif, contrôle CSRF lorsque pertinent, rate limiting, validation réelle
des fichiers, stockage sécurisé, politique de suppression, modération,
signalement et réponse aux abus.

## MVP Phase 2 — Foundation

La première implémentation autorisée comprend uniquement :

- un frontend TypeScript strict ;
- le layout, la navigation et le pied de page ;
- une homepage présentable ;
- toutes les routes et leurs états structurés ;
- les composants du design system ;
- le registre typé des langues et capacités ;
- les messages d'interface français et anglais ;
- un client API abstrait et sans modèle ML ;
- des contenus pédagogiques de démonstration clairement marqués ;
- des tests de rendu, navigation, responsive et accessibilité.

La Foundation ne comprend pas :

- migration complète de l'ASR ;
- nouveau modèle ou nouvel entraînement ;
- traduction réelle ou simulée ;
- contenu dioula définitif ;
- scoring de prononciation ;
- progression persistée ;
- authentification ;
- forum fonctionnel ;
- exposition publique du backend ou du checkpoint.

## Portée future ordonnée

1. connecter l'ASR existant à une API versionnée et sécurisée ;
2. ajouter un fournisseur de traduction derrière l'abstraction et l'évaluer
   pour chaque paire ;
3. faire valider le premier cours et ses médias par des experts linguistiques ;
4. introduire comptes, inscriptions et progression avec suppression des données ;
5. ajouter exercices, puis prononciation seulement après protocole validé ;
6. ouvrir une communauté légère après mise en place de la modération ;
7. intégrer d'autres langues au moyen du registre et de contrats identiques.

## Critères d'acceptation de la Foundation

- le démonstrateur Gradio et les tests Python existants restent opérationnels ;
- aucune page web ne charge un modèle ML directement ;
- toutes les routes demandées existent et partagent le même layout ;
- les fonctions indisponibles affichent `coming_soon` et ne renvoient aucun faux
  résultat ;
- les messages `fr` et `en` sont externalisés ;
- le code `dyu` est utilisé de façon cohérente ;
- tous les contenus linguistiques non validés portent la mention DEMO ;
- navigation clavier, contraste, reflow et annonces asynchrones sont testés ;
- aucun checkpoint, audio, corpus, chemin personnel, transcription ou secret
  n'entre dans Git ou dans les fixtures frontend ;
- aucune requête de la Foundation ne peut atteindre le final holdout ;
- les validations Python et frontend sont exécutées avant revue ;
- la documentation distingue clairement Legacy Demo UI et New Web Platform.
