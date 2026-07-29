# Carte des données

## Statut

Le corpus dioula reste hors Git et est audité en lecture seule par la Phase 3A.
Les rapports complets et le manifeste provisoire sont des artefacts locaux non
versionnés. Le français n'est pas traité dans cette phase.

La licence et le consentement restent `unknown`. Sur instruction du formateur,
le périmètre provisoire est `local_research_only` : les expériences locales
peuvent continuer, mais cela ne constitue pas une autorisation de publier les
audios, le manifeste complet ou un modèle dérivé.

## Informations vérifiées

- la racine de chaque `clips.json` inspecté est une liste ;
- les enregistrements peuvent contenir `id`, `sequence`, `glob`, `audioSrc` et
  `sentence` ;
- `sentence` peut contenir `id`, `text` et `sequence` ;
- le corpus possède des dossiers `men` et `women`, des WAV, des MP4 et des
  variantes textuelles avec et sans tons ;
- certains fichiers suffixés `.wav` sont en réalité des conteneurs ISO Base
  Media/AAC ; ils sont signalés comme `format_mismatch` et ne sont pas utilisés
  lorsqu'un WAV converti relié par `sequence` est disponible ;
- les URL `audioSrc` peuvent contenir des paramètres signés.

Ces constats structurels ne prouvent ni l'identité réelle d'un locuteur, ni la
licence, ni le consentement.

## Informations inconnues ou à confirmer

- licence et droits de redistribution ;
- consentement et usages autorisés ;
- méthode de collecte et population représentée ;
- relation exacte entre dossier et locuteur physique ;
- exhaustivité et qualité linguistique des transcriptions ;
- politique finale concernant les tons ;
- pertinence des catégories `men` et `women`, qui proviennent seulement des
  noms de dossiers.

## Schéma du manifeste

Le manifeste conserve une référence audio, une transcription et un identifiant
de locuteur. Il contient `text_raw`, une version NFC et une version dont seuls
les espaces et retours de ligne sont normalisés. Il ne supprime aucun ton. Les
chemins audio et JSON sont relatifs à la racine du corpus. Les identifiants de
locuteurs sont des pseudonymes stables dérivés de la structure locale.

Le candidat d'entraînement conserve également une variante NFC sans tons,
reliée aux fichiers `text-no-tones` du corpus. Pour le premier MVP uniquement,
`target_text_mvp` utilise cette variante et retire le marqueur d'intonation
descendante `↘` afin de réduire la complexité du vocabulaire. Le marqueur reste
intégralement conservé dans `text_raw` et sa présence est représentée par
`intonation_falling`. La variante tonale et le texte brut restent présents pour
rendre une comparaison ultérieure possible.

## Confidentialité et publication

- ne jamais publier ni journaliser les URL complètes `audioSrc` ;
- ne jamais versionner les audios ou le manifeste complet avant clarification
  de la licence ;
- ne jamais chercher à réidentifier les personnes ;
- ne partager que des agrégats et de petites fixtures artificielles ;
- conserver les données brutes inchangées.

Les expériences locales autorisées par le formateur restent limitées à
`local_research_only`. Cette instruction ne remplace pas une licence et ne
permet pas de publier un modèle dérivé.

## Partitionnement

Les ensembles d'entraînement, validation et test sont séparés strictement par
locuteur dans la version locale v0.1. La stratégie B validée humainement
affecte 15 locuteurs à l'entraînement, 3 à la validation et 3 au test. Chaque
split conserve des locuteurs issus des dossiers `men` et `women`, sans qu'il
soit possible d'en déduire une identité ou un genre déclaré. Le baoulé pourra
être ajouté ultérieurement avec sa propre configuration.

## Limites connues

- l'inférence des locuteurs repose actuellement sur la structure des dossiers ;
- les groupes trop petits limitent la stratification ;
- les fichiers manquants, ambigus ou corrompus doivent être examinés ;
- les conflits entre un même audio et plusieurs transcriptions sont mis en
  quarantaine ;
- les références répétées et les doublons SHA-256 sont exclus du candidat ;
- un SHA-256 identique indique des octets identiques, pas nécessairement une
  transcription correcte ;
- aucune métrique de modèle ne découle de cet audit.

## Version locale v0.1

Le manifeste `dioula_manifest_v0.1.csv` et ses métadonnées sont des artefacts
locaux hors Git. Le statut est `frozen_candidate`, le périmètre est
`local_research_only`, et les autorisations de publication du dataset et de
tout modèle dérivé sont toutes deux `false`. Les 1 885 lignes sans audio
récupérable et les deux lignes du conflit SHA-256 restent exclues.
