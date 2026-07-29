# Carte des données

## Statut

Le corpus dioula reste hors Git et est audité en lecture seule par la Phase 3A.
Les rapports complets et le manifeste provisoire sont des artefacts locaux non
versionnés. Le français n'est pas traité dans cette phase.

La licence et le consentement restent inconnus. Sur instruction du formateur,
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

## Confidentialité et publication

- ne jamais publier ni journaliser les URL complètes `audioSrc` ;
- ne jamais versionner les audios ou le manifeste complet avant clarification
  de la licence ;
- ne jamais chercher à réidentifier les personnes ;
- ne partager que des agrégats et de petites fixtures artificielles ;
- conserver les données brutes inchangées.

## Partitionnement

Les ensembles d'entraînement, validation et test seront séparés strictement par
locuteur. La Phase 3A ne produit qu'une proposition déterministe, stratifiée
autant que possible selon les dossiers de genre. Une validation humaine est
obligatoire avant de renseigner la colonne `split`. Le baoulé pourra être ajouté
ultérieurement avec sa propre configuration.

## Limites connues

- l'inférence des locuteurs repose actuellement sur la structure des dossiers ;
- les groupes trop petits limitent la stratification ;
- les fichiers manquants, ambigus ou corrompus doivent être examinés ;
- un SHA-256 identique indique des octets identiques, pas nécessairement une
  transcription correcte ;
- aucune métrique de modèle ne découle de cet audit.
