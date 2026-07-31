# Guide de démonstration Phase 5B — trois minutes

## Préparation hors caméra

Configurer les quatre emplacements locaux sans les inscrire dans Git :

```bash
make ui \
  DIOULA_DATA_DIR="/path/to/voices_data" \
  ARTIFACTS_DIR="/path/to/artifacts" \
  MODEL_CACHE_DIR="/path/to/cache/models" \
  DIOULA_PILOT_MODEL_PATH="/path/to/checkpoint-000140"
```

Puis :

1. ouvrir `http://127.0.0.1:7860` ;
2. vérifier la présence de Tiny baseline, Small baseline et Tiny Dioula adapté
   pilote ;
3. choisir un audio dioula court autorisé pour la démonstration et préparer sa
   référence ;
4. effectuer une répétition Tiny baseline contre Tiny adapté ;
5. vérifier que le benchmark distingue explicitement les Expériences A et B ;
6. fermer tout écran qui montre un chemin local ou un nom réel.

L'interface doit rester locale : ne jamais activer le partage public Gradio.

## Script chronométré

| Temps | Action | Message proposé |
|---:|---|---|
| 0:00–0:20 | Présenter l'accueil | « IvoireVoice AI traite un problème d'ASR à faibles ressources : transcrire le dioula avec des données locales strictement séparées par locuteur. » |
| 0:20–0:55 | Importer l'audio et sa référence | « Cet audio ne quitte pas la machine. La référence rend possibles le WER et le CER pour cette démonstration ; sans elle, ces métriques restent indisponibles. » |
| 0:55–1:30 | Sélectionner Tiny baseline et Tiny adapté, puis lancer | « Les modèles sont chargés séquentiellement et libérés après chaque exécution. Le checkpoint adapté est un pilote entraîné sur 2 250 audios et validé sur 600 audios. » |
| 1:30–1:55 | Montrer les deux cartes | « Nous comparons la transcription, le temps, le RTF, le WER et le CER. Une valeur WER, CER ou RTF plus faible est meilleure. » |
| 1:55–2:30 | Ouvrir Benchmark, Expérience A | « Sur les mêmes 600 audios de validation, l'adaptation réduit le WER de 32,25 % et le CER de 51,44 % relativement à Tiny baseline. Il s'agit de validation pilote, pas du holdout final. » |
| 2:30–2:45 | Montrer l'Expérience B | « Le pilote historique compare Tiny et Small baseline sur 150 autres audios. Ces résultats ne sont pas mélangés avec l'Expérience A. » |
| 2:45–3:00 | Conclure sur les limites | « Le modèle reste un pilote, le final_holdout n'a pas été évalué et le baoulé est une perspective. Aucun résultat final n'est revendiqué. » |

## Résultats à annoncer exactement

### Expérience A — validation pilote, 600 audios

| Modèle | WER | CER | RTF | Loss validation |
|---|---:|---:|---:|---:|
| Whisper Tiny baseline | 115,4527 % | 71,7216 % | 0,012340 | 5,267802 |
| Whisper Tiny Dioula adapté pilote | 78,2165 % | 34,8294 % | 0,018742 | 1,229892 |

- réduction WER : 37,2362 points, soit 32,25 % relativement ;
- réduction CER : 36,8922 points, soit 51,44 % relativement.

### Expérience B — pilote historique, 150 audios

| Modèle | WER | CER | RTF |
|---|---:|---:|---:|
| Whisper Tiny baseline | 114,36 % | 73,74 % | 0,0296 |
| Whisper Small baseline | 152,35 % | 86,11 % | 0,0900 |

Un WER supérieur à 100 % est possible lorsque les insertions sont nombreuses.
Les deux expériences utilisent des jeux différents et ne permettent pas une
comparaison croisée directe.

## Mode de secours

Si l'inférence en direct échoue :

1. signaler sobrement que l'échec du modèle est isolé par l'interface ;
2. ouvrir l'onglet **Analyse des erreurs**, qui utilise les prédictions
   Phase 4C préenregistrées et anonymisées ;
3. montrer une ligne améliorée, la référence, les prédictions baseline/adaptée
   et leurs erreurs individuelles ;
4. poursuivre avec le benchmark structuré préenregistré ;
5. ne jamais inventer ou recalculer une métrique pendant la présentation.

Ce mode démontre l'écart baseline/adapté avec les mêmes résultats persistés,
mais ne prétend pas être une nouvelle inférence.

## Réponses courtes

**Le modèle adapté est-il final ?**

Non. Il s'agit du checkpoint pilote `checkpoint-000140`.

**Pourquoi le RTF adapté est-il un peu supérieur ?**

La qualité pilote s'améliore ici au prix d'un temps d'inférence légèrement
supérieur. Les mesures dépendent aussi du matériel et du protocole.

**Pourquoi ne pas annoncer une performance de généralisation ?**

Les chiffres viennent de la validation pilote. Le `final_holdout` n'a pas été
évalué et reste réservé à une décision ultérieure explicite.

**Les données ou le checkpoint sont-ils publiés ?**

Non. Les audios, chemins privés, noms réels et poids adaptés restent locaux.
