# Guide de démonstration Phase 5B — trois minutes

## Préparation hors caméra

Configurer les emplacements locaux sans les inscrire dans Git :

```bash
make ui \
  DIOULA_DATA_DIR="/path/to/voices_data" \
  ARTIFACTS_DIR="/path/to/artifacts" \
  MODEL_CACHE_DIR="/path/to/cache/models" \
  DIOULA_FINAL_MODEL_PATH="/path/to/checkpoint-002052"
```

Puis :

1. ouvrir `http://127.0.0.1:7860` ;
2. vérifier les libellés exacts **Whisper Tiny — Baseline**,
   **Whisper Small — Baseline** et **Whisper Tiny — Dioula Final** ;
3. choisir un audio dioula court autorisé pour la démonstration et préparer sa
   référence ;
4. effectuer une répétition Tiny baseline contre Tiny Dioula Final ;
5. vérifier que le benchmark distingue explicitement les Expériences A et B ;
6. fermer tout écran qui montre un chemin local ou un nom réel.

L'interface doit rester locale : ne jamais activer le partage public Gradio.

## Script chronométré

| Temps | Action | Message proposé |
|---:|---|---|
| 0:00–0:20 | Présenter l'accueil | « IvoireVoice AI traite un problème d'ASR à faibles ressources : transcrire le dioula avec des données locales strictement séparées par locuteur. » |
| 0:20–0:55 | Importer l'audio et sa référence | « Cet audio ne quitte pas la machine. La référence rend possibles le WER et le CER pour cette démonstration ; sans elle, ces métriques restent indisponibles. » |
| 0:55–1:30 | Sélectionner Tiny baseline et Tiny Dioula Final, puis lancer | « Les modèles sont chargés séquentiellement et libérés après chaque exécution. Le modèle final a été refit sur 16 425 audios ; ses poids restent locaux. » |
| 1:30–1:55 | Montrer les deux cartes | « Nous comparons la transcription, le temps, le RTF, le WER et le CER. Une valeur WER, CER ou RTF plus faible est meilleure. » |
| 1:55–2:20 | Ouvrir À propos et montrer le résultat final | « Sur l'unique évaluation du holdout indépendant de 2 624 audios, le modèle gelé obtient 33,26 % de WER, 12,38 % de CER et un RTF de 0,00785. » |
| 2:20–2:45 | Ouvrir Benchmark | « Les 600 audios de validation pilote et les 150 audios du pilote historique forment deux expériences anciennes distinctes. Ils ne sont ni fusionnés ni comparés directement au holdout final. » |
| 2:45–3:00 | Conclure sur les limites | « Ce modèle final reste expérimental : seulement trois groupes de locuteurs composent le holdout, et les accents, le bruit, les microphones ou le domaine peuvent changer la qualité. Le baoulé reste une perspective. » |

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
mais ne prétend pas être une nouvelle inférence du modèle final. Le résultat
final agrégé reste disponible dans l'onglet **À propos** ; le holdout n'est
jamais rouvert pour la démonstration.

## Réponses courtes

**Le modèle adapté est-il final ?**

Oui, l'interface charge le refit gelé `checkpoint-002052`. « Final » signifie
ici fin du protocole expérimental, pas qualité industrielle ni autorisation de
publication des poids.

**Le RTF de l'audio importé sera-t-il exactement 0,00785 ?**

Non. `0,00785` est l'agrégat du final holdout sur le matériel du run officiel.
Le temps d'un autre audio dépend du matériel, de sa durée et du protocole.

**Pourquoi ne pas annoncer une performance de généralisation ?**

Le holdout de 2 624 audios est resté indépendant jusqu'à l'unique évaluation
du modèle gelé. Il ne contient que trois groupes de locuteurs et ne garantit
pas une performance identique en production ou sur un autre corpus.

**Les données ou le checkpoint sont-ils publiés ?**

Non. Les audios, chemins privés, noms réels et poids adaptés restent locaux.
