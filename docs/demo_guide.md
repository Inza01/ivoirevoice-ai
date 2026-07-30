# Guide de démonstration — trois minutes

## Préparation hors caméra

1. Lancer `make ui` avec les trois racines locales configurées.
2. Ouvrir `http://127.0.0.1:7860`.
3. Choisir un audio dioula court dont l'usage local est autorisé.
4. Préparer sa référence dans un fichier TXT.
5. Vérifier que les onglets Benchmark et Analyse des erreurs chargent leurs
   artefacts.
6. Fermer toute fenêtre affichant un chemin, un nom de fichier privé ou un
   terminal contenant une transcription.

Ne jamais activer un lien public Gradio et ne jamais montrer le manifeste
complet dans la vidéo.

## Script chronométré

| Temps | Action à l'écran | Message oral |
|---:|---|---|
| 0:00–0:20 | Onglet À propos | « IvoireVoice AI étudie la transcription du français et du dioula, deux contextes encore mal couverts par les modèles génériques. » |
| 0:20–0:40 | Résumé des données | « Le corpus dioula local v0.1 contient 19 199 audios, 21 locuteurs et un split strict par locuteur. Les données restent privées. » |
| 0:40–1:20 | Onglet Transcrire : importer l'audio, la référence et sélectionner Tiny/Small | « L'interface charge les modèles à la demande et les exécute séquentiellement, sans envoyer l'audio à une API. » |
| 1:20–1:45 | Afficher les cartes côte à côte | « Chaque carte montre la transcription, la latence, le RTF et, puisque j'ai fourni une référence, le WER et le CER. » |
| 1:45–2:15 | Onglet Benchmark | « Sur les mêmes 150 audios, Tiny obtient 114,36 % de WER contre 152,35 % pour Small. Tiny est aussi environ trois fois plus rapide. » |
| 2:15–2:38 | Onglet Analyse des erreurs | « Cette vue locale explique substitutions, insertions et suppressions sans corriger artificiellement la sortie du modèle. » |
| 2:38–3:00 | Onglet À propos | « Les baselines restent insuffisantes, ce qui justifie une adaptation locale prudente. Le futur modèle adapté s'ajoutera sans remplacer les baselines, puis l'architecture pourra être étendue au baoulé. » |

## Résultats à annoncer exactement

| Modèle | Audios réussis | WER micro | CER micro | RTF |
|---|---:|---:|---:|---:|
| Whisper Tiny | 150/150 | 114,36 % | 73,74 % | 0,0296 |
| Whisper Small | 150/150 | 152,35 % | 86,11 % | 0,0900 |

Un WER supérieur à 100 % est possible lorsque les insertions sont nombreuses.
Il ne faut pas décrire Tiny comme « précis » : il est seulement meilleur que
Small sur ce pilote.

## Réponses courtes aux questions probables

**Pourquoi Tiny plutôt que Small ?**  
Tiny obtient de meilleurs WER et CER sur la même sélection et son RTF est
environ trois fois inférieur.

**Pourquoi ne pas afficher une confiance ?**  
La sortie Whisper utilisée ici ne fournit pas une probabilité calibrée pouvant
être présentée honnêtement comme une confiance.

**Les données sont-elles publiées ?**  
Non. La licence et le consentement de redistribution ne sont pas confirmés :
audios, références, prédictions et modèles dérivés restent locaux.

**Pourquoi l'interface est-elle prête avant le fine-tuning ?**  
Elle garantit un démonstrateur fonctionnel et une comparaison avant/après sur
le même protocole, même si l'entraînement prend plus de temps que prévu.
