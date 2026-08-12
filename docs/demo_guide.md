# Guide de démonstration Phase 5B — trois minutes

## Préparation hors caméra

Le checkpoint et le cache restent hors Git. Configurer uniquement leur
emplacement local :

```bash
export IVOIREVOICE_DIOULA_FINAL_MODEL_PATH="/path/to/checkpoint-002052"
export IVOIREVOICE_MODEL_CACHE_DIR="/path/to/cache/models"
make demo-preflight
make demo-smoke
make demo
```

Le preflight exige `main` propre, CUDA, les révisions Tiny/Small en cache, le
hash final gelé, le port libre et au moins 2 GiB de disque. Il charge puis
libère le modèle final sur CUDA sans transcrire d'audio. `make demo` masque
volontairement les racines corpus et artefacts privés.

`make demo-smoke` remplace les commandes Python improvisées : il transcrit une
seconde de silence synthétique avec les trois modèles, contrôle le chemin
comparaison/métriques/exports, puis supprime ses fichiers temporaires. Ce test
est purement technique et ne constitue pas un benchmark ASR.

Un nouvel enregistrement au microphone peut être créé pendant la démo. Pour
un fichier externe préenregistré, confirmer explicitement sa provenance :

```bash
export IVOIREVOICE_DEMO_AUDIO_PATH="/path/to/external_demo.wav"
export IVOIREVOICE_DEMO_AUDIO_CONFIRMATION="SAFE_EXTERNAL_DEMO_AUDIO"
make demo-preflight
```

Le fichier ne doit provenir ni de train, validation, pilot test, final holdout,
ni d'un artefact de prédictions. Si aucun fichier sûr n'est disponible, le
preflight affiche `DEMO AUDIO REQUIRED` et aucun audio n'est choisi
automatiquement. Son statut est alors `READY WITH DEMO AUDIO REQUIRED` : les
modèles et l'environnement sont prêts, mais il faut encore fournir le nouvel
audio. Le dossier local recommandé `demo_inputs/` est ignoré par Git.

Puis :

1. ouvrir `http://127.0.0.1:7860` ;
2. vérifier les libellés exacts **Whisper Tiny — Baseline**,
   **Whisper Small — Baseline** et **Whisper Tiny — Dioula Final** ;
3. choisir un audio dioula court autorisé pour la démonstration et préparer sa
   référence ;
4. effectuer une répétition Tiny baseline contre Tiny Dioula Final ;
5. vérifier que le benchmark public distingue explicitement les expériences ;
6. constater que l'analyse d'erreurs privée est indisponible en mode démo ;
7. fermer tout écran qui montre un chemin local ou un nom réel.

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
2. ne pas ouvrir ni charger les prédictions privées historiques ;
3. poursuivre avec le benchmark public agrégé et l'onglet **À propos** ;
4. annoncer les résultats finaux gelés : WER 33,26 %, CER 12,38 %, RTF
   0,00785, sur 2 624 audios et 3 locuteurs ;
5. rappeler que le holdout indépendant a été évalué exactement une fois ;
6. ne jamais inventer ou recalculer une métrique pendant la présentation.

Le fallback utilise uniquement des agrégats publics. Il ne relance aucune
inférence scientifique et ne rouvre jamais le holdout.

## Incidents de démonstration

- Whisper Small lent : sélectionner uniquement Tiny baseline et Tiny Dioula
  Final.
- Microphone navigateur indisponible : utiliser l'upload de l'audio externe
  explicitement confirmé.
- Internet indisponible : les deux révisions épinglées doivent avoir été
  confirmées par `make demo-preflight`.
- Relance Gradio : interrompre le serveur, puis exécuter `make demo`.
- Port 7860 occupé : choisir un port libre sans exposer le serveur :

  ```bash
  make demo-preflight UI_PORT=7862
  make demo UI_PORT=7862
  ```

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
