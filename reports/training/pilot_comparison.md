# Comparaison validation — pilote Whisper Tiny dioula

> Rapport agrégé sans identifiant, chemin, référence ni prédiction. Les
> métriques utilisent le même sous-ensemble de validation gelé ; aucun audio
> test n'a été chargé ou transcrit.

| Métrique | Baseline | Adapté |
|---|---:|---:|
| WER micro | 1.154527 | 0.782165 |
| CER micro | 0.717216 | 0.348294 |
| RTF | 0.012340 | 0.018742 |
| Latence moyenne (s) | 0.054136 | 0.082223 |
| Substitutions | 3968 | 2936 |
| Insertions | 1195 | 748 |
| Suppressions | 1621 | 912 |

- réduction absolue WER : 0.372362 ;
- réduction relative WER : 32.25 % ;
- réduction absolue CER : 0.368922 ;
- réduction relative CER : 51.44 %.

Ces résultats servent au choix expérimental sur validation uniquement. Ils ne
constituent pas une évaluation finale.
