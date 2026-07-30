# Analyse de normalisation textuelle — train dioula v0.1

Ce rapport est agrégé : aucun chemin, nom réel ou texte du dataset n'y figure.

## Résultats

| Contrôle | Nombre de lignes |
|---|---:|
| Lignes train analysées | 13764 |
| text_raw différent de text_no_tones | 13764 (100.00%) |
| text_raw identique à text_no_tones | 0 |
| text_raw non NFC | 4332 |
| text_no_tones non NFC | 0 |
| text_raw avec marques de ton | 13764 |
| text_raw avec apostrophe droite ou typographique | 2271 |
| text_raw avec ponctuation Unicode | 7409 |
| text_raw avec chiffres | 0 |
| text_raw avec espaces multiples | 27 |
| text_raw vide | 0 |
| text_no_tones vide | 0 |
| target_text_mvp vide | 0 |
| Lignes avec marque d'intonation descendante | 7908 |

## Caractères rares dans text_raw

| Codepoint | Nom Unicode | Occurrences |
|---|---|---:|
| `U+0041` | LATIN CAPITAL LETTER A | 3 |
| `U+005B` | LEFT SQUARE BRACKET | 4 |
| `U+005D` | RIGHT SQUARE BRACKET | 4 |
| `U+0303` | COMBINING TILDE | 3 |

## Colonne canonique proposée

**`target_text_mvp`** est retenue pour le smoke-overfit. Elle conserve la variante NFC sans tons et retire la marque prosodique `↘`, qui ne correspond pas à une unité lexicale à prédire. Ce choix réduit la sparsité orthographique sur seulement 10 à 20 exemples et reste cohérent avec l'évaluation des baselines.

`text_raw`, `text_with_tones_nfc`, `text_without_tones_nfc` et `target_text_mvp` restent toutes conservées dans le manifeste gelé. Aucune variante n'est supprimée ou réécrite.
