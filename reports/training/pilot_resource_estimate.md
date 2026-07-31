# Estimation des ressources — entraînement complet dioula

Cette estimation extrapole le pilote Phase 4C ; aucun entraînement complet n'a
été lancé.

- audios pilote train : 2250
- audios train complets : 13764
- facteur de volume : 6.117
- durée pilote observée : 782.980 s
- estimation 1 époque complète : 79.83 min
- estimation 2 époques complètes : 159.66 min
- pic VRAM observé : 1957.83 MiB
- taille d'un checkpoint observé : 0.423 GiB
- réserve recommandée pour deux checkpoints et les journaux :
  5.00 GiB

## Configuration complète proposée, non exécutée

- base : `openai/whisper-tiny` à la même révision ;
- train complet : 13 764 audios, validation complète : 2 661 audios ;
- 2 époques maximum avec early stopping ;
- batch CUDA : 4 ;
- accumulation : 4 ;
- learning rate initial : `1e-05` ;
- warmup ratio : `0.05` ;
- weight decay : `0.01` ;
- fp16 et gradient checkpointing conservés ;
- meilleur checkpoint choisi sur WER validation ;
- test final strictement réservé à l'évaluation finale.

L'extrapolation est linéaire et doit conserver une marge d'au moins 30 % pour
les évaluations régulières, les entrées/sorties et les variations matérielles.
