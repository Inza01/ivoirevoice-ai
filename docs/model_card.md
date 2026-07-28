# Carte des modèles

## Statut

Seul `DummyBackend` est disponible. Il ne réalise aucune inférence et son texte
de sortie indique explicitement qu'il s'agit d'une transcription fictive. Il
sert à tester les contrats, l'API, l'interface et la CI sans réseau ni GPU.

Whisper Small et Wav2Vec2 XLSR sont uniquement décrits comme pistes de
configuration. Ils ne sont ni implémentés, ni téléchargés, ni évalués dans
cette phase. Aucun score de performance n'est revendiqué.

## Contrat

Tout futur backend doit implémenter `ASRBackend` et retourner le schéma
`TranscriptionResult`. Il doit déclarer ses langues prises en charge et libérer
ses ressources avec `unload()`.

