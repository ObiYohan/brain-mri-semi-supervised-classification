# Détection de tumeurs cérébrales sur IRM avec peu de labels

Projet de formation : peut-on classer des IRM cérébrales (**cancer** / **normal**) quand on ne dispose que de **100 images labellisées** et de **1 400 images sans label** ?

## Données

- 1 506 IRM en JPEG, 512×512 px
- 100 images labellisées (50 cancer / 50 normal) + 1 406 non labellisées
- Jeu de données fourni dans le cadre de la formation (usage académique), non inclus dans ce dépôt

## Démarche

1. **Exploration** : contrôle de l'intégrité des fichiers, des dimensions et des formats (aucune image corrompue, tailles homogènes).
2. **Extraction d'embeddings** : un ResNet50 pré-entraîné sur ImageNet, gelé et sans sa couche de classification, transforme chaque image en un vecteur de 2 048 dimensions.
3. **Analyse non supervisée** : standardisation, PCA et comparaison de deux méthodes de clustering sur le jeu labellisé (ARI : K-Means 0,29, clustering hiérarchique 0,77).
4. **Apprentissage semi-supervisé** : une régression logistique est entraînée sur les embeddings, puis enrichie par pseudo-labellisation des images non labellisées (`SelfTrainingClassifier`). Une variante par propagation de labels (`LabelSpreading`) est aussi testée.

## Résultats

| Approche | Évaluation | AUC | Accuracy | Rappel « cancer » |
|---|---|---|---|---|
| Régression logistique (100 images labellisées) | validation croisée 5 folds | 0,96 ± 0,02 | – | – |
| Self-training (50 labels + 1 406 non labellisées) | jeu de test de 50 images | 0,95 | 0,90 | 0,92 |

Le self-training attribue un pseudo-label à 1 252 des 1 406 images non labellisées.

> Ces chiffres reposent sur un très petit jeu de test : ils donnent une tendance, pas une performance clinique.

## Stack

Python · PyTorch / torchvision · scikit-learn · pandas · matplotlib / seaborn

## Lancer le projet

```bash
uv sync
```

Placer les images dans `data/mri_dataset_brain_cancer_oc/` (sous-dossiers `avec_labels/cancer`, `avec_labels/normal` et `sans_label`), puis exécuter le notebook `prep_data.ipynb`.
