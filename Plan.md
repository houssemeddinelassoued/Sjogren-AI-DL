# Plan de travail — Diagnostic du Syndrome de Gougerot-Sjögren par IA (Échographie des Glandes Salivaires)

## Contexte et cadre clinique

Le syndrome de Gougerot-Sjögren se manifeste à l'échographie des glandes salivaires par une altération progressive du parenchyme : hétérogénéité de la texture, apparition de zones anéchoïques et hypoéchogènes, perte de la définition des contours glandulaires. Ces signes sont formalisés par le score de **De Vita et al.**, qui gradue l'atteinte de **0 à 3**.

Cette sémiologie — essentiellement **texturale** — motive l'ensemble des choix méthodologiques du projet : les descripteurs de texture et les architectures profondes sont sélectionnés pour leur capacité à capturer ces altérations fines du parenchyme.

## Contributions scientifiques visées

1. **Pipeline hybride vision classique + deep learning** : injection de descripteurs de texture explicites (GLCM, LBP) et de points d'intérêt structurels en complément de l'image brute, plutôt que de laisser le réseau tout apprendre seul.
2. **Classification multiclasse des grades** : prédiction du stade d'avancement selon les 4 grades De Vita (0 à 3), et non une simple détection Sain/Malade — objectif cliniquement plus utile et techniquement plus exigeant.
3. **Robustesse par isolement de la région d'intérêt** : neutralisation des artefacts propres à l'échographie (bords de sonde, texte et annotations incrustés par l'appareil) afin que l'apprentissage porte exclusivement sur le tissu glandulaire.
4. **Explicabilité multi-méthodes** : confrontation de SHAP et LIME pour valider que les prédictions reposent bien sur la texture interne de la glande, condition d'acceptabilité clinique du système.
5. **Alignement vision-langage** : extraction de modalités textuelles à partir des images via CLIP, ouvrant la voie à une architecture RAG pour un diagnostic assisté.

---

## Jeu de données — HarmonicSS SGUS benchmark

Dataset public de référence (étude multicentrique européenne, [doi:10.3389/fmed.2020.581248](https://doi.org/10.3389/fmed.2020.581248)), présent dans `HarmonicSS benchmark dataset/`.

- **225 images** échographiques (`Anonymized images/001.jpg` … `225.jpg`), la numérotation correspondant à la colonne `Anonymized ID` du fichier `Anonymized images - Info.csv` (séparateur `;`).
- **Annotations disponibles** : score De Vita (0-3), score OMERACT, type de glande, appareil, centre, sexe, âge, durée de la maladie, satisfaction des critères ACR-EULAR 2016.
- **Type de glande** : 115 parotides / 110 sous-mandibulaires.

### Distribution des grades De Vita

| Grade | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| Images | 110 | 28 | 55 | 32 |

### Contraintes méthodologiques imposées par ces données

Trois caractéristiques du jeu de données conditionnent la conception du pipeline et doivent être traitées explicitement :

1. **Fort déséquilibre de classes** : le grade 0 représente près de la moitié du corpus tandis que le grade 1 n'en compte que 28 images. Impose une `CrossEntropyLoss` pondérée, un échantillonnage équilibré, et un F1 macro (et non l'accuracy) comme métrique de sélection.
2. **Confusion centre/appareil** : chaque centre utilise un seul appareil (Udine-Samsung 75, Belgrade-GE 53, Ljubljana-Philips 50, Milan-Esaote 47). La signature visuelle de l'appareil est donc parfaitement corrélée au centre, et le modèle risque d'apprendre l'appareil plutôt que la pathologie. À contrôler par normalisation d'intensité, par le masquage des incrustations propres à chaque constructeur, et à tester par une validation *leave-one-center-out*.
3. **Plusieurs images par patient** : parotide et sous-mandibulaire d'un même patient figurent en deux lignes distinctes (mêmes centre, sexe, âge et durée de maladie). Le découpage train/val/test doit être **groupé par patient** pour éviter toute fuite de données ; l'identité patient étant à reconstituer à partir de ces colonnes, faute d'identifiant dédié.

**Taille modeste (225 images)** : justifie le recours au transfer learning plutôt qu'à un entraînement from scratch profond, une augmentation de données soutenue, et une validation croisée k-fold avec intervalles de confiance bootstrap plutôt qu'un simple split unique.

---

## Stack technique requise

- **Framework principal** : PyTorch, PyTorch Lightning (optionnel mais recommandé pour structurer)
- **Vision classique & prétraitement** : scikit-image, OpenCV (cv2), Pillow
- **Modèles de fondation** : `transformers` (Hugging Face) pour CLIP ; `timm` pour ViT
- **XAI (explicabilité)** : shap, lime, captum (optimisé pour PyTorch)
- **Rigueur statistique** : scipy / scikit-learn (bootstrap, validation croisée, intervalles de confiance)
- **Outils divers** : numpy, pandas, matplotlib, scikit-learn (métriques)

---

## 🛠️ ÉTAPE 1 : Ingénierie des caractéristiques & Pipeline de Données

**Objectif** : créer un DataLoader robuste intégrant des descripteurs de texture et des algorithmes de vision classique avant l'injection dans les réseaux de neurones, en garantissant que seule l'information utile — le tissu glandulaire — est présentée au modèle.

## Tâches :

- Un script de prétraitement avec scikit-image pour extraire des descripteurs de texture (GLCM — Gray-Level Co-occurrence Matrix, LBP — Local Binary Patterns).
- L'implémentation d'un détecteur de coins (Harris ou Shi-Tomasi) pour identifier les structures anatomiques clés sur l'échographie.
- **Isolement de la région d'intérêt** : détection et neutralisation des artefacts d'acquisition (bords de sonde, texte et annotations incrustés par l'appareil) par crop de la ROI et/ou masquage, pour que l'apprentissage porte exclusivement sur le parenchyme glandulaire.
- **Préparation des labels multiclasses** : encodage selon les 4 grades De Vita (0 à 3), avec une option de regroupement binaire (Sain vs Malade) comme configuration d'étude complémentaire.
- Une classe `Dataset` PyTorch personnalisée (ex: `SjogrenDataset`) qui charge les images, applique les augmentations (`torchvision.transforms` ou `albumentations`) et concatène potentiellement l'image brute avec les cartes de caractéristiques (feature engineering).

---

## 🧠 ÉTAPE 2 : Modélisation Deep Learning (CNN, Transfer Learning & ViT)

**Objectif** : mettre en place et comparer les architectures pour la classification multiclasse des grades De Vita (0 à 3), ainsi que pour la configuration binaire complémentaire.

## Tâches :

- **CNN From Scratch** : architecture personnalisée (3 à 5 couches convolutives) optimisée pour repérer des anomalies de texture fines, servant de référence interne pour mesurer l'apport des architectures suivantes.
- **Transfer Learning (ResNet)** : script chargeant un ResNet18 ou ResNet50 pré-entraîné, en modifiant la dernière couche (Fully Connected) pour l'adapter à la configuration multiclasse comme binaire.
- **Vision Transformer (ViT)** : architecture pré-entraînée (via `timm` ou `transformers`), dont le mécanisme d'attention globale est évalué face aux convolutions locales de ResNet sur ces textures.
- **Boucle d'entraînement** complète : optimiseur AdamW, `CrossEntropyLoss` avec pondération pour gérer le déséquilibre entre grades, sauvegarde du meilleur modèle selon le F1-score (macro pour le multiclasse) ou l'AUC.
- **Évaluation rigoureuse** : validation croisée k-fold, intervalles de confiance bootstrap sur chaque métrique (accuracy, précision, sensibilité, spécificité, F1), matrice de confusion par grade pour identifier les confusions inter-stades.

---

## 🔍 ÉTAPE 3 : Explicabilité de l'IA (XAI) — Casser la boîte noire

**Objectif** : interpréter les prédictions des modèles de l'Étape 2 pour comprendre quelles zones de la glande salivaire déclenchent la classification d'un grade spécifique, et établir que le modèle s'appuie sur la texture du parenchyme.

## Tâches :

- **Intégration de SHAP** : un script utilisant `shap.DeepExplainer` ou `shap.GradientExplainer` sur le modèle ResNet/ViT entraîné pour générer des heatmaps superposées aux échographies originales.
- **Intégration de LIME** : un script utilisant `lime_image` pour segmenter l'échographie en superpixels et montrer les zones qui contribuent positivement ou négativement à la prédiction.
- **Analyse comparative SHAP vs LIME** : mesure de concordance entre les deux méthodes ; une explication corroborée par deux approches indépendantes est nettement plus défendable cliniquement.
- **Validation du focus anatomique** : vérification que les zones saillantes tombent bien dans la ROI glandulaire et non sur les artefacts d'acquisition, avec quantification du recouvrement explication/ROI.
- Une fonction pour exporter ces visualisations en haute qualité, prêtes pour être intégrées dans le manuscrit de thèse.

---

## 🌐 ÉTAPE 4 : Transition Multimodale avec CLIP (Alignement Vision-Langage)

**Objectif** : préparer le terrain pour l'architecture RAG en extrayant des modalités textuelles à partir des images en utilisant le modèle CLIP d'OpenAI.

## Tâches :

- Un script chargeant le modèle `CLIPModel` et `CLIPProcessor` via Hugging Face.
- **Génération de Prompts** : une logique pour construire un lexique clinique (ex: "Échographie d'une glande salivaire saine", "Échographie montrant une hétérogénéité sévère, grade 3 de Sjögren", etc.), aligné sur les 4 grades De Vita.
- **Inférence et Similarité** : un script qui calcule la similarité cosinus entre les embeddings de nos images de test et les embeddings de nos descriptions textuelles (zero-shot classification ou image-to-text retrieval) pour évaluer la capacité de CLIP à "comprendre" nos échographies.
- **Perspective (hors code immédiat)** : cette étape ouvre la voie à une architecture RAG (recherche de cas similaires et génération de rapport assisté), à détailler une fois l'Étape 4 validée.

---

## 📄 Documents vivants — règle de tenue permanente

Deux pages HTML à la racine du projet constituent la vitrine et la mémoire des travaux. Elles ne sont
pas des livrables ponctuels : **elles doivent refléter l'état réel du projet à tout moment**.

| Fichier | Rôle | Contenu |
|---|---|---|
| `research_works.html` | Méthodologie et suivi | Cadre clinique, données, pipeline, justification des choix, statut des étapes, feuille de route, journal des travaux |
| `results.html` | Résultats chiffrés | Toutes les mesures, étape par étape, avec intervalles de confiance et `run_id` de traçabilité |

### Règle de mise à jour

**Toute modification du code, tout résultat produit et tout avancement d'étape doit être répercuté
dans ces deux fichiers dans la même session de travail.** Concrètement :

- **Nouveau module écrit ou modifié** → mettre à jour le tableau des livrables de l'étape concernée et
  l'arborescence de code dans `research_works.html` (badge `à faire` → `fait`).
- **Run produisant des chiffres** → renseigner les cellules correspondantes dans `results.html`,
  ajouter une ligne au journal des runs avec le `run_id`, et mettre à jour le tableau de bord de
  `research_works.html`.
- **Étape franchie** → mettre à jour la section Statut, le diagramme de trajectoire et le journal des
  travaux dans `research_works.html`.
- **Décision méthodologique** (choix d'architecture, changement de métrique, révision du prétraitement)
  → consigner la décision et sa justification dans `research_works.html`, datée dans le journal.
- **Résultat négatif ou décevant** → le consigner tel quel. Ne jamais effacer ni réécrire une
  affirmation antérieure : ajouter un avertissement daté qui explique ce qui l'invalide.

### Règles de contenu

- Aucun chiffre de performance dans `results.html` sans un `run_id` traçable vers
  `outputs/<run_id>/metrics.json`. Les seules exceptions sont les valeurs **analytiques**, calculées
  directement depuis la distribution du corpus, et étiquetées comme telles.
- Les conditions d'évaluation (métriques, schéma de validation, seuils) sont fixées **avant** la mesure
  et ne sont pas ajustées après coup.
- Les deux pages doivent rester cohérentes entre elles : un statut annoncé dans l'une doit correspondre
  aux chiffres présents dans l'autre.

---