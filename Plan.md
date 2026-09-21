# Plan de travail — Diagnostic du Syndrome de Gougerot-Sjögren par IA (Échographie des Glandes Salivaires)

## Contexte et cadre clinique

Le syndrome de Gougerot-Sjögren se manifeste à l'échographie des glandes salivaires par une altération progressive du parenchyme : hétérogénéité de la texture, apparition de zones anéchoïques et hypoéchogènes, perte de la définition des contours glandulaires. Ces signes sont formalisés par le score de **De Vita et al.**, qui gradue l'atteinte de **0 à 3**.

Cette sémiologie — essentiellement **texturale** — motive l'ensemble des choix méthodologiques du projet.

---

## Découpage du projet en deux phases

Le projet avance **phase par phase**. Une phase n'est ouverte que lorsque la précédente est achevée et analysée.

| Phase | Objet | État |
|---|---|---|
| **Phase 1** | Approche **boîte noire** avec **explication post-hoc** | **en cours** |
| **Phase 2** | Intégration de l'**expertise humaine** dans la chaîne | différée — périmètre à définir |

### Ce que « boîte noire » signifie ici

En phase 1, le modèle reçoit **uniquement l'image**, sans aucune connaissance du domaine injectée en amont : pas de descripteur de texture calculé à la main, pas de région d'intérêt tracée par un expert, pas de lexique clinique. On observe ce qu'un réseau profond apprend seul, puis on cherche **après coup** à comprendre sur quoi il fonde ses décisions.

Le seul prétraitement autorisé est le **nettoyage des artefacts d'acquisition** (bords de sonde, texte incrusté par l'appareil). Il ne s'agit pas d'expertise injectée mais d'hygiène des données : sans lui, les explications post-hoc décriraient la réaction du modèle à des incrustations d'écran, et non à la glande.

### Exigence de rigueur

Chaque phase demande un travail exhaustif : l'objectif n'est pas d'obtenir un chiffre, mais de **produire une analyse défendable** — ce que le modèle a appris, ce qu'il rate, ce à quoi il est sensible, et ce que les méthodes d'explication peuvent ou ne peuvent pas établir. Les livrables d'analyse et de limites comptent autant que le code.

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

1. **Fort déséquilibre de classes** : le grade 0 représente près de la moitié du corpus tandis que le grade 1 n'en compte que 28 images. Impose une `CrossEntropyLoss` pondérée, un échantillonnage équilibré, et un F1 macro (et non l'accuracy) comme métrique de sélection.
2. **Confusion centre/appareil** : chaque centre utilise un seul appareil (Udine-Samsung 75, Belgrade-GE 53, Ljubljana-Philips 50, Milan-Esaote 47). La signature visuelle de l'appareil est parfaitement corrélée au centre, et le modèle risque d'apprendre l'appareil plutôt que la pathologie. À contrôler par normalisation d'intensité et par une validation *leave-one-center-out*.
3. **Plusieurs images par patient** : parotide et sous-mandibulaire d'un même patient figurent en deux lignes distinctes (mêmes centre, sexe, âge et durée de maladie). Le découpage train/val/test doit être **groupé par patient** ; l'identité patient est à reconstituer à partir de ces colonnes, faute d'identifiant dédié.

**Taille modeste (225 images)** : justifie le transfer learning plutôt qu'un entraînement from scratch profond, une augmentation de données soutenue, et une validation croisée k-fold avec intervalles de confiance bootstrap plutôt qu'un simple split unique.

---

## Stack technique — phase 1

- **Framework principal** : PyTorch, PyTorch Lightning (optionnel, pour structurer)
- **Prétraitement** : OpenCV (cv2), Pillow, scikit-image
- **Architectures pré-entraînées** : `torchvision` (ResNet), `timm` (ViT)
- **XAI (explicabilité)** : shap, lime, captum
- **Rigueur statistique** : scipy / scikit-learn (bootstrap, validation croisée, intervalles de confiance)
- **Outils divers** : numpy, pandas, matplotlib

Aucune dépendance au-delà de ce périmètre n'est introduite tant que la phase 2 n'est pas spécifiée.

---

# PHASE 1 — Approche boîte noire avec explication post-hoc

**Objectif global** : établir ce qu'un réseau profond apprend seul des échographies de glandes salivaires, mesurer rigoureusement sa performance, puis déterminer *a posteriori* sur quoi reposent ses décisions — et jusqu'où cette explication est fiable.

## Étape 1.1 — Pipeline de données et garanties d'intégrité

**Objectif** : produire un `DataLoader` dont on peut démontrer qu'il ne fuit pas et qu'il ne présente au modèle que du tissu.

### Tâches :

- Lecture de `Anonymized images - Info.csv` et **reconstitution de l'identité patient** (centre + sexe + âge + durée de maladie), faute d'identifiant dédié.
- **Découpage groupé par patient et stratifié par grade** : aucune image d'un même patient de part et d'autre du découpage.
- **Nettoyage des artefacts d'acquisition** : détection et neutralisation des bords de sonde et du texte incrusté par l'appareil (crop et/ou masquage).
- **Normalisation d'intensité** inter-appareils, pour limiter la signature visuelle du constructeur.
- Classe `Dataset` PyTorch (`SjogrenDataset`) avec augmentations (`torchvision.transforms` ou `albumentations`), **recevant l'image seule**.
- **Tests d'intégrité** : aucun patient partagé entre plis, quatre grades représentés dans chaque pli, aucune image perdue au prétraitement.

### Critère de sortie :

Un `DataLoader` opérationnel, une planche de contrôle visuel des régions extraites pour chacun des quatre appareils, et les contrôles d'intégrité au vert et reportés dans `results.html`.

## Étape 1.2 — Modèles boîte noire

**Objectif** : entraîner les classifieurs sur l'image seule, en configuration multiclasse (grades De Vita 0-3).

### Tâches :

- **CNN from scratch** (3 à 5 couches convolutives) : référence interne, permet de situer l'apport du pré-entraînement.
- **Transfer Learning ResNet18 / ResNet50** : dernière couche entièrement connectée adaptée aux 4 grades.
- **Vision Transformer (ViT)** pré-entraîné : attention globale évaluée face aux convolutions locales.
- **Boucle d'entraînement** : AdamW, `CrossEntropyLoss` pondérée (déséquilibre inter-grades), arrêt anticipé et sélection du modèle sur le **F1 macro** de validation.
- **Configuration binaire complémentaire** (sain vs atteint) pour situer la difficulté relative des deux tâches.

### Critère de sortie :

Pour chaque architecture, un modèle entraîné, sauvegardé, et un `run_id` traçable vers `outputs/<run_id>/metrics.json`.

## Étape 1.3 — Protocole d'évaluation

**Objectif** : produire des chiffres défendables, et non un score unique issu d'un découpage favorable.

### Tâches :

- **Validation croisée k-fold** groupée par patient et stratifiée par grade.
- **Intervalles de confiance bootstrap** (95 %), rééchantillonnage au niveau patient.
- **Leave-one-center-out** : entraînement sur trois centres, test sur le quatrième — mesure directe du risque que le modèle ait appris l'appareil.
- **Matrice de confusion par grade** : distinguer les confusions entre grades voisins des confusions extrêmes.
- Métriques rapportées : F1 macro (sélection), sensibilité et spécificité par grade, AUC one-vs-rest, accuracy (référence uniquement).

### Critère de sortie :

Tous les tableaux de performance de `results.html` renseignés, chaque cellule tracée par un `run_id`.

## Étape 1.4 — Explication post-hoc

**Objectif** : déterminer *après coup* sur quoi le modèle fonde ses décisions, avec deux méthodes indépendantes.

### Tâches :

- **SHAP** : `shap.DeepExplainer` ou `shap.GradientExplainer` sur les modèles entraînés, heatmaps superposées aux échographies originales.
- **LIME** : `lime_image`, segmentation en superpixels, contributions positives et négatives.
- **Concordance SHAP / LIME** : mesure de l'accord entre les deux méthodes ; une explication corroborée par deux approches indépendantes est nettement plus défendable qu'une explication isolée.
- **Recouvrement anatomique ρ** : fraction de la saillance totale tombant dans la région glandulaire, plutôt qu'un jugement visuel.
- **Stabilité des explications** : sensibilité aux graines, aux perturbations d'entrée et au choix des données de référence de SHAP. Une explication instable n'est pas exploitable.
- **Explications des erreurs** : comparer les cartes de saillance des cas bien classés et des cas mal classés, par grade.

### Critère de sortie :

Cartes exportées en haute résolution pour le manuscrit, valeurs de ρ, de concordance et de stabilité reportées dans `results.html`.

## Étape 1.5 — Analyse, interprétation et limites

**Objectif** : le livrable scientifique de la phase. Il ne s'agit pas d'illustrer les résultats mais de les interpréter et d'en énoncer honnêtement la portée.

### Tâches :

- **Ce que le modèle a appris** : les régions saillantes correspondent-elles à la sémiologie clinique (hétérogénéité du parenchyme, zones anéchoïques) ou à autre chose ?
- **Où le modèle échoue** : analyse du grade 1 (28 images), des confusions entre grades voisins, et des cas systématiquement mal classés.
- **Sensibilité à l'appareil** : lecture de l'écart entre k-fold et leave-one-center-out.
- **Limites des méthodes d'explication elles-mêmes** : ce que SHAP et LIME établissent réellement, leurs désaccords, leur dépendance aux hyperparamètres — une carte de saillance n'est pas une preuve de causalité.
- **Limites de l'approche boîte noire** : ce que cette approche ne peut structurellement pas apporter, et qui motive l'ouverture de la phase 2.

### Critère de sortie :

Une section d'analyse rédigée dans `research_works.html` et une section « Limites » à jour dans `results.html`, toutes deux défendables devant l'encadrant et réutilisables dans le manuscrit.

---

# PHASE 2 — Intégration de l'expertise humaine dans la chaîne

**Statut : différée. Le périmètre sera défini avec l'encadrant à l'issue de la phase 1.**

Aucun choix technique n'est arrêté à ce stade, et aucun développement n'est engagé sur cette phase. Les pistes ci-dessous ont été évoquées au cours du cadrage du projet et sont conservées comme mémoire de travail — **elles ne constituent pas un engagement** :

- descripteurs de texture calculés (GLCM, LBP) et points d'intérêt structurels injectés en amont du réseau ;
- régions d'intérêt ou annotations fournies par un clinicien ;
- alignement vision-langage à partir d'un lexique clinique.

Le choix entre ces pistes, ou d'autres, dépendra directement des limites établies à l'étape 1.5.

---

## 📄 Documents vivants — règle de tenue permanente

Deux pages HTML à la racine du projet constituent la vitrine et la mémoire des travaux. Elles ne sont pas des livrables ponctuels : **elles doivent refléter l'état réel du projet à tout moment**.

| Fichier | Rôle | Contenu |
|---|---|---|
| `research_works.html` | Méthodologie et suivi | Cadre clinique, données, phase 1 détaillée, justification des choix, statut, feuille de route, journal des travaux |
| `results.html` | Résultats chiffrés | Toutes les mesures, étape par étape, avec intervalles de confiance et `run_id` de traçabilité |

### Règle de mise à jour

**Toute modification du code, tout résultat produit et tout avancement d'étape doit être répercuté dans ces deux fichiers dans la même session de travail.** Concrètement :

- **Nouveau module écrit ou modifié** → mettre à jour le tableau des livrables de l'étape concernée et l'arborescence de code dans `research_works.html` (badge `à faire` → `fait`).
- **Run produisant des chiffres** → renseigner les cellules correspondantes dans `results.html`, ajouter une ligne au journal des runs avec le `run_id`, et mettre à jour le tableau de bord de `research_works.html`.
- **Étape franchie** → mettre à jour la section Statut, le diagramme de trajectoire et le journal des travaux.
- **Décision méthodologique** → consigner la décision et sa justification, datée dans le journal.
- **Résultat négatif ou décevant** → le consigner tel quel. Ne jamais effacer ni réécrire une affirmation antérieure : ajouter un avertissement daté qui explique ce qui l'invalide.

### Règles de contenu

- Aucun chiffre de performance dans `results.html` sans un `run_id` traçable vers `outputs/<run_id>/metrics.json`. Seule exception : les valeurs **analytiques** calculées directement depuis la distribution du corpus, étiquetées comme telles.
- Les conditions d'évaluation (métriques, schéma de validation, seuils) sont fixées **avant** la mesure et ne sont pas ajustées après coup.
- **Ne rien documenter qui ne soit pas arrêté.** Les pistes non validées sont signalées comme telles ou tenues hors des documents — une page de suivi qui décrit des travaux non engagés induit en erreur sur l'état réel du projet.
- Les deux pages doivent rester cohérentes entre elles.
