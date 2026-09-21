"""Lecture du CSV HarmonicSS et reconstitution de l'identité patient.

Ce module est la **seule** porte d'entrée vers les métadonnées du corpus. Tout le reste
du pipeline (``splits.py``, ``dataset.py``, l'évaluation, l'explication post-hoc)
consomme la table produite ici et ne relit jamais le CSV directement.

Périmètre — phase 1
-------------------
Le module lit du texte tabulaire. Il n'ouvre aucune image, ne calcule aucun descripteur
et n'injecte aucune connaissance experte : il produit des *étiquettes* et des
*identifiants de regroupement*. La contrainte « le modèle ne reçoit que l'image » porte
sur ``dataset.py`` ; les colonnes cliniques exposées ici (centre, appareil, glande, âge)
servent exclusivement au découpage, à la stratification et à l'analyse des erreurs —
jamais d'entrée au réseau.

Le problème central : il n'y a pas d'identifiant patient
--------------------------------------------------------
Le CSV fournit un ``Anonymized ID`` par **image**, pas par patient. Or un même patient
contribue plusieurs images (parotide et sous-mandibulaire, potentiellement droite et
gauche), acquises lors du même examen, sur le même appareil, sur le même tissu. Si deux
images d'un même patient se retrouvent de part et d'autre d'un découpage train/test, le
réseau n'a pas besoin d'apprendre la pathologie : il lui suffit de reconnaître le
patient. Le score devient excellent et entièrement trompeur, et rien dans les courbes
d'entraînement ne le signale.

Faute d'identifiant dédié, l'identité est **reconstituée** par la signature
démographique ``centre + sexe + âge + durée de maladie``. Cette reconstitution est
approximative et ce module ne le cache pas : il la produit *et* il en audite les
défauts (cf. :class:`PatientIdentityAudit`).

Pourquoi une reconstitution *conservatrice*
-------------------------------------------
Les deux erreurs possibles ne sont pas symétriques :

* **Sur-regroupement** (fusionner deux patients distincts en un seul groupe) — coûte de
  la puissance statistique : moins de groupes indépendants, stratification par grade
  plus difficile, plis plus déséquilibrés. Ne crée **aucune** fuite.
* **Sous-regroupement** (scinder un patient réel en deux groupes) — crée une fuite
  silencieuse qui invalide toutes les métriques en aval.

La signature démographique ne peut que sur-regrouper (deux images d'un même patient
partagent nécessairement centre, sexe, âge et durée). On conserve donc la clé telle
quelle par défaut : elle se trompe du bon côté. Le prix payé est mesuré et reporté par
l'audit, il n'est pas dissimulé.

Ce que l'audit révèle sur ce corpus
-----------------------------------
Trois anomalies structurelles, à connaître avant d'interpréter le moindre chiffre :

1. **Groupes surdimensionnés.** Un patient peut légitimement fournir jusqu'à quatre
   images (parotide et sous-mandibulaire, droite et gauche). Au-delà, la fusion est
   presque certaine. Le corpus contient des groupes de 6 et 8 images.
2. **Clé dégénérée à Ljubljana.** ``disease duration`` y vaut 0 pour les 50 images : la
   clé s'y réduit de fait à ``centre + sexe + âge``, donc y sur-regroupe davantage.
3. **Identifiants non contigus.** Certains groupes rassemblent des ``Anonymized ID``
   éloignés dans la numérotation (p. ex. 4, 5, 33, 34), ce qui suggère deux patients
   distincts de mêmes caractéristiques démographiques plutôt qu'un seul.

Ces constats ne sont pas des défauts du code : ce sont des propriétés du jeu de données,
et ils bornent ce que l'étude peut affirmer.

Utilisation
-----------
    >>> from sjogren.data.metadata import load_metadata, audit_patient_identity
    >>> df = load_metadata()
    >>> audit = audit_patient_identity(df)
    >>> print(audit.summary())

En ligne de commande, pour inspecter le corpus sans écrire de script :

    python -m sjogren.data.metadata
"""

from __future__ import annotations

import argparse
import warnings
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

__all__ = [
    "CANONICAL_COLUMNS",
    "DEFAULT_DATASET_DIR",
    "DEFAULT_PATIENT_KEY_FIELDS",
    "GRADE_LABELS",
    "MAX_PLAUSIBLE_IMAGES_PER_PATIENT",
    "MetadataWarning",
    "PatientIdentityAudit",
    "audit_patient_identity",
    "build_patient_id",
    "describe_corpus",
    "load_metadata",
]


# --------------------------------------------------------------------------------------
# Constantes du corpus
# --------------------------------------------------------------------------------------

#: Racine du jeu de données, relative à la racine du dépôt (ce fichier est à
#: ``<repo>/sjogren/data/metadata.py``, d'où les trois remontées).
DEFAULT_DATASET_DIR = Path(__file__).resolve().parents[2] / "HarmonicSS benchmark dataset"

CSV_FILENAME = "Anonymized images - Info.csv"
IMAGES_DIRNAME = "Anonymized images"

#: Le CSV est exporté depuis un tableur européen : séparateur point-virgule, et cinq
#: colonnes fantômes issues des ``;;;;;`` de fin de ligne.
CSV_SEPARATOR = ";"

#: Correspondance en-tête brut -> nom canonique. Les en-têtes bruts contiennent des
#: espaces de fin, des majuscules et une barre oblique : inutilisables tels quels comme
#: identifiants. On fige la correspondance plutôt que de normaliser automatiquement,
#: pour qu'un renommage en amont dans le CSV provoque une erreur explicite au lieu de
#: produire silencieusement une colonne au nom inattendu.
CANONICAL_COLUMNS: dict[str, str] = {
    "Center": "center",
    "Anonymized ID": "image_id",
    "parotid/submandibular": "gland",
    "machine": "machine",
    "De Vita et al. score": "devita",
    "OMERACT score": "omeract",
    "Sex": "sex",
    "Age at US evaluation": "age",
    "disease duration": "disease_duration",
    "fulfillment of 2016 ACR-EULAR classification criteria for pSS": "acr_eular_2016",
}

#: Champs composant la signature d'identité patient, dans l'ordre de concaténation.
DEFAULT_PATIENT_KEY_FIELDS: tuple[str, ...] = (
    "center",
    "sex",
    "age",
    "disease_duration",
)

#: Grades De Vita et al. attendus.
GRADE_LABELS: dict[int, str] = {
    0: "parenchyme homogène",
    1: "hétérogénéité discrète",
    2: "hétérogénéité marquée, zones hypo/anéchogènes",
    3: "destruction parenchymateuse étendue",
}

#: Nombre d'images qu'un patient unique peut plausiblement fournir : deux types de
#: glande (parotide, sous-mandibulaire) x deux côtés (droite, gauche). Au-delà, le
#: groupe fusionne très probablement plusieurs patients.
MAX_PLAUSIBLE_IMAGES_PER_PATIENT = 4

#: Nombre d'images d'un *même* type de glande plausible pour un patient unique
#: (droite + gauche).
MAX_PLAUSIBLE_IMAGES_PER_GLAND = 2

#: Domaines de valeurs attendus. Une valeur hors domaine n'interrompt pas la lecture
#: mais est signalée : c'est un fait sur les données, pas un bug à masquer.
EXPECTED_DOMAINS: dict[str, set] = {
    "gland": {"parotid", "submandibular"},
    "sex": {"F", "M"},
    "devita": {0, 1, 2, 3},
    "omeract": {0, 1, 2, 3},
    "acr_eular_2016": {0, 1},
}


class MetadataWarning(UserWarning):
    """Anomalie constatée dans les métadonnées, non bloquante mais à reporter."""


# --------------------------------------------------------------------------------------
# Lecture et normalisation
# --------------------------------------------------------------------------------------


def _read_raw_csv(csv_path: Path) -> pd.DataFrame:
    """Lit le CSV brut et ramène ses en-têtes à la forme canonique.

    Trois nettoyages sont nécessaires et assumés ici, une fois pour toutes :

    * les en-têtes portent des espaces parasites (``"...criteria for pSS "``) ;
    * l'export tableur ajoute cinq colonnes entièrement vides (``Unnamed: 10..14``),
      artefact des ``;;;;;`` terminaux ;
    * les valeurs textuelles peuvent porter des espaces de début ou de fin, ce qui
      scinderait un centre en deux modalités distinctes.
    """
    if not csv_path.is_file():
        raise FileNotFoundError(
            f"CSV de métadonnées introuvable : {csv_path}\n"
            "Vérifier l'emplacement du dossier 'HarmonicSS benchmark dataset'."
        )

    # utf-8-sig absorbe un éventuel BOM ; repli latin-1 pour un export Windows ancien.
    for encoding in ("utf-8-sig", "latin-1"):
        try:
            raw = pd.read_csv(csv_path, sep=CSV_SEPARATOR, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
    else:  # pragma: no cover - les deux encodages couvrent tous les cas observés
        raise UnicodeDecodeError(
            "utf-8-sig", b"", 0, 1, f"Encodage illisible pour {csv_path}"
        )

    raw = raw.rename(columns=lambda c: str(c).strip())

    # Colonnes fantômes : entièrement vides et absentes de la correspondance canonique.
    phantom = [
        c
        for c in raw.columns
        if c not in CANONICAL_COLUMNS and raw[c].isna().all()
    ]
    raw = raw.drop(columns=phantom)

    missing = [c for c in CANONICAL_COLUMNS if c not in raw.columns]
    if missing:
        raise KeyError(
            "Colonnes attendues absentes du CSV : "
            + ", ".join(repr(c) for c in missing)
            + f"\nColonnes présentes : {list(raw.columns)}\n"
            "Si le CSV a été renommé en amont, mettre à jour CANONICAL_COLUMNS."
        )

    unexpected = [c for c in raw.columns if c not in CANONICAL_COLUMNS]
    if unexpected:
        warnings.warn(
            f"Colonnes inattendues ignorées : {unexpected}",
            MetadataWarning,
            stacklevel=3,
        )

    df = raw[list(CANONICAL_COLUMNS)].rename(columns=CANONICAL_COLUMNS)

    for col in df.columns:
        if df[col].dtype == object or isinstance(df[col].dtype, pd.StringDtype):
            df[col] = df[col].astype("string").str.strip()

    return df


def _coerce_types(df: pd.DataFrame) -> pd.DataFrame:
    """Force les types numériques et catégoriels.

    Les entiers sont lus en ``Int64`` (entier *nullable*) plutôt qu'en ``int64`` : une
    cellule vide deviendrait sinon un flottant ``NaN`` et ``devita`` passerait
    silencieusement de ``3`` à ``3.0``, ce qui casse toute comparaison d'étiquette.
    """
    integer_columns = ("image_id", "devita", "omeract", "age", "disease_duration", "acr_eular_2016")
    for col in integer_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in ("center", "gland", "machine", "sex"):
        df[col] = df[col].astype("string")

    return df


def _validate(df: pd.DataFrame) -> list[str]:
    """Contrôle les domaines de valeurs et retourne la liste des anomalies constatées.

    Ne lève pas : une anomalie de contenu est une information sur le corpus, qui doit
    remonter jusqu'aux documents de suivi plutôt que d'interrompre le pipeline.
    """
    issues: list[str] = []

    missing_counts = df.isna().sum()
    for col, n in missing_counts.items():
        if n:
            issues.append(f"{n} valeur(s) manquante(s) dans la colonne '{col}'")

    for col, domain in EXPECTED_DOMAINS.items():
        observed = set(df[col].dropna().tolist())
        out_of_domain = observed - domain
        if out_of_domain:
            n = int(df[col].isin(list(out_of_domain)).sum())
            issues.append(
                f"colonne '{col}' : {n} ligne(s) hors domaine attendu {sorted(domain)} "
                f"-> valeurs {sorted(out_of_domain, key=str)}"
            )

    if df["image_id"].duplicated().any():
        dups = sorted(df.loc[df["image_id"].duplicated(keep=False), "image_id"].dropna().unique())
        issues.append(f"identifiants d'image dupliqués : {dups}")

    # Confusion centre/appareil : contrainte structurelle n°2 du protocole. On la
    # vérifie au lieu de la supposer, car elle conditionne le leave-one-center-out.
    per_center = df.groupby("center", observed=True)["machine"].nunique()
    multi = per_center[per_center > 1]
    if not multi.empty:
        issues.append(
            "plusieurs appareils dans un même centre : "
            + ", ".join(f"{c} ({n})" for c, n in multi.items())
            + " — la confusion centre/appareil n'est plus totale, le leave-one-center-out "
            "change d'interprétation"
        )

    return issues


def _attach_image_paths(df: pd.DataFrame, images_dir: Path, check_exists: bool) -> pd.DataFrame:
    """Associe à chaque ligne le chemin de son image (``001.jpg`` … ``225.jpg``).

    La correspondance ``Anonymized ID`` -> nom de fichier est un zéro-padding sur trois
    chiffres. Elle est vérifiée sur disque par défaut : une ligne sans image est une
    image perdue avant même le prétraitement, et le contrôle « aucune image perdue » de
    l'étape 1.1 commence ici.
    """
    df["image_filename"] = df["image_id"].map(
        lambda i: f"{int(i):03d}.jpg" if pd.notna(i) else pd.NA
    ).astype("string")
    df["image_path"] = df["image_filename"].map(
        lambda name: str(images_dir / name) if pd.notna(name) else pd.NA
    ).astype("string")

    if check_exists:
        if not images_dir.is_dir():
            raise FileNotFoundError(f"Dossier d'images introuvable : {images_dir}")
        missing = [
            name for name in df["image_filename"].dropna() if not (images_dir / name).is_file()
        ]
        if missing:
            warnings.warn(
                f"{len(missing)} image(s) référencée(s) au CSV mais absente(s) du disque : "
                f"{missing[:10]}{' …' if len(missing) > 10 else ''}",
                MetadataWarning,
                stacklevel=3,
            )
        orphans = sorted(
            p.name
            for p in images_dir.glob("*.jpg")
            if p.name not in set(df["image_filename"].dropna())
        )
        if orphans:
            warnings.warn(
                f"{len(orphans)} image(s) sur disque sans ligne au CSV : "
                f"{orphans[:10]}{' …' if len(orphans) > 10 else ''}",
                MetadataWarning,
                stacklevel=3,
            )

    return df


# --------------------------------------------------------------------------------------
# Reconstitution de l'identité patient
# --------------------------------------------------------------------------------------


def _center_code(center: str) -> str:
    """Code court et stable d'un centre, pour rendre les identifiants lisibles."""
    return "".join(ch for ch in str(center).upper() if ch.isalpha())[:3] or "XXX"


def build_patient_id(
    df: pd.DataFrame,
    key_fields: Sequence[str] = DEFAULT_PATIENT_KEY_FIELDS,
) -> pd.Series:
    """Construit l'identifiant patient reconstitué, un par ligne.

    L'identifiant est une chaîne **lisible** du type ``UDI-F-A072-D018`` plutôt qu'un
    hachage : lors du débogage d'un découpage ou de l'analyse d'un cas mal classé, on
    veut lire le groupe fautif, pas un condensé hexadécimal. Il reste parfaitement
    déterministe, donc utilisable tel quel comme argument ``groups`` de
    ``StratifiedGroupKFold``.

    Parameters
    ----------
    df
        Table canonique issue de :func:`load_metadata`.
    key_fields
        Champs composant la signature. Paramétrable pour permettre une étude de
        sensibilité du découpage à la définition de l'identité (p. ex. retirer
        ``disease_duration``, constant à Ljubljana, et mesurer l'effet sur le nombre
        de groupes).

    Returns
    -------
    pandas.Series
        Identifiant patient, aligné sur l'index de ``df``.
    """
    missing = [f for f in key_fields if f not in df.columns]
    if missing:
        raise KeyError(f"Champs de clé absents de la table : {missing}")

    def _fmt(row: pd.Series) -> str:
        parts: list[str] = []
        for fld in key_fields:
            value = row[fld]
            if pd.isna(value):
                parts.append("NA")
            elif fld == "center":
                parts.append(_center_code(value))
            elif fld == "age":
                parts.append(f"A{int(value):03d}")
            elif fld == "disease_duration":
                parts.append(f"D{int(value):03d}")
            else:
                parts.append(str(value))
        return "-".join(parts)

    return df.apply(_fmt, axis=1).astype("string")


@dataclass
class PatientIdentityAudit:
    """Diagnostic de la reconstitution d'identité patient.

    La reconstitution démographique ne peut pas être validée (aucune vérité terrain
    n'existe) ; elle peut seulement être **auditée**. Cet objet rassemble les indices
    de sur-regroupement, afin qu'ils soient reportés dans les documents de suivi plutôt
    que découverts après coup en lisant une matrice de confusion suspecte.
    """

    n_images: int
    n_patients: int
    key_fields: tuple[str, ...]
    group_size_distribution: dict[int, int]
    oversized_groups: pd.DataFrame
    duplicate_gland_groups: pd.DataFrame
    non_contiguous_groups: pd.DataFrame
    degenerate_key_fields: dict[str, list[str]]
    validation_issues: list[str] = field(default_factory=list)

    @property
    def images_per_patient(self) -> float:
        return self.n_images / self.n_patients if self.n_patients else float("nan")

    @property
    def n_suspicious_groups(self) -> int:
        """Nombre de groupes distincts portant au moins un indice de fusion."""
        keys = set()
        for frame in (
            self.oversized_groups,
            self.duplicate_gland_groups,
            self.non_contiguous_groups,
        ):
            if not frame.empty:
                keys.update(frame["patient_id"].tolist())
        return len(keys)

    def summary(self) -> str:
        """Rapport textuel, destiné à la console et aux documents de suivi."""
        lines = [
            "Reconstitution de l'identité patient",
            "=" * 60,
            f"Champs de la signature   : {' + '.join(self.key_fields)}",
            f"Images                   : {self.n_images}",
            f"Patients reconstitués    : {self.n_patients}",
            f"Images par patient       : {self.images_per_patient:.2f} en moyenne",
            "",
            "Distribution de la taille des groupes",
            "-" * 60,
        ]
        for size, count in sorted(self.group_size_distribution.items()):
            flag = "  <-- au-delà du plausible" if size > MAX_PLAUSIBLE_IMAGES_PER_PATIENT else ""
            lines.append(f"  {count:3d} groupe(s) de {size} image(s){flag}")

        lines += ["", "Indices de sur-regroupement", "-" * 60]
        lines.append(
            f"  groupes de plus de {MAX_PLAUSIBLE_IMAGES_PER_PATIENT} images : "
            f"{len(self.oversized_groups)}"
        )
        lines.append(
            f"  groupes avec plus de {MAX_PLAUSIBLE_IMAGES_PER_GLAND} images d'une même "
            f"glande : {len(self.duplicate_gland_groups)}"
        )
        lines.append(
            f"  groupes à identifiants non contigus            : "
            f"{len(self.non_contiguous_groups)}"
        )
        lines.append(f"  groupes suspects (au moins un indice)         : {self.n_suspicious_groups}")

        if self.degenerate_key_fields:
            lines += ["", "Champs dégénérés (constants sur un centre)", "-" * 60]
            for fld, centers in self.degenerate_key_fields.items():
                lines.append(
                    f"  '{fld}' constant à : {', '.join(centers)} "
                    "-> la clé y perd une dimension et sur-regroupe davantage"
                )

        if self.validation_issues:
            lines += ["", "Anomalies de contenu", "-" * 60]
            lines += [f"  - {issue}" for issue in self.validation_issues]

        lines += [
            "",
            "Lecture : la signature démographique ne peut que fusionner des patients",
            "distincts, jamais scinder un patient réel. Le sur-regroupement coûte de la",
            "puissance statistique ; il ne crée pas de fuite. Les groupes listés ci-dessus",
            "sont donc à considérer comme des unités de découpage prudentes, non comme des",
            "patients avérés.",
        ]
        return "\n".join(lines)


def audit_patient_identity(
    df: pd.DataFrame,
    key_fields: Sequence[str] = DEFAULT_PATIENT_KEY_FIELDS,
    validation_issues: Iterable[str] | None = None,
) -> PatientIdentityAudit:
    """Audite la reconstitution d'identité et recense les collisions probables.

    Trois indices indépendants de fusion de patients sont recherchés :

    ``oversized``
        Groupe de plus de :data:`MAX_PLAUSIBLE_IMAGES_PER_PATIENT` images. Un examen
        complet couvre au plus deux glandes x deux côtés ; au-delà, la fusion est
        quasi certaine.
    ``duplicate_gland``
        Groupe comportant plus de :data:`MAX_PLAUSIBLE_IMAGES_PER_GLAND` images d'un
        même type de glande — un patient n'a que deux parotides.
    ``non_contiguous``
        Groupe dont les ``Anonymized ID`` ne se suivent pas. Les images d'un même examen
        sont numérotées consécutivement dans ce corpus ; un écart marqué signale deux
        patients de mêmes caractéristiques démographiques saisis à des moments
        différents. Indice **heuristique** : il repose sur une régularité observée de la
        numérotation, non sur une garantie documentée.

    Un quatrième contrôle porte sur la clé elle-même : un champ constant à l'intérieur
    d'un centre n'apporte aucun pouvoir discriminant et y réduit d'autant la signature.
    """
    if "patient_id" not in df.columns:
        df = df.assign(patient_id=build_patient_id(df, key_fields))

    grouped = df.groupby("patient_id", observed=True)
    sizes = grouped.size()

    size_distribution = {int(k): int(v) for k, v in sizes.value_counts().items()}

    def _rows(mask_keys: list[str], reason: str) -> pd.DataFrame:
        if not mask_keys:
            return pd.DataFrame(
                columns=["patient_id", "n_images", "image_ids", "glands", "devita", "reason"]
            )
        records = []
        for key in sorted(mask_keys):
            sub = df[df["patient_id"] == key].sort_values("image_id")
            records.append(
                {
                    "patient_id": key,
                    "n_images": len(sub),
                    "image_ids": [int(i) for i in sub["image_id"].dropna()],
                    "glands": sub["gland"].tolist(),
                    "devita": [int(g) for g in sub["devita"].dropna()],
                    "reason": reason,
                }
            )
        return pd.DataFrame.from_records(records)

    oversized_keys = sizes[sizes > MAX_PLAUSIBLE_IMAGES_PER_PATIENT].index.tolist()

    duplicate_gland_keys = [
        key
        for key, sub in grouped
        if (sub["gland"].value_counts() > MAX_PLAUSIBLE_IMAGES_PER_GLAND).any()
    ]

    non_contiguous_keys = []
    for key, sub in grouped:
        ids = sorted(int(i) for i in sub["image_id"].dropna())
        if len(ids) > 1 and (ids[-1] - ids[0] + 1) != len(ids):
            non_contiguous_keys.append(key)

    degenerate: dict[str, list[str]] = {}
    for fld in key_fields:
        if fld == "center":
            continue
        constant_centers = [
            str(center)
            for center, sub in df.groupby("center", observed=True)
            if sub[fld].nunique(dropna=False) <= 1
        ]
        if constant_centers:
            degenerate[fld] = sorted(constant_centers)

    return PatientIdentityAudit(
        n_images=len(df),
        n_patients=int(df["patient_id"].nunique()),
        key_fields=tuple(key_fields),
        group_size_distribution=size_distribution,
        oversized_groups=_rows(oversized_keys, "taille au-delà du plausible"),
        duplicate_gland_groups=_rows(duplicate_gland_keys, "glande en surnombre"),
        non_contiguous_groups=_rows(non_contiguous_keys, "identifiants non contigus"),
        degenerate_key_fields=degenerate,
        validation_issues=list(validation_issues or []),
    )


# --------------------------------------------------------------------------------------
# API publique
# --------------------------------------------------------------------------------------


def load_metadata(
    dataset_dir: Path | str | None = None,
    *,
    csv_path: Path | str | None = None,
    images_dir: Path | str | None = None,
    key_fields: Sequence[str] = DEFAULT_PATIENT_KEY_FIELDS,
    check_images: bool = True,
    warn_on_collisions: bool = True,
) -> pd.DataFrame:
    """Charge les métadonnées et retourne la table canonique du corpus.

    Parameters
    ----------
    dataset_dir
        Racine du jeu de données. Par défaut :data:`DEFAULT_DATASET_DIR`.
    csv_path, images_dir
        Surcharges explicites, utiles en test sur un corpus factice.
    key_fields
        Champs de la signature d'identité patient.
    check_images
        Vérifie sur disque la présence de chaque image et signale les orphelines.
    warn_on_collisions
        Émet un :class:`MetadataWarning` si l'audit détecte des groupes suspects. À
        désactiver uniquement dans les tests qui auditent explicitement ces cas.

    Returns
    -------
    pandas.DataFrame
        Une ligne par image, triée par ``image_id``, index réinitialisé. Colonnes :

        ======================  ===========================================================
        ``image_id``            identifiant anonymisé de l'**image** (1..225)
        ``center``              centre d'acquisition
        ``machine``             appareil (confondu avec le centre sur ce corpus)
        ``gland``               ``parotid`` ou ``submandibular``
        ``devita``              grade De Vita 0-3 — **étiquette cible principale**
        ``omeract``             score OMERACT 0-3, étiquette alternative
        ``sex``, ``age``        démographie
        ``disease_duration``    durée de maladie, en années
        ``acr_eular_2016``      satisfaction des critères de classification 2016
        ``patient_id``          identité **reconstituée** — argument ``groups`` du découpage
        ``grade_binary``        dérivée : 0 si ``devita == 0``, 1 sinon (sain vs atteint)
        ``image_filename``      ``001.jpg`` …
        ``image_path``          chemin absolu, seule colonne consommée par ``dataset.py``
        ======================  ===========================================================

    Notes
    -----
    ``grade_binary`` est une **dérivation d'étiquette**, pas une connaissance experte
    ajoutée : elle recode la cible pour la configuration binaire complémentaire de
    l'étape 1.2. Le seuil retenu (grade 0 contre grades 1-3) suit la lecture clinique
    usuelle, où tout grade non nul traduit une atteinte parenchymateuse.
    """
    root = Path(dataset_dir) if dataset_dir is not None else DEFAULT_DATASET_DIR
    csv = Path(csv_path) if csv_path is not None else root / CSV_FILENAME
    imgs = Path(images_dir) if images_dir is not None else root / IMAGES_DIRNAME

    df = _read_raw_csv(csv)
    df = _coerce_types(df)
    issues = _validate(df)

    df["patient_id"] = build_patient_id(df, key_fields)

    # Cible binaire dérivée : Int64 pour rester homogène avec ``devita``.
    df["grade_binary"] = (df["devita"] > 0).astype("Int64")
    df.loc[df["devita"].isna(), "grade_binary"] = pd.NA

    df = _attach_image_paths(df, imgs, check_exists=check_images)

    df = df.sort_values("image_id", kind="stable").reset_index(drop=True)

    # L'audit est systématique : on ne veut pas d'un chemin d'exécution où les
    # collisions passent inaperçues parce que personne n'a pensé à appeler l'auditeur.
    df.attrs["validation_issues"] = issues
    df.attrs["key_fields"] = tuple(key_fields)
    audit = audit_patient_identity(df, key_fields, validation_issues=issues)
    df.attrs["patient_identity_audit"] = audit

    for issue in issues:
        warnings.warn(f"Métadonnées : {issue}", MetadataWarning, stacklevel=2)

    if warn_on_collisions and audit.n_suspicious_groups:
        warnings.warn(
            f"Identité patient : {audit.n_suspicious_groups} groupe(s) sur "
            f"{audit.n_patients} portent un indice de fusion de patients distincts "
            f"(centre+sexe+âge+durée ne désambiguïse pas tout). Le découpage reste sûr "
            f"— la clé sur-regroupe, elle ne scinde pas — mais le nombre de groupes "
            f"indépendants est surestimé à la baisse. Détail : "
            f"df.attrs['patient_identity_audit'].summary()",
            MetadataWarning,
            stacklevel=2,
        )

    return df


def describe_corpus(df: pd.DataFrame) -> dict[str, object]:
    """Statistiques descriptives du corpus, destinées aux documents de suivi.

    Toutes les valeurs retournées sont **analytiques** : elles se déduisent de la
    distribution du corpus, sans entraînement ni tirage aléatoire. Elles peuvent donc
    figurer dans ``results.html`` sans ``run_id``, à condition d'être étiquetées comme
    telles.
    """
    audit: PatientIdentityAudit | None = df.attrs.get("patient_identity_audit")

    grade_counts = {int(k): int(v) for k, v in df["devita"].value_counts().sort_index().items()}
    n = len(df)
    majority = max(grade_counts, key=grade_counts.get) if grade_counts else None

    # F1 macro du classifieur trivial (répond toujours la classe majoritaire) : la
    # seule classe prédite a une précision = sa prévalence et un rappel de 1 ; les
    # autres ont un F1 nul. D'où F1_macro = (2p / (1 + p)) / n_classes.
    trivial_f1_macro = None
    if majority is not None and n:
        prevalence = grade_counts[majority] / n
        trivial_f1_macro = (2 * prevalence / (1 + prevalence)) / len(grade_counts)

    return {
        "n_images": n,
        "n_patients": audit.n_patients if audit else int(df["patient_id"].nunique()),
        "grade_counts": grade_counts,
        "grade_binary_counts": {
            int(k): int(v) for k, v in df["grade_binary"].value_counts().sort_index().items()
        },
        "gland_counts": {str(k): int(v) for k, v in df["gland"].value_counts().items()},
        "center_counts": {str(k): int(v) for k, v in df["center"].value_counts().items()},
        "machine_counts": {str(k): int(v) for k, v in df["machine"].value_counts().items()},
        "center_machine_pairs": {
            str(c): sorted(str(m) for m in sub["machine"].dropna().unique())
            for c, sub in df.groupby("center", observed=True)
        },
        "grade_by_center": {
            str(c): {int(g): int(v) for g, v in sub["devita"].value_counts().sort_index().items()}
            for c, sub in df.groupby("center", observed=True)
        },
        "trivial_accuracy": grade_counts[majority] / n if majority is not None and n else None,
        "trivial_f1_macro": trivial_f1_macro,
        "group_size_distribution": audit.group_size_distribution if audit else None,
        "n_suspicious_groups": audit.n_suspicious_groups if audit else None,
    }


# --------------------------------------------------------------------------------------
# Inspection en ligne de commande
# --------------------------------------------------------------------------------------


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspection des métadonnées HarmonicSS et audit de l'identité patient."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help="Racine du jeu de données (défaut : %(default)s)",
    )
    parser.add_argument(
        "--key-fields",
        nargs="+",
        default=list(DEFAULT_PATIENT_KEY_FIELDS),
        help="Champs de la signature d'identité patient.",
    )
    parser.add_argument(
        "--no-check-images",
        action="store_true",
        help="N'effectue pas la vérification de présence des images sur disque.",
    )
    parser.add_argument(
        "--show-collisions",
        action="store_true",
        help="Détaille chaque groupe suspect.",
    )
    args = parser.parse_args(argv)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MetadataWarning)
        df = load_metadata(
            args.dataset_dir,
            key_fields=args.key_fields,
            check_images=not args.no_check_images,
        )

    stats = describe_corpus(df)
    audit: PatientIdentityAudit = df.attrs["patient_identity_audit"]

    print("Corpus HarmonicSS SGUS")
    print("=" * 60)
    print(f"Images                : {stats['n_images']}")
    print(f"Grades De Vita        : {stats['grade_counts']}")
    print(f"Binaire sain/atteint  : {stats['grade_binary_counts']}")
    print(f"Glandes               : {stats['gland_counts']}")
    print(f"Centre -> appareil    : {stats['center_machine_pairs']}")
    print(f"Trivial (majoritaire) : accuracy={stats['trivial_accuracy']:.3f} "
          f"F1 macro={stats['trivial_f1_macro']:.3f}  [analytique]")
    print()
    print(audit.summary())

    if args.show_collisions:
        print()
        print("Détail des groupes suspects")
        print("-" * 60)
        detail = pd.concat(
            [
                audit.oversized_groups,
                audit.duplicate_gland_groups,
                audit.non_contiguous_groups,
            ],
            ignore_index=True,
        )
        if detail.empty:
            print("  aucun")
        else:
            with pd.option_context("display.max_rows", None, "display.width", 200):
                print(detail.sort_values(["patient_id", "reason"]).to_string(index=False))

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
