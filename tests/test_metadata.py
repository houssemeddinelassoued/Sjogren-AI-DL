"""Tests d'intégrité de ``sjogren.data.metadata``.

Ce que ces tests verrouillent
-----------------------------
Le module des métadonnées est la fondation du découpage groupé par patient. Une
régression silencieuse ici ne produit pas une exception : elle produit des métriques
flatteuses et fausses, découvertes des semaines plus tard. Les invariants testés sont
donc ceux dont la violation serait *invisible* à l'exécution :

1. **Intégrité du corpus** — 225 lignes, une image existante par ligne, aucune perte.
2. **Déterminisme de ``patient_id``** — un identifiant de groupe qui varierait d'un
   appel à l'autre invaliderait toute comparaison entre plis et toute reproduction
   d'un résultat à partir d'une graine.
3. **Détection des collisions d'identité** — chaque indice implémenté est déclenché par
   un cas construit pour lui, et un corpus propre n'en déclenche aucun. Sans ce test,
   l'audit pourrait devenir silencieux sans que rien ne le signale.
4. **Typage entier *nullable*** — une cellule vide ne doit pas transformer ``devita``
   de ``3`` en ``3.0``. Un grade flottant casse toute comparaison d'étiquette, et le
   fait en silence.
5. **Anomalie de domaine signalée, non bloquante** — une valeur hors domaine est un
   fait sur le corpus ; elle doit remonter sans interrompre le pipeline.

Les tests 3 à 5 s'exécutent sur des corpus **factices**, construits en mémoire ou dans
un dossier temporaire : ils restent valides si le jeu de données réel est absent, et ils
testent des cas que le corpus réel ne contient pas nécessairement.

Exécution
---------
    python tests/test_metadata.py     # exécuteur de repli intégré, sans dépendance
    pytest tests/test_metadata.py     # si pytest est disponible

Le fichier suit les conventions pytest (fonctions ``test_*``, ``assert`` nu). Un
exécuteur minimal est fourni en fin de fichier afin que ces garanties soient
vérifiables dès maintenant : ``pytest`` n'est pas dans la pile déclarée de la phase 1,
et l'ajouter serait une décision d'outillage à prendre séparément.
"""

from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

import pandas as pd

# Le dépôt n'est pas installé comme paquet : on ajoute sa racine au chemin d'import.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from sjogren.data.metadata import (  # noqa: E402
    CANONICAL_COLUMNS,
    DEFAULT_DATASET_DIR,
    DEFAULT_PATIENT_KEY_FIELDS,
    MAX_PLAUSIBLE_IMAGES_PER_GLAND,
    MAX_PLAUSIBLE_IMAGES_PER_PATIENT,
    MetadataWarning,
    audit_patient_identity,
    build_patient_id,
    describe_corpus,
    load_metadata,
)

N_IMAGES_EXPECTED = 225
GRADE_COUNTS_EXPECTED = {0: 110, 1: 28, 2: 55, 3: 32}


class Skipped(Exception):
    """Test non exécutable dans cet environnement (jeu de données absent)."""


def _require_real_corpus() -> Path:
    """Ignore le test si le corpus réel n'est pas présent à côté du dépôt."""
    if not (DEFAULT_DATASET_DIR / "Anonymized images - Info.csv").is_file():
        raise Skipped(f"corpus absent : {DEFAULT_DATASET_DIR}")
    return DEFAULT_DATASET_DIR


# ======================================================================================
# Fabrique de corpus factices
# ======================================================================================

#: En-têtes bruts, dans l'ordre du CSV réel. Le dernier porte volontairement l'espace de
#: fin observé dans le fichier d'origine : c'est lui que la normalisation doit absorber.
RAW_HEADER = [
    "Center",
    "Anonymized ID",
    "parotid/submandibular",
    "machine",
    "De Vita et al. score",
    "OMERACT score",
    "Sex",
    "Age at US evaluation",
    "disease duration",
    "fulfillment of 2016 ACR-EULAR classification criteria for pSS ",
]

_DEFAULTS = {
    "center": "Testville (Nowhere)",
    "machine": "testscan",
    "omeract": 0,
    "sex": "F",
    "age": 60,
    "disease_duration": 5,
    "acr_eular_2016": 1,
}


def _row(**overrides) -> dict:
    """Une ligne de corpus factice, champs non précisés remplis par défaut."""
    row = dict(_DEFAULTS)
    row.update(overrides)
    return row


def make_frame(rows: list[dict], key_fields=DEFAULT_PATIENT_KEY_FIELDS) -> pd.DataFrame:
    """Construit en mémoire une table au format canonique, avec ``patient_id``.

    Court-circuite volontairement la lecture de fichier : les tests d'audit portent sur
    la logique de détection, pas sur l'analyse syntaxique du CSV.
    """
    df = pd.DataFrame(rows)
    for col in ("image_id", "devita", "omeract", "age", "disease_duration", "acr_eular_2016"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
    for col in ("center", "gland", "machine", "sex"):
        if col in df.columns:
            df[col] = df[col].astype("string")
    df["patient_id"] = build_patient_id(df, key_fields)
    return df


def write_fake_csv(directory: Path, rows: list[dict], *, with_images: bool = True) -> Path:
    """Écrit un CSV factice fidèle au format réel et crée les images correspondantes.

    Reproduit trois particularités du fichier d'origine que le module doit absorber :
    séparateur point-virgule, espace de fin dans le dernier en-tête, et cinq colonnes
    fantômes issues des ``;;;;;`` terminaux.
    """
    directory.mkdir(parents=True, exist_ok=True)
    images_dir = directory / "Anonymized images"
    images_dir.mkdir(exist_ok=True)

    lines = [";".join(RAW_HEADER) + ";;;;;"]
    for row in rows:
        values = [
            row["center"],
            row["image_id"],
            row["gland"],
            row["machine"],
            row["devita"],
            row["omeract"],
            row["sex"],
            row["age"],
            row["disease_duration"],
            row["acr_eular_2016"],
        ]
        lines.append(";".join("" if v is None else str(v) for v in values) + ";;;;;")

    csv_path = directory / "Anonymized images - Info.csv"
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    if with_images:
        for row in rows:
            # Le module ne décode jamais l'image : un fichier vide suffit à prouver
            # l'existence du chemin.
            (images_dir / f"{int(row['image_id']):03d}.jpg").write_bytes(b"")

    return csv_path


# ======================================================================================
# 1. Intégrité du corpus réel
# ======================================================================================


def test_corpus_reel_225_lignes_et_distribution_attendue():
    """Le corpus doit contenir exactement 225 images et la distribution documentée.

    Ce test est un garde-fou contre une modification du CSV en amont : toute la
    méthodologie (déséquilibre des classes, pondération de la perte, stratification)
    est calibrée sur cette distribution.
    """
    _require_real_corpus()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MetadataWarning)
        df = load_metadata()

    assert len(df) == N_IMAGES_EXPECTED, f"{len(df)} lignes au lieu de {N_IMAGES_EXPECTED}"

    counts = {int(k): int(v) for k, v in df["devita"].value_counts().items()}
    assert counts == GRADE_COUNTS_EXPECTED, f"distribution des grades modifiée : {counts}"

    assert df["image_id"].is_unique, "identifiants d'image dupliqués"
    assert df["image_id"].is_monotonic_increasing, "table non triée par image_id"


def test_aucune_image_perdue():
    """Chaque ligne du CSV pointe vers un fichier existant, et réciproquement.

    « Aucune image perdue » est l'un des contrôles de sortie de l'étape 1.1. Il commence
    ici : une ligne sans fichier serait un échantillon fantôme qui ne planterait qu'au
    premier passage du DataLoader, potentiellement après des heures d'entraînement.
    """
    root = _require_real_corpus()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MetadataWarning)
        df = load_metadata()

    missing = [p for p in df["image_path"] if not Path(p).is_file()]
    assert not missing, f"{len(missing)} chemin(s) d'image inexistant(s) : {missing[:5]}"

    on_disk = {p.name for p in (root / "Anonymized images").glob("*.jpg")}
    referenced = set(df["image_filename"])
    assert on_disk == referenced, (
        f"désaccord disque/CSV — orphelines : {sorted(on_disk - referenced)[:5]}, "
        f"manquantes : {sorted(referenced - on_disk)[:5]}"
    )


def test_colonnes_fantomes_ecartees_et_entetes_normalises():
    """Les colonnes ``Unnamed: *`` sont supprimées et les noms canoniques présents."""
    _require_real_corpus()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MetadataWarning)
        df = load_metadata()

    assert not [c for c in df.columns if str(c).startswith("Unnamed")]
    for canonical in CANONICAL_COLUMNS.values():
        assert canonical in df.columns, f"colonne canonique absente : {canonical}"
    # L'en-tête réel porte un espace de fin ; il ne doit pas subsister dans le nom.
    assert all(c == c.strip() for c in df.columns)


def test_confusion_centre_appareil_confirmee():
    """Un seul appareil par centre — contrainte structurelle n°2 du protocole.

    Ce n'est pas un test de correction du code mais un test de *fait sur le corpus* :
    tant qu'il passe, le leave-one-center-out reste équivalent à un leave-one-device-out,
    et l'interprétation prévue à l'étape 1.3 tient.
    """
    _require_real_corpus()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MetadataWarning)
        df = load_metadata()

    per_center = df.groupby("center", observed=True)["machine"].nunique()
    assert (per_center == 1).all(), f"centre multi-appareils détecté : {per_center.to_dict()}"


# ======================================================================================
# 2. Déterminisme de l'identité patient
# ======================================================================================


def test_patient_id_deterministe_entre_deux_appels():
    """Deux chargements successifs produisent exactement les mêmes identifiants.

    Un ``patient_id`` instable rendrait tout découpage irreproductible : même graine,
    plis différents, chiffres différents, sans qu'aucune erreur ne soit levée.
    """
    _require_real_corpus()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MetadataWarning)
        first = load_metadata()
        second = load_metadata()

    assert first["patient_id"].equals(second["patient_id"])
    assert first["patient_id"].nunique() == second["patient_id"].nunique()

    # Et la fonction elle-même, appelée deux fois sur la même table.
    assert build_patient_id(first).equals(build_patient_id(first))


def test_patient_id_ne_scinde_jamais_un_patient():
    """Deux lignes de même signature reçoivent toujours le même identifiant.

    C'est la propriété qui fonde toute la stratégie : la clé peut fusionner des patients
    distincts, elle ne peut pas en scinder un. Le sur-regroupement coûte de la puissance
    statistique ; la scission créerait la fuite.
    """
    rows = [
        _row(image_id=1, gland="parotid", devita=2, age=64, disease_duration=7),
        _row(image_id=2, gland="submandibular", devita=1, age=64, disease_duration=7),
        _row(image_id=3, gland="parotid", devita=0, age=41, disease_duration=7),
    ]
    df = make_frame(rows)

    assert df.loc[0, "patient_id"] == df.loc[1, "patient_id"], (
        "même signature démographique mais identifiants différents — risque de fuite"
    )
    assert df.loc[0, "patient_id"] != df.loc[2, "patient_id"], (
        "âges différents fusionnés à tort"
    )


def test_patient_id_reste_stable_quel_que_soit_lordre_des_lignes():
    """L'identifiant ne dépend que du contenu de la ligne, pas de sa position."""
    rows = [
        _row(image_id=1, gland="parotid", devita=0, age=50, disease_duration=2),
        _row(image_id=2, gland="submandibular", devita=0, age=71, disease_duration=9),
    ]
    direct = make_frame(rows)
    shuffled = make_frame(list(reversed(rows)))

    assert direct.loc[0, "patient_id"] == shuffled.loc[1, "patient_id"]
    assert direct.loc[1, "patient_id"] == shuffled.loc[0, "patient_id"]


def test_champs_de_cle_parametrables():
    """Retirer un champ de la signature ne peut que réduire le nombre de groupes.

    Sert l'étude de sensibilité du découpage prévue pour ``splits.py`` : mesurer ce que
    coûte la dégénérescence de ``disease_duration`` à Ljubljana.
    """
    rows = [
        _row(image_id=1, gland="parotid", devita=0, age=60, disease_duration=1),
        _row(image_id=2, gland="parotid", devita=0, age=60, disease_duration=8),
    ]
    df = make_frame(rows)

    assert df["patient_id"].nunique() == 2
    reduced = build_patient_id(df, ("center", "sex", "age"))
    assert reduced.nunique() == 1, "la clé réduite devrait fusionner ces deux lignes"


# ======================================================================================
# 3. Détection des collisions d'identité
# ======================================================================================


def test_corpus_propre_ne_declenche_aucun_indice():
    """Un examen bilatéral complet — 4 images, IDs contigus — n'est pas suspect.

    Cas de contrôle négatif : sans lui, un auditeur qui signalerait *tout* passerait
    les trois tests positifs sans rien détecter d'utile.
    """
    rows = [
        _row(image_id=1, gland="parotid", devita=2),
        _row(image_id=2, gland="parotid", devita=2),
        _row(image_id=3, gland="submandibular", devita=1),
        _row(image_id=4, gland="submandibular", devita=1),
    ]
    audit = audit_patient_identity(make_frame(rows))

    assert audit.n_patients == 1
    assert audit.n_images == 4
    assert audit.oversized_groups.empty, "4 images restent plausibles (2 glandes x 2 côtés)"
    assert audit.duplicate_gland_groups.empty, "2 images par glande restent plausibles"
    assert audit.non_contiguous_groups.empty
    assert audit.n_suspicious_groups == 0


def test_indice_taille_excessive():
    """Un groupe de plus de 4 images dépasse ce qu'un patient unique peut fournir.

    Note assumée : avec deux types de glande seulement, un groupe de plus de 4 images
    compte nécessairement plus de 2 images d'une même glande. Les deux indices ne sont
    donc pas isolables l'un de l'autre par construction ; on vérifie la présence de
    chacun, pas leur exclusivité.
    """
    rows = [
        _row(image_id=i, gland="parotid" if i % 2 else "submandibular", devita=i % 4)
        for i in range(1, 7)
    ]
    audit = audit_patient_identity(make_frame(rows))

    assert audit.n_patients == 1
    assert len(audit.oversized_groups) == 1
    assert int(audit.oversized_groups.iloc[0]["n_images"]) == 6
    assert audit.group_size_distribution == {6: 1}
    assert audit.n_suspicious_groups == 1


def test_indice_glande_en_surnombre_sans_taille_excessive():
    """Trois parotides pour un même patient : anatomiquement impossible.

    Cas construit pour déclencher l'indice « glande en surnombre » *seul* — 3 images,
    donc sous le seuil de taille, et IDs contigus.
    """
    rows = [
        _row(image_id=1, gland="parotid", devita=3),
        _row(image_id=2, gland="parotid", devita=3),
        _row(image_id=3, gland="parotid", devita=2),
    ]
    audit = audit_patient_identity(make_frame(rows))

    assert audit.oversized_groups.empty, "3 images ne dépassent pas le seuil de taille"
    assert len(audit.duplicate_gland_groups) == 1
    assert audit.non_contiguous_groups.empty
    assert audit.n_suspicious_groups == 1
    assert audit.duplicate_gland_groups.iloc[0]["glands"].count("parotid") == 3


def test_indice_identifiants_non_contigus_seul():
    """Des ``Anonymized ID`` éloignés signalent deux patients de même signature.

    Cas construit pour isoler cet indice : 2 images seulement, glandes différentes,
    mais numéros distants. Indice heuristique, fondé sur une régularité de numérotation
    observée et non documentée — le test verrouille son comportement, pas sa validité
    clinique.
    """
    rows = [
        _row(image_id=6, gland="parotid", devita=0),
        _row(image_id=75, gland="submandibular", devita=0),
    ]
    audit = audit_patient_identity(make_frame(rows))

    assert audit.oversized_groups.empty
    assert audit.duplicate_gland_groups.empty
    assert len(audit.non_contiguous_groups) == 1
    assert audit.non_contiguous_groups.iloc[0]["image_ids"] == [6, 75]
    assert audit.n_suspicious_groups == 1


def test_groupes_suspects_comptes_une_seule_fois():
    """Un groupe cumulant plusieurs indices ne compte que pour un groupe suspect."""
    rows = [
        _row(image_id=1, gland="parotid", devita=3),
        _row(image_id=2, gland="parotid", devita=3),
        _row(image_id=3, gland="parotid", devita=2),
        _row(image_id=4, gland="submandibular", devita=3),
        _row(image_id=40, gland="submandibular", devita=3),
    ]
    audit = audit_patient_identity(make_frame(rows))

    assert len(audit.oversized_groups) == 1
    assert len(audit.duplicate_gland_groups) == 1
    assert len(audit.non_contiguous_groups) == 1
    assert audit.n_suspicious_groups == 1, "le même groupe est compté trois fois"


def test_detection_champ_degenere():
    """Un champ constant sur un centre y réduit la signature d'une dimension.

    Reproduit le cas Ljubljana, où ``disease duration`` vaut 0 pour les 50 images.
    L'âge varie dans les deux centres : il sert de témoin négatif, et vérifie que le
    détecteur ne signale pas indistinctement tous les champs de la signature.
    """
    rows = [
        _row(image_id=1, center="Flatland", gland="parotid", devita=0, age=60, disease_duration=0),
        _row(image_id=2, center="Flatland", gland="parotid", devita=0, age=70, disease_duration=0),
        _row(image_id=3, center="Varyland", gland="parotid", devita=0, age=55, disease_duration=2),
        _row(image_id=4, center="Varyland", gland="parotid", devita=0, age=65, disease_duration=9),
    ]
    audit = audit_patient_identity(make_frame(rows))

    assert "disease_duration" in audit.degenerate_key_fields
    assert audit.degenerate_key_fields["disease_duration"] == ["Flatland"], (
        "seul Flatland a une durée de maladie constante"
    )
    assert "age" not in audit.degenerate_key_fields, "l'âge varie dans les deux centres"


def test_seuils_de_plausibilite_coherents_avec_lanatomie():
    """Les seuils codés correspondent à 2 types de glande x 2 côtés."""
    assert MAX_PLAUSIBLE_IMAGES_PER_GLAND == 2
    assert MAX_PLAUSIBLE_IMAGES_PER_PATIENT == 4
    assert MAX_PLAUSIBLE_IMAGES_PER_PATIENT == 2 * MAX_PLAUSIBLE_IMAGES_PER_GLAND


def test_audit_du_corpus_reel_reste_stable():
    """Les chiffres publiés dans les documents de suivi restent vrais.

    Verrou de non-régression documentaire : ``results.html`` affiche 89 patients et
    12 groupes suspects. Si ce test casse, ce sont les pages de suivi qu'il faut
    corriger — pas le test qu'il faut assouplir.
    """
    _require_real_corpus()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MetadataWarning)
        df = load_metadata()
    audit = df.attrs["patient_identity_audit"]

    assert audit.n_images == N_IMAGES_EXPECTED
    assert audit.n_patients == 89, f"nombre de patients reconstitués modifié : {audit.n_patients}"
    assert audit.n_suspicious_groups == 12
    assert len(audit.oversized_groups) == 3
    assert len(audit.duplicate_gland_groups) == 6
    assert len(audit.non_contiguous_groups) == 9
    assert audit.degenerate_key_fields.get("disease_duration") == ["Ljubljana (Slovenia)"]


# ======================================================================================
# 4. Typage entier nullable
# ======================================================================================


def test_int64_nullable_preserve_les_valeurs_manquantes():
    """Une cellule vide reste ``pd.NA`` et ne convertit pas la colonne en flottant.

    Le bug que ce test empêche est silencieux : en ``int64`` classique, pandas promeut
    la colonne en ``float64`` dès qu'une valeur manque, et ``devita`` passe de ``3`` à
    ``3.0``. Toute comparaison d'étiquette, tout indexage de matrice de confusion et
    toute pondération de perte par classe deviennent alors faux sans lever d'erreur.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rows = [
            _row(image_id=1, gland="parotid", devita=3, omeract=None),
            _row(image_id=2, gland="submandibular", devita=0, omeract=1),
        ]
        write_fake_csv(root, rows)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", MetadataWarning)
            df = load_metadata(root)

        assert str(df["devita"].dtype) == "Int64", f"dtype devita : {df['devita'].dtype}"
        assert str(df["omeract"].dtype) == "Int64"

        assert df.loc[0, "devita"] == 3
        assert not isinstance(df.loc[0, "devita"], float), "grade promu en flottant"
        assert pd.isna(df.loc[0, "omeract"]), "valeur manquante perdue"
        assert df.loc[1, "omeract"] == 1

        # La cible binaire dérivée respecte la même discipline.
        assert str(df["grade_binary"].dtype) == "Int64"
        assert df.loc[0, "grade_binary"] == 1
        assert df.loc[1, "grade_binary"] == 0


def test_grade_binary_suit_le_seuil_clinique():
    """Sain (grade 0) contre atteint (grades 1-3), et ``pd.NA`` propagé."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rows = [
            _row(image_id=i + 1, gland="parotid", devita=g)
            for i, g in enumerate([0, 1, 2, 3])
        ]
        rows.append(_row(image_id=5, gland="parotid", devita=None))
        write_fake_csv(root, rows)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", MetadataWarning)
            df = load_metadata(root)

        assert df.loc[df["devita"] == 0, "grade_binary"].tolist() == [0]
        assert df.loc[df["devita"] > 0, "grade_binary"].tolist() == [1, 1, 1]
        assert pd.isna(df.loc[df["devita"].isna(), "grade_binary"]).all()


# ======================================================================================
# 5. Anomalies de domaine : signalées, non bloquantes
# ======================================================================================


def test_valeur_hors_domaine_signalee_sans_echec_du_chargement():
    """Une valeur hors domaine remonte en avertissement et le chargement aboutit.

    Reproduit l'anomalie réelle du corpus (``acr_eular_2016 = 2``). Interrompre le
    pipeline pour une colonne qui n'est ni cible ni entrée serait disproportionné ;
    la masquer serait malhonnête. Le module fait donc les deux : il charge et il crie.
    """
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rows = [
            _row(image_id=1, gland="parotid", devita=0, acr_eular_2016=2),
            _row(image_id=2, gland="submandibular", devita=1, acr_eular_2016=1),
        ]
        write_fake_csv(root, rows)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            df = load_metadata(root)

        assert len(df) == 2, "le chargement doit aboutir malgré l'anomalie"

        issues = df.attrs["validation_issues"]
        assert any("acr_eular_2016" in i for i in issues), f"anomalie non consignée : {issues}"

        messages = [str(w.message) for w in caught if issubclass(w.category, MetadataWarning)]
        assert any("acr_eular_2016" in m for m in messages), "aucun avertissement émis"

        # La valeur fautive est conservée telle quelle, pas corrigée en douce.
        assert df.loc[0, "acr_eular_2016"] == 2


def test_grade_hors_domaine_signale():
    """Un grade De Vita hors 0-3 est détecté — la cible est la colonne critique."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rows = [
            _row(image_id=1, gland="parotid", devita=4),
            _row(image_id=2, gland="parotid", devita=0),
        ]
        write_fake_csv(root, rows)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", MetadataWarning)
            df = load_metadata(root)

        assert any("devita" in i for i in df.attrs["validation_issues"])


def test_image_absente_signalee_sans_echec():
    """Une ligne sans fichier image déclenche un avertissement, pas une exception."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        rows = [
            _row(image_id=1, gland="parotid", devita=0),
            _row(image_id=2, gland="submandibular", devita=1),
        ]
        write_fake_csv(root, rows)
        (root / "Anonymized images" / "002.jpg").unlink()

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            df = load_metadata(root)

        assert len(df) == 2
        messages = [str(w.message) for w in caught]
        assert any("absente" in m for m in messages), f"image manquante non signalée : {messages}"


def test_csv_introuvable_leve_une_erreur_explicite():
    """Un chemin erroné doit échouer immédiatement et lisiblement.

    Contre-exemple utile : toutes les anomalies ne sont pas des avertissements. Un CSV
    absent n'est pas un fait sur le corpus, c'est une erreur de configuration, et elle
    doit interrompre plutôt que produire une table vide.
    """
    with tempfile.TemporaryDirectory() as tmp:
        try:
            load_metadata(Path(tmp) / "inexistant")
        except FileNotFoundError as exc:
            assert "introuvable" in str(exc).lower()
        else:
            raise AssertionError("FileNotFoundError attendue")


# ======================================================================================
# 6. Statistiques descriptives
# ======================================================================================


def test_describe_corpus_reproduit_la_reference_triviale():
    """Le F1 macro du classifieur trivial doit valoir 0,164 sur le corpus réel.

    Cette valeur est publiée dans ``results.html`` comme référence de comparaison de
    tous les modèles. Elle est analytique : elle se dérive de la distribution
    110/28/55/32, sans apprentissage. Le test verrouille l'accord entre le code et la
    page de résultats.
    """
    _require_real_corpus()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MetadataWarning)
        stats = describe_corpus(load_metadata())

    assert stats["n_images"] == N_IMAGES_EXPECTED
    assert stats["n_patients"] == 89
    assert stats["grade_counts"] == GRADE_COUNTS_EXPECTED
    assert abs(stats["trivial_accuracy"] - 0.489) < 5e-4
    assert abs(stats["trivial_f1_macro"] - 0.164) < 5e-4
    assert stats["grade_binary_counts"] == {0: 110, 1: 115}
    assert all(len(m) == 1 for m in stats["center_machine_pairs"].values())


# ======================================================================================
# Exécuteur de repli — permet de lancer ces tests sans pytest
# ======================================================================================


def _run_standalone() -> int:
    """Exécute tous les ``test_*`` du module et affiche un rapport.

    ``pytest`` ne figure pas dans la pile déclarée de la phase 1. Plutôt que d'ajouter
    une dépendance d'outillage sans arbitrage, le fichier reste compatible pytest *et*
    exécutable seul. Le jour où pytest est adopté, ce bloc devient inerte.
    """
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]

    passed, skipped, failures = 0, [], []
    for name, fn in tests:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", MetadataWarning)
                fn()
        except Skipped as exc:
            skipped.append((name, str(exc)))
            print(f"  SKIP  {name} — {exc}")
        except AssertionError as exc:
            failures.append((name, str(exc) or "assertion nue"))
            print(f"  ECHEC {name} — {exc}")
        except Exception as exc:  # noqa: BLE001 - un test ne doit jamais planter
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  ERREUR {name} — {type(exc).__name__}: {exc}")
        else:
            passed += 1
            print(f"  ok    {name}")

    print("-" * 70)
    print(f"{passed} réussi(s) · {len(skipped)} ignoré(s) · {len(failures)} échec(s)")
    if failures:
        print("\nDétail des échecs :")
        for name, msg in failures:
            print(f"  - {name}: {msg}")
    return 1 if failures else 0


if __name__ == "__main__":  # pragma: no cover
    print("Tests d'intégrité de sjogren.data.metadata")
    print("-" * 70)
    raise SystemExit(_run_standalone())
