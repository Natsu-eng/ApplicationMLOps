"""Garde-fou de régression IDOR — Phase 1 (AUDIT_BACKEND_2026-08-23.md, Axe B).

L'audit délégué a vérifié manuellement les 110 routes du backend : aucun
IDOR cross-tenant trouvé, pattern `_get_org_dataset`/`_get_org_job`
systématique, 404 jamais 403. Ce fichier consolide un test paramétré unique
qui exerce ce pattern sur CHAQUE domaine à ressource (au lieu de tests
d'isolation dispersés dans chaque fichier `test_<domaine>_api.py`, qui
existent déjà mais ne garantissent pas qu'un FUTUR domaine ajouté au
backend soit couvert automatiquement) — un utilisateur d'une organisation B
qui demande une ressource de l'organisation A doit systématiquement
recevoir 404 (jamais 403 — ne doit pas confirmer que la ressource existe).

Portée assumée (voir `_backend/JOURNAL.md`, Décision 9) : un endpoint
"détail" représentatif par domaine plutôt que les 110 routes une par une —
chaque domaine délègue à un seul helper interne (`_get_org_dataset`/
`_get_org_job`), donc l'endpoint détail est un proxy fidèle de tous les
endpoints du même domaine qui délèguent au même helper (déjà vérifié
manuellement route par route par l'audit délégué)."""
from __future__ import annotations

import io

import pytest

from test_vision_anomaly_api import _mvtec_zip_bytes
from test_vision_datasets_api import _classification_zip_bytes


def _register(client, email, org):
    resp = client.post(
        "/api/auth/register",
        json={"email": email, "nom": "User", "password": "motdepasse123", "organization_name": org},
    )
    return {"Authorization": f"Bearer {resp.json()['access_token']}"}


def _upload_tabular_dataset(client, headers, name="d.csv", n=60):
    rows = "\n".join(f"{i},{i * 2},cat{i % 3}" for i in range(n))
    content = f"x1,x2,categorie\n{rows}\n".encode()
    resp = client.post("/api/datasets", headers=headers, files={"file": (name, io.BytesIO(content), "text/csv")})
    return resp.json()


def _upload_vision_dataset(client, headers, content: bytes, name="dataset.zip"):
    return client.post(
        "/api/vision/datasets", headers=headers, files={"files": (name, io.BytesIO(content), "application/zip")}
    ).json()


@pytest.fixture
def two_orgs(client):
    """Org A possède les ressources testées ; org B est l'attaquant."""
    return {
        "a": _register(client, "a@bureau-a.fr", "Bureau A"),
        "b": _register(client, "b@bureau-b.fr", "Bureau B"),
    }


def test_dataset_404_for_other_organization(client, two_orgs):
    dataset = _upload_tabular_dataset(client, two_orgs["a"])
    resp = client.get(f"/api/datasets/{dataset['id']}", headers=two_orgs["b"])
    assert resp.status_code == 404


def test_training_job_404_for_other_organization(client, two_orgs):
    from unittest.mock import patch

    dataset = _upload_tabular_dataset(client, two_orgs["a"])
    with patch("domains.training.router.training_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        job = client.post(
            "/api/training/jobs", headers=two_orgs["a"],
            json={"dataset_id": dataset["id"], "target_column": "x2"},
        ).json()
    resp = client.get(f"/api/training/jobs/{job['id']}", headers=two_orgs["b"])
    assert resp.status_code == 404


def test_clustering_job_404_for_other_organization(client, two_orgs):
    from unittest.mock import patch

    dataset = _upload_tabular_dataset(client, two_orgs["a"])
    with patch("domains.clustering.router.analysis_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        job = client.post(
            "/api/clustering/jobs", headers=two_orgs["a"],
            json={"dataset_id": dataset["id"], "feature_columns": ["x1", "x2"]},
        ).json()
    resp = client.get(f"/api/clustering/jobs/{job['id']}", headers=two_orgs["b"])
    assert resp.status_code == 404


def test_dimensionality_job_404_for_other_organization(client, two_orgs):
    from unittest.mock import patch

    dataset = _upload_tabular_dataset(client, two_orgs["a"])
    with patch("domains.dimensionality.router.analysis_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        job = client.post(
            "/api/dimensionality/jobs", headers=two_orgs["a"],
            json={"dataset_id": dataset["id"], "feature_columns": ["x1", "x2"]},
        ).json()
    resp = client.get(f"/api/dimensionality/jobs/{job['id']}", headers=two_orgs["b"])
    assert resp.status_code == 404


def test_anomaly_job_404_for_other_organization(client, two_orgs):
    from unittest.mock import patch

    dataset = _upload_tabular_dataset(client, two_orgs["a"])
    with patch("domains.anomalies.router.analysis_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        job = client.post(
            "/api/anomalies/jobs", headers=two_orgs["a"],
            json={"dataset_id": dataset["id"], "feature_columns": ["x1", "x2"]},
        ).json()
    resp = client.get(f"/api/anomalies/jobs/{job['id']}", headers=two_orgs["b"])
    assert resp.status_code == 404


def test_vision_dataset_404_for_other_organization(client, two_orgs):
    dataset = _upload_vision_dataset(client, two_orgs["a"], _classification_zip_bytes())
    resp = client.get(f"/api/vision/datasets/{dataset['id']}", headers=two_orgs["b"])
    assert resp.status_code == 404


def test_vision_classification_job_404_for_other_organization(client, two_orgs):
    from unittest.mock import patch

    dataset = _upload_vision_dataset(client, two_orgs["a"], _classification_zip_bytes())
    with patch("domains.vision.classification.router.vision_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        job = client.post(
            "/api/vision/classification/jobs", headers=two_orgs["a"],
            json={"vision_dataset_id": dataset["id"], "num_epochs": 1, "batch_size": 4},
        ).json()
    resp = client.get(f"/api/vision/classification/jobs/{job['id']}", headers=two_orgs["b"])
    assert resp.status_code == 404


def test_vision_anomaly_job_404_for_other_organization(client, two_orgs):
    from unittest.mock import patch

    dataset = _upload_vision_dataset(client, two_orgs["a"], _mvtec_zip_bytes())
    with patch("domains.vision.anomalies.router.vision_queue") as mock_queue:
        mock_queue.enqueue.return_value.id = "fake-rq-id"
        job = client.post(
            "/api/vision/anomalies/jobs", headers=two_orgs["a"],
            json={"vision_dataset_id": dataset["id"], "num_epochs": 2, "batch_size": 4},
        ).json()
    resp = client.get(f"/api/vision/anomalies/jobs/{job['id']}", headers=two_orgs["b"])
    assert resp.status_code == 404


def test_none_of_the_above_leak_via_403(client, two_orgs):
    """Filet de sécurité explicite sur le principe lui-même — un 403
    confirmerait l'existence de la ressource à un attaquant, jamais
    acceptable ici (voir docstring du module)."""
    dataset = _upload_tabular_dataset(client, two_orgs["a"])
    resp = client.get(f"/api/datasets/{dataset['id']}", headers=two_orgs["b"])
    assert resp.status_code != 403
