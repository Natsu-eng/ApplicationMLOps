"""Phase 3 (AUDIT_BACKEND_2026-08-23.md, Axe I) -- avant ce correctif,
`workers/run_worker.py` appelait `logging.basicConfig` avec un format texte
libre, independant de `api/core/observability.py::configure_logging` (JSON,
`request_id` correle) utilise cote API -- les logs du process worker RQ
etaient illisibles par un collecteur JSON et jamais correlables a la
requete HTTP d'origine. Meme technique en sous-processus que
`test_database_startup.py` (le formatage a lieu a l'import du module)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_BACKEND_DIR = Path(__file__).resolve().parent.parent


def test_worker_process_emits_structured_json_logs():
    env = os.environ.copy()
    code = "import workers.run_worker\nimport logging\nlogging.getLogger('datalab.worker').info('sonde de test')\n"
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(_BACKEND_DIR),
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    lines = [line for line in result.stdout.strip().splitlines() if line.strip()]
    assert lines, f"aucune ligne de log sur stdout -- stderr: {result.stderr}"
    payload = json.loads(lines[-1])
    assert payload["message"] == "sonde de test"
    assert payload["logger"] == "datalab.worker"
    assert "request_id" in payload
    assert "level" in payload
