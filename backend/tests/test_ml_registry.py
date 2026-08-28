"""Tests de domains/training/services/registry.py — mode expert
hyperparamètres (retour utilisateur direct : "laisser le choix sur les
hyperparamètres, profondeur des arbres etc."). Deux garanties vérifiées :
1) les bornes affichées au frontend (`ModelSpec.tunable_hyperparameters`)
   correspondent EXACTEMENT aux bornes réellement utilisées par la
   recherche Optuna (`ModelSpec.hyperparameter_space`) — jamais un contrôle
   qui affiche une plage différente de celle réellement explorée ;
2) une valeur fixée par l'utilisateur (`overrides`) est bien celle
   utilisée, jamais recherchée par Optuna ni remplacée par un défaut."""
from __future__ import annotations

from typing import Any

import pytest

from domains.training.services.registry import MODEL_REGISTRY


class _RecordingTrial:
    """Trial Optuna factice — capture CE QUI a été demandé (nom, bornes,
    type) sans jamais vraiment échantillonner, pour comparer ensuite à
    `tunable_hyperparameters` (garde-fou anti-dérive) sans dépendre d'un
    vrai run Optuna."""

    def __init__(self) -> None:
        self.calls: dict[str, dict[str, Any]] = {}

    def suggest_int(self, name: str, low: int, high: int, **kwargs: Any) -> int:
        self.calls[name] = {"kind": "int", "low": low, "high": high, "log": bool(kwargs.get("log", False))}
        return low

    def suggest_float(self, name: str, low: float, high: float, **kwargs: Any) -> float:
        self.calls[name] = {"kind": "float", "low": low, "high": high, "log": bool(kwargs.get("log", False))}
        return low

    def suggest_categorical(self, name: str, choices: list[Any]) -> Any:
        self.calls[name] = {"kind": "categorical", "choices": tuple(choices)}
        return choices[0]


@pytest.mark.parametrize("model_id", list(MODEL_REGISTRY.keys()))
def test_tunable_hyperparameters_match_the_actual_search_space(model_id):
    """Pour chaque modèle du catalogue : les hyperparamètres déclarés dans
    `tunable_hyperparameters` (consommés par le frontend pour construire les
    contrôles du mode expert) doivent être EXACTEMENT ceux réellement
    demandés à Optuna par `hyperparameter_space`, avec les mêmes bornes —
    sinon l'utilisateur verrait une plage qui ne correspond pas à ce que le
    moteur explore réellement."""
    spec = MODEL_REGISTRY[model_id]
    trial = _RecordingTrial()
    spec.hyperparameter_space(trial)  # sans overrides — capture l'espace complet

    declared_names = {m.name for m in spec.tunable_hyperparameters}
    called_names = set(trial.calls.keys())
    assert declared_names == called_names, (
        f"{model_id} : hyperparamètres déclarés {declared_names} != réellement recherchés {called_names}"
    )

    for meta in spec.tunable_hyperparameters:
        call = trial.calls[meta.name]
        assert meta.kind == call["kind"], f"{model_id}.{meta.name} : kind déclaré {meta.kind} != réel {call['kind']}"
        if meta.kind == "categorical":
            assert meta.choices == call["choices"], f"{model_id}.{meta.name} : choices déclarés != réels"
        else:
            assert meta.low == call["low"], f"{model_id}.{meta.name} : low déclaré {meta.low} != réel {call['low']}"
            assert meta.high == call["high"], f"{model_id}.{meta.name} : high déclaré {meta.high} != réel {call['high']}"
            assert meta.log == call["log"], f"{model_id}.{meta.name} : log déclaré {meta.log} != réel {call['log']}"


@pytest.mark.parametrize("model_id", list(MODEL_REGISTRY.keys()))
def test_every_tunable_hyperparameter_has_a_label_and_help(model_id):
    """Jamais un contrôle affiché sans explication — chaque hyperparamètre
    réglable doit avoir un nom lisible ET un texte d'aide, jamais l'un sans
    l'autre (l'utilisateur cible n'est pas nécessairement data scientist)."""
    for meta in MODEL_REGISTRY[model_id].tunable_hyperparameters:
        assert meta.label.strip() != ""
        assert meta.help.strip() != ""


def test_override_fixes_the_value_without_ever_asking_optuna():
    """Un hyperparamètre fixé par l'utilisateur ne doit JAMAIS déclencher
    `trial.suggest_*` — sinon l'exploration Optuna pourrait s'écarter de la
    valeur choisie (TPE explore autour des valeurs déjà vues, mais rien ne
    garantit qu'il reste sur celle imposée)."""
    spec = MODEL_REGISTRY["random_forest"]
    trial = _RecordingTrial()
    params = spec.hyperparameter_space(trial, {"max_depth": 6})

    assert params["max_depth"] == 6
    assert "max_depth" not in trial.calls  # jamais demandé à Optuna
    # Les hyperparamètres NON fixés restent recherchés normalement.
    assert "n_estimators" in trial.calls
    assert "min_samples_split" in trial.calls


def test_override_applies_to_categorical_hyperparameters_too():
    spec = MODEL_REGISTRY["svm"]
    trial = _RecordingTrial()
    params = spec.hyperparameter_space(trial, {"kernel": "linear"})

    assert params["kernel"] == "linear"
    assert "kernel" not in trial.calls
    assert "C" in trial.calls  # non fixé, toujours recherché


def test_no_overrides_reproduces_historical_behavior():
    """`overrides=None` (comportement historique, avant ce correctif) doit
    produire exactement le même espace de recherche qu'un appel à un seul
    argument — non-régression explicite."""
    spec = MODEL_REGISTRY["lightgbm"]
    trial_a = _RecordingTrial()
    trial_b = _RecordingTrial()
    spec.hyperparameter_space(trial_a)
    spec.hyperparameter_space(trial_b, None)
    assert trial_a.calls == trial_b.calls


def test_empty_overrides_dict_reproduces_historical_behavior():
    spec = MODEL_REGISTRY["xgboost"]
    trial_a = _RecordingTrial()
    trial_b = _RecordingTrial()
    spec.hyperparameter_space(trial_a)
    spec.hyperparameter_space(trial_b, {})
    assert trial_a.calls == trial_b.calls
