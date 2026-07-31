"""Presentation-only helpers for the IvoireVoice Gradio dashboard."""

from __future__ import annotations

import html
from typing import Any

from ivoirevoice.services.comparison_service import ComparisonRun
from ivoirevoice.services.evaluation_service import (
    BenchmarkView,
    ErrorSample,
    EvaluationService,
)

APP_CSS = """
:root {
  --iv-ink: #17211b;
  --iv-muted: #5f6f64;
  --iv-green: #0f7a4d;
  --iv-green-dark: #075d3a;
  --iv-gold: #e7a928;
  --iv-cream: #f8f5ec;
  --iv-card: #ffffff;
}
.gradio-container { max-width: 1480px !important; background: var(--iv-cream); }
.iv-hero {
  padding: 1.6rem 1.8rem;
  border-radius: 22px;
  color: white;
  background: linear-gradient(118deg, #075d3a 0%, #0f7a4d 58%, #d89a1d 140%);
  box-shadow: 0 18px 45px rgba(7, 93, 58, .18);
  margin-bottom: 1rem;
}
.iv-hero h1 { margin: 0 0 .4rem 0; font-size: 2rem; }
.iv-hero p { margin: 0; max-width: 850px; opacity: .92; }
.iv-kicker { text-transform: uppercase; letter-spacing: .12em; font-size: .72rem; }
.iv-result-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(310px, 1fr));
  gap: 1rem;
  margin-top: .5rem;
}
.iv-model-card {
  background: var(--iv-card);
  border: 1px solid rgba(15, 122, 77, .16);
  border-top: 5px solid var(--iv-green);
  border-radius: 18px;
  padding: 1rem 1.1rem;
  box-shadow: 0 8px 24px rgba(23, 33, 27, .07);
}
.iv-model-card.failed { border-top-color: #b53a3a; }
.iv-verdict-improved { border-left: 5px solid var(--iv-green); }
.iv-verdict-unchanged { border-left: 5px solid var(--iv-gold); }
.iv-verdict-degraded { border-left: 5px solid #b53a3a; }
.iv-model-card h3 { margin: 0 0 .25rem 0; color: var(--iv-ink); }
.iv-badge {
  display: inline-block; padding: .18rem .55rem; border-radius: 999px;
  color: var(--iv-green-dark); background: #e4f4eb; font-size: .76rem;
}
.iv-transcript {
  background: #f5f7f5; padding: .8rem; border-radius: 10px;
  min-height: 70px; white-space: pre-wrap; overflow-wrap: anywhere;
}
.iv-metrics { display: grid; grid-template-columns: 1fr 1fr; gap: .4rem .8rem; }
.iv-metric { border-bottom: 1px solid #edf0ed; padding: .35rem 0; }
.iv-metric span { color: var(--iv-muted); display: block; font-size: .75rem; }
.iv-private {
  border-left: 4px solid var(--iv-gold); padding: .7rem .9rem;
  background: #fff7dd; border-radius: 8px; margin-bottom: .8rem;
}
.diff-equal { color: #1f513b; }
.diff-delete { background: #ffe1df; color: #8c2525; text-decoration: line-through; }
.diff-insert { background: #daf4e5; color: #075d3a; text-decoration: none; }
.iv-error-block { padding: .8rem; background: white; border-radius: 12px; margin: .6rem 0; }
.iv-error { color: #8c2525; font-weight: 600; }
"""

EMPTY_RESULTS = """
<div class="iv-model-card">
  <h3>Prêt pour la comparaison</h3>
  <p>Ajoutez un audio, choisissez un ou plusieurs modèles, puis lancez
  l'inférence. Les modèles seront chargés et libérés séquentiellement.</p>
</div>
"""


def _metric(label: str, value: str) -> str:
    return f"<div class='iv-metric'><span>{html.escape(label)}</span>{html.escape(value)}</div>"


def _rate(value: float | None) -> str:
    return f"{value * 100:.2f} %" if value is not None else "Non disponible"


def _optional(value: object) -> str:
    return "Non disponible" if value is None else str(value)


def _status_label(status: str) -> str:
    return {
        "baseline": "modèle baseline",
        "adapted": "modèle adapté",
        "pilot_adapted": "modèle pilote adapté",
    }.get(status, status)


def render_comparison_cards(run: ComparisonRun) -> str:
    """Render escaped side-by-side cards without a local path or fake confidence."""

    cards: list[str] = []
    for result in run.results:
        evaluation = result.evaluation
        failure_class = " failed" if not result.success else ""
        transcript = result.transcription or result.error or "Aucune sortie."
        processing = (
            f"{result.processing_time_seconds:.3f} s"
            if result.processing_time_seconds is not None
            else "Non disponible"
        )
        rtf = f"{result.rtf:.4f}" if result.rtf is not None else "Non disponible"
        metrics = "".join(
            (
                _metric("Temps de traitement", processing),
                _metric("Durée audio", f"{result.audio_duration_seconds:.3f} s"),
                _metric("RTF", rtf),
                _metric("Appareil", result.device.upper()),
                _metric("Matériel", result.hardware),
                _metric("WER", _rate(evaluation.wer)),
                _metric("CER", _rate(evaluation.cer)),
                _metric("Révision", result.model_revision),
                _metric("Checkpoint", result.checkpoint_name or "Non applicable"),
                _metric("Tâche", result.task),
                _metric(
                    "Configuration langue",
                    (
                        f"{result.configured_language} — multilingue, sans token forcé"
                        if result.configured_language
                        else "Multilingue, sans token forcé"
                    ),
                ),
                _metric("Audios d'entraînement", _optional(result.training_audio_count)),
                _metric("Audios de validation", _optional(result.validation_audio_count)),
                _metric(
                    "Métriques de référence",
                    evaluation.message,
                ),
            )
        )
        cards.append(
            f"<article class='iv-model-card{failure_class}'>"
            f"<h3>{html.escape(result.display_name)}</h3>"
            f"<span class='iv-badge'>{html.escape(_status_label(result.model_status))}</span>"
            f"<p class='iv-transcript'>{html.escape(transcript)}</p>"
            f"<div class='iv-metrics'>{metrics}</div>"
            "</article>"
        )
    return "<div class='iv-result-grid'>" + "".join(cards) + "</div>"


def render_benchmark_context(view: BenchmarkView | None, error: str | None = None) -> str:
    if view is None:
        message = html.escape(error or "Artefacts de benchmark indisponibles.")
        return f"<div class='iv-model-card failed'><p class='iv-error'>{message}</p></div>"
    return (
        "<div class='iv-model-card'>"
        f"<h3>{html.escape(view.experiment_title)}</h3>"
        f"<p>{html.escape(view.comparison_scope)}</p>"
        "<div class='iv-metrics'>"
        + _metric("Dataset", view.dataset_name)
        + _metric("Split", view.split)
        + _metric("Audios", str(view.audio_count))
        + _metric("Locuteurs", str(view.speaker_count))
        + _metric("Seed", str(view.seed))
        + _metric("Matériel", view.hardware)
        + _metric("Date du rapport", view.run_date)
        + "</div>"
        f"<p><strong>Normalisation :</strong> {html.escape(view.normalization)}</p>"
        "</div>"
    )


def render_error_sample(sample: ErrorSample, evaluation_service: EvaluationService) -> str:
    baseline_evaluation = evaluation_service.evaluate(
        sample.reference,
        sample.predictions.get("Whisper Tiny — baseline", ""),
    )
    adapted_evaluation = evaluation_service.evaluate(
        sample.reference,
        sample.predictions.get("Whisper Tiny Dioula — adapté pilote", ""),
    )
    baseline_wer = baseline_evaluation.wer
    adapted_wer = adapted_evaluation.wer
    if baseline_wer is None or adapted_wer is None:
        verdict = "Comparaison indisponible."
        verdict_class = "unchanged"
    elif adapted_wer < baseline_wer:
        verdict = "L’adaptation améliore cet échantillon."
        verdict_class = "improved"
    elif adapted_wer > baseline_wer:
        verdict = "L’adaptation dégrade cet échantillon."
        verdict_class = "degraded"
    else:
        verdict = "Le résultat est inchangé sur cet échantillon."
        verdict_class = "unchanged"
    blocks = [
        "<div class='iv-private'>Analyse locale privée : ne pas capturer ni publier "
        "les transcriptions sans autorisation.</div>",
        f"<div class='iv-model-card'><h3>Audio anonymisé {html.escape(sample.audio_id)}</h3>",
        f"<p><strong>Référence :</strong> {html.escape(sample.reference)}</p></div>",
        f"<div class='iv-model-card iv-verdict-{verdict_class}'>"
        f"<strong>{html.escape(verdict)}</strong></div>",
    ]
    for model_name, prediction in sample.predictions.items():
        evaluation = evaluation_service.evaluate(sample.reference, prediction)
        diff = evaluation_service.render_word_diff(
            evaluation.reference_normalized or "",
            evaluation.prediction_normalized or "",
        )
        blocks.append(
            "<div class='iv-error-block'>"
            f"<h4>{html.escape(model_name)}</h4>"
            f"<p><strong>Hypothèse :</strong> {html.escape(prediction)}</p>"
            "<div class='iv-metrics'>"
            + _metric("WER individuel", _rate(evaluation.wer))
            + _metric("CER individuel", _rate(evaluation.cer))
            + _metric("Substitutions", str(evaluation.substitutions))
            + _metric("Insertions", str(evaluation.insertions))
            + _metric("Suppressions", str(evaluation.deletions))
            + "</div>"
            f"<p><strong>Différences :</strong> {diff}</p>"
            "</div>"
        )
    return "".join(blocks)


def benchmark_rows(view: BenchmarkView | None) -> list[list[Any]]:
    if view is None:
        return []
    return [
        [
            row["model"],
            row["type"],
            row["audios"],
            row["successes"],
            row["wer_percent"],
            row["cer_percent"],
            row["rtf"],
            row["processing_time_seconds"],
            row["device"],
            row["date"],
            row["seed"],
            row["revision"],
            row["validation_loss"],
            row["substitutions"],
            row["insertions"],
            row["deletions"],
            row["wer_absolute_reduction_points"],
            row["wer_relative_reduction_percent"],
            row["cer_absolute_reduction_points"],
            row["cer_relative_reduction_percent"],
        ]
        for row in view.rows
    ]


def benchmark_plot_rows(view: BenchmarkView | None) -> list[dict[str, Any]]:
    if view is None:
        return []
    return [
        {
            "Modèle": str(row["model"]).replace("openai/whisper-", "Whisper "),
            "WER (%)": row["wer_percent"],
            "CER (%)": row["cer_percent"],
            "RTF": row["rtf"],
        }
        for row in view.rows
    ]


ABOUT_MARKDOWN = """
## Un ASR pensé pour le contexte ivoirien

IvoireVoice AI étudie la transcription du **français** et du **dioula**,
avec une architecture extensible au baoulé. Le corpus dioula local v0.1
contient 19 199 audios uniques provenant de 21 groupes de locuteurs, séparés
strictement entre entraînement, validation et test.

### Architecture

`Gradio → ComparisonService → ModelRegistry → ASRBackend → EvaluationService → ExportService`

Les modèles disponibles sont Whisper Tiny et Whisper Small en baseline, ainsi
que Whisper Tiny Dioula adapté pilote. Ils sont chargés à la demande, un par
un, puis libérés. Le checkpoint pilote reste hors de Git et son emplacement
provient exclusivement d'une variable d'environnement.

### Lire les métriques

- **WER** : erreurs au niveau des mots ;
- **CER** : erreurs au niveau des caractères ;
- **RTF** : temps de calcul divisé par la durée audio. Un RTF inférieur à 1
  indique une inférence plus rapide que le temps réel.

Plus le WER, le CER et le RTF sont faibles, meilleur est le résultat. Whisper
ne fournit pas ici de score de confiance calibré.

### Limites et éthique

Le modèle adapté reste un pilote et n'est pas le modèle final. Les données,
références, prédictions et modèles dérivés restent strictement locaux, car la
licence et le consentement de redistribution ne sont pas confirmés. Le
`final_holdout` n'a pas été évalué. Le baoulé constitue une perspective, pas
une capacité actuelle. Cette interface ne doit pas être exposée publiquement
avec les artefacts privés.

Projet : [github.com/Inza01/ivoirevoice-ai](https://github.com/Inza01/ivoirevoice-ai)
"""
