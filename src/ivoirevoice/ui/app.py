"""Local Gradio dashboard for sequential ASR comparison and evaluation."""

from __future__ import annotations

import html
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

import gradio as gr
import pandas as pd

from ivoirevoice.exceptions import ConfigError
from ivoirevoice.services.comparison_service import ComparisonService
from ivoirevoice.services.evaluation_service import (
    BenchmarkView,
    ErrorSample,
    EvaluationService,
    load_benchmark_view,
    load_error_samples,
)
from ivoirevoice.services.export_service import ExportService
from ivoirevoice.services.transcription_service import (
    ModelCatalog,
    TranscriptionService,
    build_model_registry,
    load_model_catalog,
)
from ivoirevoice.ui.components import (
    ABOUT_MARKDOWN,
    APP_CSS,
    EMPTY_RESULTS,
    benchmark_plot_rows,
    benchmark_rows,
    render_benchmark_context,
    render_comparison_cards,
    render_error_sample,
)
from ivoirevoice.ui.state import UIState, empty_state

REFERENCE_FILE_MAX_BYTES = 256 * 1024
BENCHMARK_HEADERS = [
    "Modèle",
    "Type",
    "Audios",
    "Réussis",
    "WER (%)",
    "CER (%)",
    "RTF",
    "Temps total (s)",
    "Appareil",
    "Date",
    "Seed",
    "Révision",
]
BENCHMARK_DATATYPES: tuple[Literal["str", "number", "bool", "date", "markdown", "html"], ...] = (
    "str",
    "str",
    "number",
    "number",
    "number",
    "number",
    "number",
    "number",
    "str",
    "str",
    "number",
    "str",
)


@dataclass(frozen=True, slots=True)
class DemoServices:
    """All injectable dependencies required by the Gradio surface."""

    catalog: ModelCatalog
    comparison: ComparisonService
    evaluation: EvaluationService
    exports: ExportService
    benchmark: BenchmarkView | None
    benchmark_error: str | None
    error_samples: tuple[ErrorSample, ...]
    error_samples_error: str | None
    dataset_root: Path | None


def _external_root(variable_name: str) -> Path | None:
    raw_value = os.getenv(variable_name)
    return Path(raw_value).expanduser().resolve() if raw_value else None


def build_demo_services(catalog_path: str | Path | None = None) -> DemoServices:
    """Build services without loading model weights or launching a browser."""

    catalog = load_model_catalog(catalog_path)
    transcription = TranscriptionService(catalog, build_model_registry(catalog))
    evaluation = EvaluationService()
    exports = ExportService()
    artifacts_root = _external_root("IVOIREVOICE_ARTIFACTS_DIR")
    dataset_root = _external_root("IVOIREVOICE_DIOULA_DATA_DIR")

    benchmark: BenchmarkView | None = None
    benchmark_error: str | None = None
    error_samples: tuple[ErrorSample, ...] = ()
    error_samples_error: str | None = None
    if artifacts_root is None:
        benchmark_error = "IVOIREVOICE_ARTIFACTS_DIR n'est pas configuré."
        error_samples_error = benchmark_error
    else:
        try:
            benchmark = load_benchmark_view(
                artifacts_root / "reports/baselines/baseline_dy_pilot_comparison.json",
                artifacts_root / "reports/baselines/environment_report.json",
            )
        except ConfigError as exc:
            benchmark_error = str(exc)
        try:
            error_samples = load_error_samples(
                artifacts_root / "baselines/baseline-dy-whisper-tiny-pilot/predictions_private.csv",
                artifacts_root
                / "baselines/baseline-dy-whisper-small-pilot/predictions_private.csv",
            )
        except ConfigError as exc:
            error_samples_error = str(exc)
    if dataset_root is None and error_samples:
        error_samples_error = "IVOIREVOICE_DIOULA_DATA_DIR n'est pas configuré."
        error_samples = ()
    return DemoServices(
        catalog=catalog,
        comparison=ComparisonService(transcription, evaluation),
        evaluation=evaluation,
        exports=exports,
        benchmark=benchmark,
        benchmark_error=benchmark_error,
        error_samples=error_samples,
        error_samples_error=error_samples_error,
        dataset_root=dataset_root,
    )


def _reference_text(text: str | None, file_path: str | None) -> str | None:
    if text and text.strip():
        return text.strip()
    if not file_path:
        return None
    path = Path(file_path)
    if path.suffix.lower() != ".txt":
        raise ConfigError("La référence importée doit être un fichier TXT.")
    try:
        if path.stat().st_size > REFERENCE_FILE_MAX_BYTES:
            raise ConfigError("Le fichier de référence dépasse 256 Kio.")
        content = path.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as exc:
        raise ConfigError("Le fichier de référence est illisible en UTF-8.") from exc
    return content or None


def _run_comparison(
    services: DemoServices,
    audio_path: str | None,
    language: str,
    model_keys: list[str] | None,
    reference_text: str | None,
    reference_file: str | None,
) -> tuple[str, str | None, str | None, str | None, UIState, str]:
    if not audio_path:
        return (
            "<p class='iv-error'>Ajoutez ou enregistrez un audio.</p>",
            None,
            None,
            None,
            empty_state(),
            "Aucune inférence lancée.",
        )
    try:
        reference = _reference_text(reference_text, reference_file)
        run = services.comparison.compare(
            audio_path=audio_path,
            language=language,
            model_keys=tuple(model_keys or ()),
            reference=reference,
        )
        exports = services.exports.export_all(run)
    except (ConfigError, ValueError) as exc:
        return (
            f"<p class='iv-error'>{html.escape(str(exc))}</p>",
            None,
            None,
            None,
            empty_state(),
            "La comparaison a été refusée avant traitement.",
        )
    successful = sum(result.success for result in run.results)
    status = f"{successful}/{len(run.results)} modèle(s) exécuté(s) avec succès."
    return (
        render_comparison_cards(run),
        exports[0],
        exports[1],
        exports[2],
        UIState(run=run, export_paths=exports),
        status,
    )


def _select_error_sample(
    services: DemoServices,
    audio_id: str | None,
) -> tuple[str | None, str]:
    if not audio_id or services.dataset_root is None:
        return None, render_benchmark_context(
            None,
            services.error_samples_error or "Sélectionnez un échantillon.",
        )
    sample = next((item for item in services.error_samples if item.audio_id == audio_id), None)
    if sample is None:
        return None, render_benchmark_context(None, "Échantillon anonymisé inconnu.")
    relative = PurePosixPath(sample.relative_audio_path)
    if relative.is_absolute() or ".." in relative.parts:
        return None, render_benchmark_context(None, "Référence audio privée invalide.")
    source = services.dataset_root.joinpath(*relative.parts)
    try:
        preview = services.exports.prepare_audio_preview(source, sample.audio_id)
    except ConfigError as exc:
        return None, render_benchmark_context(None, str(exc))
    return preview, render_error_sample(sample, services.evaluation)


def _reset(
    services: DemoServices,
    default_models: list[str],
) -> tuple[None, str, list[str], str, None, str, None, None, None, UIState, str]:
    services.exports.cleanup()
    return (
        None,
        "dyu",
        default_models,
        "",
        None,
        EMPTY_RESULTS,
        None,
        None,
        None,
        empty_state(),
        "Interface réinitialisée.",
    )


def _benchmark_components(
    interface_services: DemoServices,
) -> tuple[list[list[Any]], Any, str]:
    return (
        benchmark_rows(interface_services.benchmark),
        pd.DataFrame(
            benchmark_plot_rows(interface_services.benchmark),
            columns=["Modèle", "WER (%)", "CER (%)", "RTF"],
        ),
        render_benchmark_context(
            interface_services.benchmark,
            interface_services.benchmark_error,
        ),
    )


def create_interface(services: DemoServices | None = None) -> Any:
    """Create the five-tab dashboard without loading any model."""

    selected_services = services or build_demo_services()
    enabled_models = selected_services.catalog.enabled_models
    model_choices = [(model.display_name, model.key) for model in enabled_models]
    default_models = [model.key for model in enabled_models]
    benchmark_table_rows, benchmark_charts, benchmark_context = _benchmark_components(
        selected_services
    )
    error_choices = [sample.audio_id for sample in selected_services.error_samples]

    with gr.Blocks(
        title="IvoireVoice AI — Comparateur ASR",
        css=APP_CSS,
        theme=gr.themes.Soft(primary_hue="green", secondary_hue="amber"),
    ) as interface:
        session_state = gr.State(empty_state())
        gr.HTML(
            """
            <section class="iv-hero">
              <div class="iv-kicker">Démonstrateur local · Français · Dioula</div>
              <h1>IvoireVoice AI</h1>
              <p>Comparez des modèles ASR réels, mesurez leurs erreurs et
              explorez les limites de la transcription dioula — sans envoyer
              l'audio vers une API externe.</p>
            </section>
            """
        )

        with gr.Tab("1 · Transcrire et comparer"):
            with gr.Row():
                with gr.Column(scale=2):
                    audio = gr.Audio(
                        sources=["upload", "microphone"],
                        type="filepath",
                        format="wav",
                        label="Audio — WAV, MP3, M4A ou OGG",
                    )
                    language = gr.Dropdown(
                        choices=[("Dioula", "dyu"), ("Français", "fr")],
                        value="dyu",
                        label="Langue",
                    )
                    models = gr.CheckboxGroup(
                        choices=model_choices,
                        value=default_models,
                        label="Modèles à comparer",
                    )
                with gr.Column(scale=2):
                    reference = gr.Textbox(
                        label="Référence facultative",
                        lines=4,
                        placeholder="Sans référence, le WER et le CER restent indisponibles.",
                    )
                    reference_file = gr.File(
                        label="Ou importer une référence TXT",
                        file_types=[".txt"],
                        type="filepath",
                    )
                    with gr.Row():
                        run_button = gr.Button("Comparer les modèles", variant="primary")
                        cancel_button = gr.Button("Annuler", variant="stop")
                        reset_button = gr.Button("Réinitialiser")
                    run_status = gr.Markdown("Prêt. Aucun modèle n'est chargé.")
            comparison_output = gr.HTML(EMPTY_RESULTS)
            gr.Markdown("### Télécharger les résultats structurés")
            with gr.Row():
                json_download = gr.DownloadButton("JSON")
                csv_download = gr.DownloadButton("CSV")
                txt_download = gr.DownloadButton("TXT")

            run_event = cast(Any, run_button).click(
                fn=lambda audio_path, lang, selected, text, file_path: _run_comparison(
                    selected_services,
                    audio_path,
                    lang,
                    selected,
                    text,
                    file_path,
                ),
                inputs=[audio, language, models, reference, reference_file],
                outputs=[
                    comparison_output,
                    json_download,
                    csv_download,
                    txt_download,
                    session_state,
                    run_status,
                ],
                concurrency_id="asr_models",
                concurrency_limit=1,
            )
            cast(Any, cancel_button).click(fn=None, cancels=[run_event])
            cast(Any, reset_button).click(
                fn=lambda: _reset(selected_services, default_models),
                outputs=[
                    audio,
                    language,
                    models,
                    reference,
                    reference_file,
                    comparison_output,
                    json_download,
                    csv_download,
                    txt_download,
                    session_state,
                    run_status,
                ],
                queue=False,
            )

        with gr.Tab("2 · Benchmark"):
            gr.HTML(benchmark_context)
            gr.Dataframe(
                value=benchmark_table_rows,
                headers=BENCHMARK_HEADERS,
                datatype=BENCHMARK_DATATYPES,
                interactive=False,
                wrap=True,
                label="Résultats structurés du pilote",
            )
            gr.Markdown(
                "**Lecture : plus le WER, le CER et le RTF sont faibles, "
                "meilleur est le résultat.**"
            )
            with gr.Row():
                gr.BarPlot(
                    value=benchmark_charts,
                    x="Modèle",
                    y="WER (%)",
                    title="WER par modèle — plus faible = meilleur",
                    y_title="WER (%)",
                    height=300,
                )
                gr.BarPlot(
                    value=benchmark_charts,
                    x="Modèle",
                    y="CER (%)",
                    title="CER par modèle — plus faible = meilleur",
                    y_title="CER (%)",
                    height=300,
                )
                gr.BarPlot(
                    value=benchmark_charts,
                    x="Modèle",
                    y="RTF",
                    title="RTF par modèle — plus faible = meilleur",
                    y_title="RTF",
                    height=300,
                )

        with gr.Tab("3 · Analyse des erreurs"):
            gr.Markdown(
                "Les textes de cet onglet proviennent des artefacts privés locaux et "
                "ne doivent pas être publiés."
            )
            error_selector = gr.Dropdown(
                choices=error_choices,
                value=error_choices[0] if error_choices else None,
                label="Identifiant audio anonymisé",
            )
            error_audio = gr.Audio(label="Aperçu privé", interactive=False)
            error_output = gr.HTML(
                render_benchmark_context(
                    None,
                    selected_services.error_samples_error
                    or "Sélectionnez un échantillon puis chargez l'analyse.",
                )
            )
            error_button = gr.Button("Charger l'analyse", variant="primary")
            cast(Any, error_button).click(
                fn=lambda audio_id: _select_error_sample(selected_services, audio_id),
                inputs=[error_selector],
                outputs=[error_audio, error_output],
                queue=False,
            )

        with gr.Tab("4 · Évaluation personnalisée"):
            gr.Markdown(
                "MVP unitaire : un audio, une référence facultative et plusieurs "
                "modèles. Le futur traitement par lot réutilisera les mêmes services."
            )
            with gr.Row():
                custom_audio = gr.Audio(
                    sources=["upload", "microphone"],
                    type="filepath",
                    format="wav",
                    label="Audio personnalisé",
                )
                with gr.Column():
                    custom_reference = gr.Textbox(label="Référence", lines=4)
                    custom_reference_file = gr.File(
                        label="Référence TXT",
                        file_types=[".txt"],
                        type="filepath",
                    )
                    custom_language = gr.Dropdown(
                        choices=[("Dioula", "dyu"), ("Français", "fr")],
                        value="dyu",
                        label="Langue",
                    )
                    custom_models = gr.CheckboxGroup(
                        choices=model_choices,
                        value=default_models,
                        label="Modèles",
                    )
            custom_button = gr.Button("Évaluer et exporter", variant="primary")
            custom_output = gr.HTML(EMPTY_RESULTS)
            with gr.Row():
                custom_json = gr.DownloadButton("JSON")
                custom_csv = gr.DownloadButton("CSV")
                custom_txt = gr.DownloadButton("TXT")
            custom_status = gr.Markdown("")
            cast(Any, custom_button).click(
                fn=lambda audio_path, lang, selected, text, file_path: _run_comparison(
                    selected_services,
                    audio_path,
                    lang,
                    selected,
                    text,
                    file_path,
                ),
                inputs=[
                    custom_audio,
                    custom_language,
                    custom_models,
                    custom_reference,
                    custom_reference_file,
                ],
                outputs=[
                    custom_output,
                    custom_json,
                    custom_csv,
                    custom_txt,
                    session_state,
                    custom_status,
                ],
                concurrency_id="asr_models",
                concurrency_limit=1,
            )

        with gr.Tab("5 · À propos"):
            gr.Markdown(ABOUT_MARKDOWN)

    return interface.queue(default_concurrency_limit=1, max_size=8)


def main() -> None:
    """Launch a local-only server; public Gradio sharing stays disabled."""

    services = build_demo_services(os.getenv("IVOIREVOICE_UI_MODEL_CATALOG"))
    max_size_mb = max(1, services.catalog.max_audio_size_bytes // (1024 * 1024))
    create_interface(services).launch(
        server_name=os.getenv("IVOIREVOICE_UI_HOST", "127.0.0.1"),
        server_port=int(os.getenv("IVOIREVOICE_UI_PORT", "7860")),
        share=False,
        show_api=False,
        show_error=False,
        max_file_size=f"{max_size_mb}mb",
    )


if __name__ == "__main__":
    main()
