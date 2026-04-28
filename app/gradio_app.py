"""Single Gradio app: 3 tabs (Training Lab / Inference / A/B Test).

Action surface only — observation is in MLflow UI (port 5000).
"""

import gradio as gr
import mlflow

from mlops_churn import config, train


def _params_for_algo(algo: str) -> dict:
    return config.HYPERPARAM_SPEC[algo]


def _slider_components(algo: str):
    """Build sliders for a given algo. Returns dict {param_name: gr.Slider}."""
    spec = _params_for_algo(algo)
    sliders = {}
    for param_name, p in spec.items():
        sliders[param_name] = gr.Slider(
            minimum=p["min"],
            maximum=p["max"],
            value=p["default"],
            label=param_name,
            step=0.01 if isinstance(p["default"], float) else 1,
        )
    return sliders


def build_training_lab() -> None:
    gr.Markdown("## 🎛️ Training Lab — atur knob → train → MLflow merekam")

    algo = gr.Dropdown(
        choices=list(config.HYPERPARAM_SPEC.keys()),
        value="random_forest",
        label="Algoritma",
    )

    col_groups = {}
    slider_refs = {}
    for a in config.HYPERPARAM_SPEC:
        with gr.Column(visible=(a == "random_forest")) as col:
            slider_refs[a] = _slider_components(a)
        col_groups[a] = col

    tag = gr.Textbox(label="Tag opsional", placeholder="demo-pm-april")  # noqa: F841
    btn = gr.Button("🚀 Train & Log to MLflow", variant="primary")

    status = gr.Markdown("", visible=False)
    metrics_md = gr.Markdown("", visible=False)
    run_link = gr.Markdown("", visible=False)

    def _toggle_columns(selected_algo):
        return [gr.update(visible=(a == selected_algo)) for a in config.HYPERPARAM_SPEC]

    algo.change(_toggle_columns, algo, list(col_groups.values()))

    def _run_training(algo_value, *all_params):
        all_specs = list(config.HYPERPARAM_SPEC.items())
        offset = 0
        per_algo = {}
        for a, spec in all_specs:
            per_algo[a] = dict(
                zip(spec.keys(), all_params[offset : offset + len(spec)], strict=True)
            )
            offset += len(spec)
        params = per_algo[algo_value]

        try:
            run_id = train.train_one(algo_value, params, source="gradio-lab")
            metrics = mlflow.get_run(run_id).data.metrics
            metrics_table = (
                "| Metric    | Value |\n|-----------|-------|\n"
                f"| Accuracy  | {metrics['accuracy']:.3f} |\n"
                f"| F1        | {metrics['f1']:.3f} |\n"
                f"| ROC AUC   | {metrics['roc_auc']:.3f} |\n"
                f"| Precision | {metrics['precision']:.3f} |\n"
                f"| Recall    | {metrics['recall']:.3f} |"
            )
            return (
                gr.update(value="✅ Selesai", visible=True),
                gr.update(value=metrics_table, visible=True),
                gr.update(
                    value=(
                        f"🔗 Run ID: `{run_id}` — "
                        f"[Buka di MLflow UI]"
                        f"({config.MLFLOW_TRACKING_URI}/#/experiments/0/runs/{run_id})"
                    ),
                    visible=True,
                ),
            )
        except Exception as e:
            return (
                gr.update(value=f"❌ Error: {e}", visible=True),
                gr.update(value="", visible=False),
                gr.update(value="", visible=False),
            )

    all_sliders = []
    for a in config.HYPERPARAM_SPEC:
        all_sliders.extend(slider_refs[a].values())

    btn.click(
        _run_training,
        inputs=[algo, *all_sliders],
        outputs=[status, metrics_md, run_link],
    )


def build_inference() -> None:
    gr.Markdown("## 🚀 Inference\n_(diisi di Task 20)_")


def build_ab_test() -> None:
    gr.Markdown("## 🔀 A/B Test\n_(diisi di Task 21)_")


def build_app() -> gr.Blocks:
    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    with gr.Blocks(title="BNI Churn MLOps Simulation") as demo:
        gr.Markdown("# 🏦 BNI Churn — MLOps Simulation")
        with gr.Tabs():
            with gr.Tab("🎛️ Training Lab"):
                build_training_lab()
            with gr.Tab("🚀 Inference"):
                build_inference()
            with gr.Tab("🔀 A/B Test"):
                build_ab_test()
        gr.Markdown(f"📊 MLflow UI: [{config.MLFLOW_TRACKING_URI}]({config.MLFLOW_TRACKING_URI})")
    return demo


if __name__ == "__main__":
    build_app().launch(server_port=7860, share=False)
