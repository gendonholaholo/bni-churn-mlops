"""Single Gradio app: 3 tabs (Training Lab / Inference / A/B Test).

Action surface only — observation is in MLflow UI (port 5000).
"""

import gradio as gr
import mlflow

from mlops_churn import config, registry, serving, train


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
                        f"[Buka di MLflow UI]("
                        f"{config.MLFLOW_TRACKING_URI}/#/experiments/"
                        f"{mlflow.get_run(run_id).info.experiment_id}/runs/{run_id})"
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


def render_customer_form() -> dict:
    """Render the 10-field customer form. Returns dict of components."""
    components = {}
    schema = config.CUSTOMER_FEATURE_SCHEMA
    with gr.Row():
        with gr.Column():
            gr.Markdown("**Demografi**")
            components["CreditScore"] = gr.Slider(
                schema["CreditScore"]["min"],
                schema["CreditScore"]["max"],
                value=schema["CreditScore"]["default"],
                step=1,
                label="CreditScore",
            )
            components["Age"] = gr.Slider(
                schema["Age"]["min"],
                schema["Age"]["max"],
                value=schema["Age"]["default"],
                step=1,
                label="Age",
            )
            components["Tenure"] = gr.Slider(
                schema["Tenure"]["min"],
                schema["Tenure"]["max"],
                value=schema["Tenure"]["default"],
                step=1,
                label="Tenure (years)",
            )
            components["Geography"] = gr.Dropdown(
                choices=schema["Geography"]["choices"],
                value=schema["Geography"]["default"],
                label="Geography",
            )
            components["Gender"] = gr.Radio(
                choices=schema["Gender"]["choices"],
                value=schema["Gender"]["default"],
                label="Gender",
            )
        with gr.Column():
            gr.Markdown("**Akun**")
            components["Balance"] = gr.Number(
                value=schema["Balance"]["default"],
                label="Balance",
                minimum=schema["Balance"]["min"],
                maximum=schema["Balance"]["max"],
            )
            components["NumOfProducts"] = gr.Slider(
                schema["NumOfProducts"]["min"],
                schema["NumOfProducts"]["max"],
                value=schema["NumOfProducts"]["default"],
                step=1,
                label="NumOfProducts",
            )
            components["HasCrCard"] = gr.Radio(
                choices=[("Yes", 1), ("No", 0)],
                value=schema["HasCrCard"]["default"],
                label="HasCrCard",
            )
            components["IsActiveMember"] = gr.Radio(
                choices=[("Yes", 1), ("No", 0)],
                value=schema["IsActiveMember"]["default"],
                label="IsActiveMember",
            )
            components["EstimatedSalary"] = gr.Number(
                value=schema["EstimatedSalary"]["default"],
                label="EstimatedSalary",
                minimum=schema["EstimatedSalary"]["min"],
                maximum=schema["EstimatedSalary"]["max"],
            )
    return components


def _features_dict_from_components(component_values: list, keys: list[str]) -> dict:
    return dict(zip(keys, component_values, strict=True))


def build_inference() -> None:
    gr.Markdown("## 🚀 Inference — Prediksi Churn")

    status = gr.Markdown("📦 Model serving: _(klik Refresh)_")
    refresh_btn = gr.Button("🔄 Refresh", size="sm")

    components = render_customer_form()
    feature_keys = list(components.keys())

    predict_btn = gr.Button("🔮 Predict", variant="primary")
    result_label = gr.Markdown("", visible=False)
    result_meta = gr.Markdown("", visible=False)

    def _refresh_status():
        try:
            mv = registry.get_version_by_alias(config.ALIAS_PRODUCTION)
            return (
                f"📦 Model serving: `{config.REGISTERED_MODEL_NAME}` "
                f"**v{mv.version}** (@{config.ALIAS_PRODUCTION})"
            )
        except Exception:
            return (
                "⚠️ Belum ada model `@production`. "
                "Jalankan `uv run python -m scripts.seed_runs` dulu."
            )

    refresh_btn.click(_refresh_status, outputs=status)

    def _do_predict(*values):
        features = _features_dict_from_components(values, feature_keys)
        try:
            out = serving.predict(features, alias=config.ALIAS_PRODUCTION)
            label_text = "🔴 **LIKELY TO CHURN**" if out["label"] == 1 else "🟢 **LIKELY TO STAY**"
            return (
                gr.update(
                    value=f"{label_text} — probability: {out['prob'] * 100:.1f}%",
                    visible=True,
                ),
                gr.update(
                    value=f"Latency: {out['latency_ms']:.1f} ms · Model v{out['version']}",
                    visible=True,
                ),
            )
        except Exception as e:
            return (
                gr.update(value=f"❌ Error: {e}", visible=True),
                gr.update(value="", visible=False),
            )

    predict_btn.click(
        _do_predict, inputs=list(components.values()), outputs=[result_label, result_meta]
    )


def build_ab_test() -> None:
    gr.Markdown("## 🔀 A/B Test — Bandingkan Production vs Staging")

    status = gr.Markdown("📦 _(klik Refresh)_")
    refresh_btn = gr.Button("🔄 Refresh", size="sm")

    sample_choices = ["Custom (isi manual)"] + list(config.AB_TEST_SAMPLES.keys())
    sample_dropdown = gr.Dropdown(
        choices=sample_choices, value="Custom (isi manual)", label="Pilih customer contoh"
    )

    components = render_customer_form()
    feature_keys = list(components.keys())

    compare_btn = gr.Button("🔀 Compare A/B", variant="primary")
    result_md = gr.Markdown("", visible=False)
    agreement_md = gr.Markdown("", visible=False)

    def _refresh_status():
        try:
            prod = registry.get_version_by_alias(config.ALIAS_PRODUCTION).version
        except Exception:
            prod = "—"
        try:
            stag = registry.get_version_by_alias(config.ALIAS_STAGING).version
        except Exception:
            stag = "—"
        return f"🅰️ Production: **v{prod}**  |  🅱️ Staging: **v{stag}**"

    refresh_btn.click(_refresh_status, outputs=status)

    def _load_sample(name: str):
        if name == "Custom (isi manual)":
            return [components[k].value for k in feature_keys]
        sample = config.AB_TEST_SAMPLES[name]
        return [sample[k] for k in feature_keys]

    sample_dropdown.change(_load_sample, sample_dropdown, list(components.values()))

    def _do_compare(*values):
        features = _features_dict_from_components(values, feature_keys)
        try:
            out = serving.predict_ab(features)
            prod = out["production"]
            stag = out["staging"]
            prod_lab = "🔴 CHURN" if prod["label"] == 1 else "🟢 STAY"
            stag_lab = "🔴 CHURN" if stag["label"] == 1 else "🟢 STAY"

            md = (
                f"| | Production v{prod['version']} | Staging v{stag['version']} |\n"
                f"|---|---|---|\n"
                f"| Verdict   | {prod_lab} | {stag_lab} |\n"
                f"| Probability | {prod['prob'] * 100:.1f}% | {stag['prob'] * 100:.1f}% |\n"
                f"| Latency   | {prod['latency_ms']:.1f} ms | {stag['latency_ms']:.1f} ms |\n"
            )
            agreement = (
                "🟢 Kedua model SETUJU"
                if out["agreement"]
                else "🟡 Model BERBEDA pendapat — kandidat case yang menarik"
            )
            return (
                gr.update(value=md, visible=True),
                gr.update(value=agreement, visible=True),
            )
        except Exception as e:
            return (
                gr.update(value=f"❌ Error: {e}", visible=True),
                gr.update(value="", visible=False),
            )

    compare_btn.click(
        _do_compare, inputs=list(components.values()), outputs=[result_md, agreement_md]
    )


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
