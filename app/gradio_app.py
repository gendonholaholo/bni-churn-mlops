"""Single Gradio app: 3 tabs (Training Lab / Inference / A/B Test).

Action surface only — observation is in MLflow UI (port 5000).
"""

import gradio as gr
import mlflow

from mlops_churn import config


def build_training_lab() -> None:
    gr.Markdown("## 🎛️ Training Lab\n_(diisi di Task 19)_")


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
