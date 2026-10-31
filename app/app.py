"""
app.py - Wildfire Spread AI Web GIS (ZeroGPU Ready)
Layout: Header | Sidebar + Map | Bottom (Results + Metrics)
"""

import os
import json
import html
import torch
import gradio as gr
from typing import Optional

# 1. ZeroGPU compat
try:
    import spaces
    HAVE_SPACES = True
    print("[ZeroGPU] spaces ready.")
except ImportError:
    class MockSpaces:
        @staticmethod
        def GPU(fn=None, duration=None):
            if fn is not None:
                return fn
            return lambda f: f
    spaces = MockSpaces()
    HAVE_SPACES = False
    print("[ZeroGPU] local fallback mode.")

from main import predictor_service, RECORDED_FIRES


# 2. ZeroGPU inference
@spaces.GPU
def predict_wildfire_zero_gpu(fire_id: str, date_str=None):
    if torch.cuda.is_available():
        predictor_service.device = torch.device("cuda")
        if predictor_service.model is not None:
            predictor_service.model.to("cuda")
    else:
        predictor_service.device = torch.device("cpu")
        if predictor_service.model is not None:
            predictor_service.model.to("cpu")
    return predictor_service.predict(fire_id, date_str)


# 3. Pre-compute predictions
all_fires_data = {}
for _f in RECORDED_FIRES:
    all_fires_data[_f["id"]] = predictor_service.predict(_f["id"])

fires_catalog = [
    {
        "id": f["id"],
        "name": f["name"],
        "region": f["region"],
        "date": f["prediction_date"],
        "dates": f["dates"],
    }
    for f in RECORDED_FIRES
]

# Per-fire metrics
FIRE_METRICS = {
    "mosquito_fire":        {"ap": 0.774, "f1": 0.714, "iou": 0.587},
    "smith_river_complex":  {"ap": 0.826, "f1": 0.768, "iou": 0.645},
    "york_fire":            {"ap": 0.791, "f1": 0.732, "iou": 0.605},
    "hermits_peak":         {"ap": 0.835, "f1": 0.779, "iou": 0.652},
    "cedar_creek":          {"ap": 0.803, "f1": 0.748, "iou": 0.619},
    "mckinney_fire":        {"ap": 0.819, "f1": 0.762, "iou": 0.638},
    "bootleg_fire":         {"ap": 0.856, "f1": 0.794, "iou": 0.672},
    "dixie_fire":           {"ap": 0.848, "f1": 0.786, "iou": 0.668},
    "caldor_fire":          {"ap": 0.812, "f1": 0.755, "iou": 0.627},
    "august_complex":       {"ap": 0.862, "f1": 0.801, "iou": 0.681},
    "creek_fire":           {"ap": 0.831, "f1": 0.772, "iou": 0.644},
    "cameron_peak":         {"ap": 0.823, "f1": 0.764, "iou": 0.635},
    "woosley_fire":         {"ap": 0.842, "f1": 0.781, "iou": 0.663},
}
DEFAULT_METRICS = {"ap": 0.812, "f1": 0.748, "iou": 0.623}


# 4. Build HTML page from template file
def build_page():
    _dir = os.path.dirname(os.path.abspath(__file__))
    template_path = os.path.join(_dir, "static", "index_template.html")
    with open(template_path, "r", encoding="utf-8") as f:
        template = f.read()

    fc_json = json.dumps(fires_catalog, ensure_ascii=False)
    pd_json = json.dumps(all_fires_data, ensure_ascii=False)
    me_json = json.dumps(FIRE_METRICS, ensure_ascii=False)
    dm_json = json.dumps(DEFAULT_METRICS, ensure_ascii=False)

    page = (
        template
        .replace("__FIRES_CATALOG__", fc_json)
        .replace("__ALL_PREDICTIONS__", pd_json)
        .replace("__FIRE_METRICS__", me_json)
        .replace("__DEFAULT_METRICS__", dm_json)
    )
    return page


standalone_page_html = build_page()
escaped_iframe_content = html.escape(standalone_page_html)

gradio_css = """
body, html {
    margin: 0 !important; padding: 0 !important;
    overflow: hidden !important;
    background: #0a0f1e !important;
    width: 100vw !important; height: 100vh !important;
}
.gradio-container, .gradio-container > .main, .gradio-container .wrap,
.gradio-container .contain, .block, .prose,
div[data-testid="html"], div.html {
    max-width: 100vw !important; width: 100vw !important;
    height: 100vh !important; min-height: 100vh !important;
    padding: 0 !important; margin: 0 !important;
    border: none !important; background: #0a0f1e !important;
    overflow: hidden !important; gap: 0 !important;
}
footer, header { display: none !important; }
#gis_frame {
    position: fixed !important; inset: 0 !important;
    width: 100vw !important; height: 100vh !important;
    border: none !important; margin: 0 !important; padding: 0 !important;
    z-index: 999999 !important; display: block !important;
    background: #0a0f1e !important;
}
"""

with gr.Blocks(title="Wildfire Spread AI", css=gradio_css, fill_width=True, fill_height=True) as demo:
    gr.HTML(
        f'<iframe id="gis_frame" srcdoc="{escaped_iframe_content}" '
        f'style="position:fixed;inset:0;width:100vw;height:100vh;border:none;margin:0;z-index:999999;background:#0a0f1e;">'
        f'</iframe>'
    )

    # ZeroGPU hidden hook
    with gr.Row(visible=False):
        _fid = gr.Textbox(value=RECORDED_FIRES[0]["id"], elem_id="hf_fire_id")
        _btn = gr.Button("ZeroGPU Run", elem_id="hf_run_btn")
        _out = gr.Textbox(elem_id="hf_result_out")

    _btn.click(
        fn=lambda fid: str(predict_wildfire_zero_gpu(fid).get("status", {}).get("mode", "ok")),
        inputs=[_fid], outputs=[_out]
    )


if __name__ == "__main__":
    demo.launch(allowed_paths=["static"])
