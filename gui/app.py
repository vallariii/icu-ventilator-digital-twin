"""
ICU Ventilator Digital Twin — Live Dashboard.

Run:   python gui/app.py
Open:  http://localhost:8050
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, dash_table

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "gui"))
import data_stream
import metrics

# ---- palette (kept in sync with assets/styles.css) --------------------
P = {
    "bg":      "#0f172a",
    "bg_elev": "#141e33",
    "card":    "#1a2744",
    "ink":     "#e8ecff",
    "muted":   "#93a0c7",
    "accent":  "#06b6d4",
    "accent2": "#22d3ee",
    "pink":    "#ec4899",
    "amber":   "#fbbf24",
    "ok":      "#10b981",
    "bad":     "#ef4444",
    "border":  "#2a3a5c",
}


def hex_to_rgba(h: str, alpha: float) -> str:
    """Convert '#rrggbb' to 'rgba(r,g,b,a)'."""
    h = h.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ---- plotly theme -----------------------------------------------------
def base_layout(title=None, height=240, xtitle=None, ytitle=None):
    """Fresh layout dict for every figure — avoids kwarg clashes."""
    return dict(
        paper_bgcolor=P["card"],
        plot_bgcolor=P["card"],
        font=dict(family="-apple-system,BlinkMacSystemFont,Segoe UI,Roboto,sans-serif",
                  color=P["ink"], size=12),
        margin=dict(l=50, r=20, t=50, b=40),
        height=height,
        title=dict(text=title or "",
                   font=dict(color=P["accent2"], size=13),
                   x=0.02, xanchor="left"),
        xaxis=dict(title=dict(text=xtitle or ""),
                   gridcolor=P["border"], zerolinecolor=P["border"],
                   linecolor=P["border"], tickfont=dict(color=P["muted"], size=10),
                   title_font=dict(color=P["muted"], size=11)),
        yaxis=dict(title=dict(text=ytitle or ""),
                   gridcolor=P["border"], zerolinecolor=P["border"],
                   linecolor=P["border"], tickfont=dict(color=P["muted"], size=10),
                   title_font=dict(color=P["muted"], size=11)),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=P["muted"])),
    )


# ----------------------------------------------------------------------
# 3D VENTILATOR MODEL
# ----------------------------------------------------------------------
def build_3d_ventilator(state):
    """Stylised 3D ventilator with sensor-zone colour-coded markers."""
    # Main chassis (rectangular box via mesh3d)
    chassis = go.Mesh3d(
        x=[0, 5, 5, 0, 0, 5, 5, 0],
        y=[0, 0, 2.5, 2.5, 0, 0, 2.5, 2.5],
        z=[0, 0, 0, 0, 2, 2, 2, 2],
        i=[0, 0, 0, 0, 4, 4, 4, 2, 2, 1, 1, 5],
        j=[1, 2, 4, 7, 5, 6, 7, 3, 7, 2, 5, 6],
        k=[2, 3, 7, 4, 6, 7, 0, 7, 6, 6, 6, 7],
        opacity=0.35, color=P["accent"], flatshading=True,
        hoverinfo="skip", showscale=False, name="Chassis",
    )

    # Turbine cylinder — two caps + lateral mesh via parametric surface
    theta = np.linspace(0, 2 * np.pi, 40)
    zz = np.linspace(0.3, 1.7, 20)
    T, Z = np.meshgrid(theta, zz)
    R = 0.55
    Xc = 5.9 + R * np.cos(T)
    Yc = 1.25 + R * np.sin(T)
    turbine = go.Surface(
        x=Xc, y=Yc, z=Z,
        colorscale=[[0, P["pink"]], [1, P["amber"]]],
        showscale=False, opacity=0.75, hoverinfo="skip",
    )

    # Sensor markers — colour reflects live reading
    def zone(val, lo, hi):
        return P["ok"] if val < lo else (P["amber"] if val < hi else P["bad"])

    labels = ["DHT22", "MPU6050", "MPX5010", "ACS712", "HX710B"]
    positions = [
        (0.6, 0.3, 2.1),
        (2.5, 2.3, 2.1),
        (4.2, 0.3, 2.1),
        (5.9, 2.1, 1.9),
        (4.8, 1.0, 0.1),
    ]
    live_vals = [
        zone(state.get("temperature_c", 25), 30, 40),
        zone(state.get("vibration_rms", 0), 0.15, 0.3),
        zone(state.get("airflow_kpa", 2.4), 3.0, 5.0),
        zone(state.get("current_a", 1.4), 1.6, 2.0),
        zone(state.get("pressure_pa", 1000) / 1000, 1.5, 2.0),
    ]
    xs, ys, zs = zip(*positions)
    sensors = go.Scatter3d(
        x=xs, y=ys, z=zs, mode="markers+text",
        marker=dict(size=16, color=live_vals,
                    line=dict(width=2, color="white"), symbol="circle"),
        text=labels, textposition="top center",
        textfont=dict(color=P["ink"], size=11),
        hovertemplate="<b>%{text}</b><extra></extra>",
        name="Sensors",
    )

    fig = go.Figure([chassis, turbine, sensors])
    fig.update_layout(
        paper_bgcolor=P["card"], plot_bgcolor=P["card"],
        margin=dict(l=0, r=0, t=0, b=0), showlegend=False, height=420,
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False),
            zaxis=dict(visible=False),
            bgcolor=P["card"],
            camera=dict(eye=dict(x=2.2, y=-2.2, z=1.4)),
            aspectmode="data",
        ),
    )
    return fig


# ----------------------------------------------------------------------
# Dash app
# ----------------------------------------------------------------------
app = Dash(__name__, title="ICU Ventilator Digital Twin",
           assets_folder=str(ROOT / "gui" / "assets"))
server = app.server

data_stream.start()
HEADLINE = metrics.aggregate_headline()
METRICS_ROWS = metrics.summary_table()
CM_LABELS, CM_MATRIX = metrics.confusion_matrix_data()


def kpi(title, value, unit="", cls=""):
    return html.Div(className=f"card kpi {cls}", children=[
        html.H3(title),
        html.Div([
            html.Span(value, className="value"),
            html.Span(unit, className="unit"),
        ]),
    ])


app.layout = html.Div([
    html.Div(className="banner", children=[
        html.H1([
            "ICU Ventilator Digital Twin",
            html.Span(" · Real-time sensor intelligence", className="sub"),
        ]),
        html.Span(id="source-badge", className="badge-live", children="LIVE"),
    ]),

    html.Div(className="container", children=[

        html.Div(className="section-title", children="Live Sensor Readings"),
        html.Div(className="grid cols-6", id="kpi-row"),

        html.Div(className="section-title", children="3D Digital Twin · Sensor Zones"),
        html.Div(className="grid cols-2", children=[
            html.Div(className="card", style={"padding": "4px"}, children=[
                dcc.Graph(id="ventilator-3d", config={"displayModeBar": False}),
            ]),
            html.Div(className="card", children=[
                html.H3("Twin State"),
                html.Div(id="state-panel",
                         style={"fontFamily": "ui-monospace, Menlo, Consolas, monospace",
                                "marginTop": "8px"}),
            ]),
        ]),

        html.Div(className="section-title", children="Live Telemetry"),
        html.Div(className="grid cols-3", children=[
            dcc.Graph(id="temp-graph", config={"displayModeBar": False}),
            dcc.Graph(id="vib-graph", config={"displayModeBar": False}),
            dcc.Graph(id="current-graph", config={"displayModeBar": False}),
            dcc.Graph(id="anomaly-graph", config={"displayModeBar": False}),
            dcc.Graph(id="rul-graph", config={"displayModeBar": False}),
            dcc.Graph(id="airflow-graph", config={"displayModeBar": False}),
        ]),

        html.Div(className="section-title", children="Model Performance Metrics"),
        html.Div(className="grid cols-4", children=[
            kpi("Accuracy",         f"{HEADLINE['accuracy']:.1f}", "%", "ok"),
            kpi("Precision",        f"{HEADLINE['precision']:.1f}", "%"),
            kpi("Recall",           f"{HEADLINE['recall']:.1f}", "%", "ok"),
            kpi("F1-Score",         f"{HEADLINE['f1']:.3f}", ""),
            kpi("RMSE (RUL)",       f"{HEADLINE['rmse_cycles']:.1f}", "cycles", "warn"),
            kpi("Avg. Latency",     "< 5", "ms", "ok"),
            kpi("Failure Reduction", f"{HEADLINE['failure_reduction']:.1f}", "%", "ok"),
            kpi("Models Served",    "6", "", ""),
        ]),

        html.Div(className="grid cols-2", children=[
            html.Div(className="card", children=[
                html.H3("Per-Model Metrics"),
                dash_table.DataTable(
                    data=METRICS_ROWS,
                    columns=[{"name": k, "id": k} for k in METRICS_ROWS[0].keys()],
                    style_as_list_view=True,
                    style_cell={
                        "backgroundColor": P["card"], "color": P["ink"],
                        "fontFamily": "-apple-system,Segoe UI,Roboto,sans-serif",
                        "fontSize": "11px",
                        "padding": "8px 10px", "textAlign": "left",
                        "border": "none",
                        "borderBottom": f"1px solid {P['border']}",
                    },
                    style_header={
                        "backgroundColor": P["bg_elev"], "color": P["muted"],
                        "fontWeight": "600", "textTransform": "uppercase",
                        "letterSpacing": "0.6px", "border": "none",
                        "fontSize": "10px",
                    },
                    style_data_conditional=[
                        {"if": {"column_id": "Model"},
                         "color": P["accent2"], "fontWeight": "600"},
                    ],
                ),
            ]),
            html.Div(className="card", style={"padding": "4px"}, children=[
                dcc.Graph(id="confusion-matrix", config={"displayModeBar": False}),
            ]),
        ]),

        html.Div(className="grid cols-2", children=[
            html.Div(className="card", style={"padding": "4px"}, children=[
                dcc.Graph(id="latency-bars", config={"displayModeBar": False}),
            ]),
            html.Div(className="card", style={"padding": "4px"}, children=[
                dcc.Graph(id="metric-radar", config={"displayModeBar": False}),
            ]),
        ]),

        html.Footer(style={"marginTop": 40, "textAlign": "center",
                           "color": P["muted"], "fontSize": "0.85rem",
                           "paddingBottom": "20px"},
                    children=[
            "ICU Ventilator Digital Twin  ·  Raspberry Pi 5 + IoT + ML  ·  ",
            "Built from PDF blueprint",
        ]),

        dcc.Interval(id="tick", interval=1500, n_intervals=0),
    ]),
])


# ----------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------
def _ts_line(history, ymap, title, ylabel, color):
    layout = base_layout(title=title, height=240,
                         xtitle="seconds ago", ytitle=ylabel)
    if not history:
        return go.Figure(layout=layout)
    last_ts = history[-1].ts
    x_rel = [-(last_ts - h.ts) for h in history]
    ys = [ymap(h) for h in history]
    fig = go.Figure([go.Scatter(
        x=x_rel, y=ys, mode="lines",
        line=dict(color=color, width=2.5, shape="spline"),
        fill="tozeroy",
        fillcolor=hex_to_rgba(color, 0.18),
        hovertemplate=f"{ylabel}: %{{y:.3f}}<br>%{{x:.1f}} s ago<extra></extra>",
    )])
    fig.update_layout(layout)
    return fig


@app.callback(
    Output("kpi-row", "children"),
    Output("state-panel", "children"),
    Output("ventilator-3d", "figure"),
    Output("temp-graph", "figure"),
    Output("vib-graph", "figure"),
    Output("current-graph", "figure"),
    Output("anomaly-graph", "figure"),
    Output("rul-graph", "figure"),
    Output("airflow-graph", "figure"),
    Output("confusion-matrix", "figure"),
    Output("latency-bars", "figure"),
    Output("metric-radar", "figure"),
    Output("source-badge", "children"),
    Input("tick", "n_intervals"),
)
def update(_):
    latest = data_stream.get_latest()
    history = data_stream.get_history()
    if latest is None:
        empty = go.Figure(layout=PLOTLY_LAYOUT)
        return ([], "", empty, empty, empty, empty, empty, empty,
                empty, empty, empty, empty, "…")

    rul_pct = latest.rul_fraction * 100
    rul_cls = "ok" if rul_pct > 50 else "warn" if rul_pct > 20 else "bad"
    fault_cls = "ok" if latest.fault_class == "Normal" else "warn"
    kpi_row = [
        kpi("Temperature",     f"{latest.temperature_c:.1f}", "°C"),
        kpi("Vibration RMS",   f"{latest.vibration_rms:.3f}", "g"),
        kpi("Motor Current",   f"{latest.current_a:.2f}", "A"),
        kpi("Airway Pressure", f"{latest.airflow_kpa:.2f}", "kPa"),
        kpi("RUL",             f"{rul_pct:.1f}", "%", rul_cls),
        kpi("Fault Class",     latest.fault_class, "", fault_cls),
    ]

    state_rows = [
        ("Anomaly score",     f"{latest.anomaly_score:+.3f}"),
        ("Fault probability", f"{latest.fault_probability * 100:.1f} %"),
        ("Humidity",          f"{latest.humidity_pct:.1f} %"),
        ("HX710B pressure",   f"{latest.pressure_pa:.0f} Pa"),
        ("Accel X / Y / Z",   f"{latest.vib_ax:+.2f}  /  {latest.vib_ay:+.2f}  /  {latest.vib_az:+.2f} g"),
        ("Data source",       getattr(latest, "source", "sim").upper()),
    ]
    state_panel = html.Div([
        html.Div(style={"display": "flex", "justifyContent": "space-between",
                        "padding": "10px 0",
                        "borderBottom": f"1px solid {P['border']}"},
                 children=[html.Span(k, style={"color": P["muted"], "fontSize": "0.85rem"}),
                           html.Span(v, style={"color": P["accent2"],
                                                "fontWeight": 600, "fontSize": "0.95rem"})])
        for k, v in state_rows
    ])

    fig_3d = build_3d_ventilator({
        "temperature_c": latest.temperature_c,
        "vibration_rms": latest.vibration_rms,
        "airflow_kpa":   latest.airflow_kpa,
        "current_a":     latest.current_a,
        "pressure_pa":   latest.pressure_pa,
    })

    fig_t  = _ts_line(history, lambda h: h.temperature_c,    "Temperature",      "°C",   P["accent2"])
    fig_v  = _ts_line(history, lambda h: h.vibration_rms,    "Vibration RMS",    "g",    P["pink"])
    fig_c  = _ts_line(history, lambda h: h.current_a,        "Motor Current",    "A",    P["amber"])
    fig_a  = _ts_line(history, lambda h: h.anomaly_score,    "Anomaly Score",    "score", P["bad"])
    fig_r  = _ts_line(history, lambda h: h.rul_fraction * 100, "RUL",             "%",    P["ok"])
    fig_af = _ts_line(history, lambda h: h.airflow_kpa,      "Airway Pressure",  "kPa",  P["accent"])

    # --- Confusion matrix ---
    if CM_MATRIX:
        cm_fig = go.Figure(go.Heatmap(
            z=CM_MATRIX, x=CM_LABELS, y=CM_LABELS,
            colorscale=[[0, P["card"]], [0.3, P["accent"]], [1, P["pink"]]],
            text=CM_MATRIX, texttemplate="%{text}",
            textfont={"color": "white", "size": 12},
            showscale=False,
            hovertemplate="actual %{y}<br>predicted %{x}<br>count %{z}<extra></extra>",
        ))
        cm_layout = base_layout(title="Fault Classifier — Confusion Matrix",
                                 height=380, xtitle="Predicted", ytitle="Actual")
        cm_layout["yaxis"]["autorange"] = "reversed"
        cm_fig.update_layout(cm_layout)
    else:
        cm_fig = go.Figure(layout=base_layout(height=380))

    # --- Latency bar chart ---
    lat_data = metrics.measure_latency_ms()
    lat_pairs = [(k, v) for k, v in lat_data.items() if v is not None]
    lat_pairs.sort(key=lambda kv: kv[1], reverse=True)
    lat_names = [k for k, _ in lat_pairs]
    lat_vals = [v for _, v in lat_pairs]
    lat_fig = go.Figure(go.Bar(
        x=lat_vals, y=lat_names, orientation="h",
        marker=dict(color=P["accent2"],
                    line=dict(color=P["accent"], width=1)),
        text=[f"{v:.1f} ms" for v in lat_vals], textposition="inside",
        textfont=dict(color="white", size=11),
    ))
    lat_layout = base_layout(title="Inference Latency per Model",
                              height=380, xtitle="milliseconds")
    lat_layout["bargap"] = 0.35
    lat_fig.update_layout(lat_layout)

    # --- Radar ---
    categories = ["Accuracy", "Precision", "Recall", "F1×100",
                  "Failure Red.", "Speed"]
    avg_lat = np.mean(lat_vals) if lat_vals else 1
    speed_score = max(0.0, 100 - min(avg_lat * 2, 100))
    radar_vals = [HEADLINE["accuracy"], HEADLINE["precision"], HEADLINE["recall"],
                  HEADLINE["f1"] * 100, HEADLINE["failure_reduction"], speed_score]
    radar = go.Figure(go.Scatterpolar(
        r=radar_vals + [radar_vals[0]],
        theta=categories + [categories[0]],
        fill="toself",
        line=dict(color=P["pink"], width=2.5),
        fillcolor=hex_to_rgba(P["pink"], 0.25),
    ))
    radar.update_layout(
        paper_bgcolor=P["card"], font=dict(color=P["ink"], size=11),
        polar=dict(bgcolor=P["card"],
                   radialaxis=dict(visible=True, range=[0, 100],
                                   gridcolor=P["border"], linecolor=P["border"],
                                   tickfont=dict(color=P["muted"], size=9)),
                   angularaxis=dict(gridcolor=P["border"], linecolor=P["border"],
                                    tickfont=dict(color=P["ink"], size=11))),
        title=dict(text="Overall System Scorecard",
                   font=dict(color=P["accent2"], size=13), x=0.5),
        height=380, showlegend=False,
        margin=dict(l=60, r=60, t=60, b=60),
    )

    source = getattr(latest, "source", "sim")
    badge = "LIVE · PI" if source == "pi" else "LIVE · SIM"

    return (kpi_row, state_panel, fig_3d, fig_t, fig_v, fig_c, fig_a, fig_r, fig_af,
            cm_fig, lat_fig, radar, badge)


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=8050)
