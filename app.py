# === app.py ===
import pandas as pd
import geopandas as gpd
import numpy as np
from pathlib import Path
import re, warnings, unidecode, janitor
from dash import Dash, html, dcc, Input, Output
import plotly.express as px
from mapclassify import Quantiles, NaturalBreaks

warnings.filterwarnings("ignore")

# === RUTAS ===
base = Path(__file__).parent
path_mm = base / "Datos_2023_550.xlsx"
path_nv = base / "Nacimientos_2023.xlsx"
path_shp = base / "MGN_DPTO_POLITICO.shp"


# === CARGA DE DATOS ===
mm = pd.read_excel(path_mm, engine="openpyxl")
deptos = gpd.read_file(path_shp).to_crs(4326)

# === NORMALIZADORES ===
def _norm(s):
    s = "" if pd.isna(s) else str(s)
    return unidecode.unidecode(s).upper().strip()

def _series1d(df, col):
    if col is None or col not in df.columns:
        return pd.Series([""] * len(df), index=df.index)
    x = df[col]
    if isinstance(x, pd.DataFrame):
        x = x.iloc[:, 0]
    return x

def _to_num(s):
    return pd.to_numeric(s.astype(str).str.replace(r"[^\d\-]", "", regex=True), errors="coerce")

# === REFERENCIAS DEPARTAMENTALES ===
cols = deptos.columns.astype(str)
cand_code = [c for c in cols if re.search(r"(CCDGO|CODIGO|CDGO)$", c, flags=re.I)]
cand_name = [c for c in cols if re.search(r"(NMBR|NOMBRE|NOM)$", c, flags=re.I)]
col_code, col_name = cand_code[0], cand_name[0]

ref_raw = deptos[[col_code, col_name]].copy()
ref_raw = ref_raw.rename(columns={col_code: "DEPTO_CODE", col_name: "DEPTO_NOMBRE"})
ref_raw["DEPTO_CODE"] = ref_raw["DEPTO_CODE"].astype(str).str.extract(r"(\d{2})", expand=False)
ref_raw["DEPTO_NOMBRE"] = ref_raw["DEPTO_NOMBRE"].astype(str).map(_norm)
ref = ref_raw.drop_duplicates()

# === NACIMIENTOS ===
sheet = "Cuadro3"
raw = pd.read_excel(path_nv, sheet_name=sheet, header=None, dtype=str).dropna(how="all")

hdr = None
for i in range(min(100, len(raw))):
    row = " ".join(_norm(x) for x in raw.iloc[i].tolist())
    if "DEPART" in row and "TOTAL" in row:
        hdr = i
        break

if hdr is None:
    raise ValueError("No se detectó el encabezado del Cuadro 3.")

df = raw.iloc[hdr + 1:].copy()
df.columns = [_norm(c) for c in raw.iloc[hdr].tolist()]
df = df.clean_names().remove_empty().dropna(how="all")

col_nom = next((c for c in df.columns if c.startswith("departamento")), None)
col_tot = next((c for c in df.columns if c.startswith("total")), None)

name_nom = _series1d(df, col_nom).astype(str)
tot = _to_num(_series1d(df, col_tot))

df_nv = pd.DataFrame({"DEPTO_NOM": name_nom, "NV_TOTAL": tot})
df_nv["DEPTO_CODE"] = df_nv["DEPTO_NOM"].str.extract(r"^(\d{2})", expand=False)
df_nv = df_nv.dropna(subset=["DEPTO_CODE", "NV_TOTAL"])
df_nv["DEPTO_CODE"] = df_nv["DEPTO_CODE"].astype(str).str.zfill(2)
nv_agg = df_nv.groupby("DEPTO_CODE", as_index=False)["NV_TOTAL"].sum()

# === MUERTES MATERNAS ===
pref = ["COD_DPTO_R", "COD_DPTO_O", "COD_DPTO_N"]
mm_col = next((c for c in pref if c in mm.columns), None)
mm["DEPTO_CODE"] = mm[mm_col].astype(str).str.extract(r"(\d{2})", expand=False).str.zfill(2)
mm_agg = mm.groupby("DEPTO_CODE", as_index=False).size().rename(columns={"size": "MM_MUERTES"})

# === MERGE  ===
datos = deptos.rename(columns={"DPTO_CCDGO": "DEPTO_CODE", "DPTO_CNMBR": "DEPTO_NOMBRE"})
datos["DEPTO_CODE"] = datos["DEPTO_CODE"].astype(str).str.zfill(2)
datos = (
    datos.merge(mm_agg, on="DEPTO_CODE", how="left")
          .merge(nv_agg, on="DEPTO_CODE", how="left")
)
datos["MM_MUERTES"] = datos["MM_MUERTES"].fillna(0).astype(int)
datos["NV_TOTAL"] = datos["NV_TOTAL"].fillna(0).astype(int)
datos["RMM_100k"] = np.where(datos["NV_TOTAL"] > 0,
                             datos["MM_MUERTES"] / datos["NV_TOTAL"] * 100000, np.nan)

gjson_global = datos.__geo_interface__

# === DASH APP ===
app = Dash(__name__)
app.title = "Tasa de Mortalidad Materna"
server = app.server


app.layout = html.Div([
    html.H1("Tasa de Mortalidad Materna en Colombia", style={"textAlign": "center", "fontWeight": "bold"}),
    html.P("Comparación entre departamentos según registros de SIVIGILA y DANE (2023)",
           style={"textAlign": "center", "marginBottom": "20px"}),

    html.Div([
        html.Label("Selecciona tipo de clasificación:", style={"fontWeight": "bold"}),
        dcc.Dropdown(
            id="clasificacion",
            options=[
                {"label": "Quantiles", "value": "Quantiles"},
                {"label": "Natural Breaks (Jenks)", "value": "NaturalBreaks"},
                {"label": "BoxPlot (IQR)", "value": "BoxPlot"},
            ],
            value="Quantiles",
            clearable=False,
            style={"width": "50%", "margin": "auto"}
        )
    ], style={"textAlign": "center", "marginBottom": "30px"}),

    html.Div([
        dcc.Graph(id="mapa_rmm", style={"height": "600px"}),
        dcc.Graph(id="mapa_mm", style={"height": "600px"})
    ])
])

# === CALLBACK ===
@app.callback(
    [Output("mapa_rmm", "figure"),
     Output("mapa_mm", "figure")],
    [Input("clasificacion", "value")]
)
def actualizar_mapas(clasif):
    df = datos.copy()
    gjson = gjson_global
    rmm = df["RMM_100k"].replace([np.inf, -np.inf], np.nan).fillna(0)
    mm = df["MM_MUERTES"].replace([np.inf, -np.inf], np.nan).fillna(0)

    def safe_bins(values, method):
        values = np.array(values)
        values = values[~np.isnan(values)]
        if len(np.unique(values)) < 3:
            return np.linspace(values.min(), values.max(), 6)

        if method == "Quantiles":
            return Quantiles(values, k=5).bins
        elif method == "NaturalBreaks":
            return NaturalBreaks(values, k=5).bins
        elif method == "BoxPlot":
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            low = max(values.min(), q1 - 1.5 * iqr)
            high = min(values.max(), q3 + 1.5 * iqr)
            return np.linspace(low, high, 6)
        else:
            return np.linspace(values.min(), values.max(), 6)

    bins_rmm = safe_bins(rmm, clasif)
    bins_mm = safe_bins(mm, clasif)

    title_suffix = {
        "Quantiles": "Quantiles",
        "NaturalBreaks": "Natural Breaks (Jenks)",
        "BoxPlot": "BoxPlot (IQR)"
    }.get(clasif, clasif)

    fig_rmm = px.choropleth_mapbox(
        df,
        geojson=gjson,
        locations="DEPTO_CODE",
        featureidkey="properties.DEPTO_CODE",
        color="RMM_100k",
        hover_name="DEPTO_NOMBRE",
        color_continuous_scale="YlOrRd",
        mapbox_style="carto-positron",
        zoom=4.3,
        center={"lat": 4.5, "lon": -74},
        title=f"RMM (por 100.000 NV) — {title_suffix}",
        range_color=[min(bins_rmm), max(bins_rmm)]
    )

    fig_mm = px.choropleth_mapbox(
        df,
        geojson=gjson,
        locations="DEPTO_CODE",
        featureidkey="properties.DEPTO_CODE",
        color="MM_MUERTES",
        hover_name="DEPTO_NOMBRE",
        color_continuous_scale="Oranges",
        mapbox_style="carto-positron",
        zoom=4.3,
        center={"lat": 4.5, "lon": -74},
        title=f"Muertes maternas — {title_suffix}",
        range_color=[min(bins_mm), max(bins_mm)]
    )

    for fig in [fig_rmm, fig_mm]:
        fig.update_layout(
            margin={"r": 0, "t": 50, "l": 0, "b": 0},
            title_x=0.5,
            title_font=dict(size=20),
            coloraxis_colorbar=dict(
                title="",
                tickfont=dict(size=12),
                thicknessmode="pixels",
                thickness=20
            )
        )

    return fig_rmm, fig_mm

# === EJECUCIÓN ===
if __name__ == "__main__":
    app.run(debug=False)











