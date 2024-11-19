import csv
import datetime
import os
import tempfile
from typing import Optional

import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np
import shapely
from flask import Flask, render_template, request, send_file

from nshmdb import nshmdb
from nshmdb.nshmdb import Rupture
from qcore.uncertainties import mag_scaling
from source_modelling.sources import Fault

app = Flask(__name__)
NSHMDB_PATH = os.environ["NSHMDB_PATH"]


def default_magnitude_estimation(
    faults: dict[str, Fault], rakes: dict[str, float]
) -> dict[str, float]:
    """Estimate the magnitudes for a set of faults based on their areas and average rake.

    Parameters
    ----------
    faults : dict
        A dictionary where the keys are fault names and the values are `Fault` objects containing information about each fault.
    rakes : dict
        A dictionary where the keys are fault names and the values are rake angles (in degrees) for each fault.

    Returns
    -------
    dict
        A dictionary where the keys are fault names and the values are the estimated magnitudes for each fault.
    """
    total_area = sum(fault.area() for fault in faults.values())
    avg_rake = np.mean(list(rakes.values()))
    estimated_mw = mag_scaling.a_to_mw_leonard(total_area, 4, 3.99, avg_rake)
    estimated_moment = mag_scaling.mag2mom(estimated_mw)
    return {
        fault_name: mag_scaling.mom2mag((fault.area() / total_area) * estimated_moment)
        for fault_name, fault in faults.items()
    }


@app.route("/rupture_map/<rupture_id>")
def rupture_map(rupture_id: int) -> str:
    """Return a map of the rupture.

    Parameters
    ----------
    rupture_id : int
        The id of the rupture map to serve.

    Returns
    -------
    str
        A map of the rupture faults.
    """
    db = nshmdb.NSHMDB(NSHMDB_PATH)
    rupture = db.get_rupture(rupture_id)
    fault_info = db.get_rupture_fault_info(rupture_id)
    magnitudes = default_magnitude_estimation(
        rupture.faults, {name: info.rake for name, info in fault_info.items()}
    )
    fault_rates = db.most_likely_fault(rupture_id, magnitudes)

    fmap = folium.Map(tiles="cartodbpositron")
    all_ruptures_fg = folium.FeatureGroup(
        name=f"Rupture {rupture_id}", overlay=False, control=False, show=True
    )

    all_ruptures_fg.add_to(fmap)
    ring = gpd.GeoDataFrame(
        index=list(rupture.faults),
        crs="epsg:2193",
        geometry=[
            shapely.transform(fault.geometry, lambda coord: coord[:, ::-1])
            for fault in rupture.faults.values()
        ],
    )
    min_rate = min(fault_rates.values())
    linear = cm.LinearColormap(
        ["green", "yellow", "red"],
        vmin=np.log(min_rate),
        vmax=np.log(max(fault_rates.values())),
    )
    ring["style"] = [
        {
            "color": "black",
            "fillOpacity": 0.5,
            "weight": 1,
            "fillColor": linear(np.log(fault_rates.get(fault_name, min_rate))),
        }
        for fault_name in rupture.faults
    ]
    ring["Name"] = list(rupture.faults)
    ring["Width (km)"] = [int(round(fault.width)) for fault in rupture.faults.values()]
    ring["Length (km)"] = [
        int(round(fault.length)) for fault in rupture.faults.values()
    ]
    ring["Segments"] = [len(fault.planes) for fault in rupture.faults.values()]
    ring["Mean Segment Rupture Rate"] = [
        f"{fault_rates.get(fault_name, 0) / len(fault.planes):.2e}"
        for fault_name, fault in rupture.faults.items()
    ]

    tooltip = folium.GeoJsonPopup(
        fields=[
            "Name",
            "Length (km)",
            "Width (km)",
            "Segments",
            "Mean Segment Rupture Rate",
        ],
        localize=True,
        labels=True,
    )

    folium.GeoJson(ring, popup=tooltip).add_to(all_ruptures_fg)

    folium.LatLngPopup().add_to(fmap)
    folium.LayerControl(collapsed=False, draggable=True).add_to(fmap)
    folium.FitOverlays().add_to(fmap)

    folium_map_render: str = fmap.get_root()._repr_html_()
    # The following replace removes the erroneous request to trust the notebook.
    return folium_map_render.replace(
        '<span style="color:#565656">Make this Notebook Trusted to load map: File -> Trust Notebook</span>',
        "",
    )


@app.template_filter("fault_summary")
def rupture_fault_summary(faults: dict[str, Fault]) -> str:
    """Summarise the faults in a rupture.

    Parameters
    ----------
    faults : dict[str, Fault]
        The faults in the rupture.


    Returns
    -------
    str
        A summary string listing the faults in the rupture.
    """
    fault_names = sorted(list(faults))
    summary = fault_names[0]
    if len(fault_names) > 1:
        summary += f" + {len(fault_names) - 1} others"
    return summary


@app.route("/ruptures", methods=["POST"])
def ruptures() -> str:
    """Query the NSHMDB based on a query string and filtering parameters.

    Returns
    -------
    str
        A table containing all the ruptures.
    """
    query: str = request.form.get("query", type=str)

    magnitude_lower_bound: Optional[float] = request.form.get(
        "magnitude_lower_bound", default=None, type=float
    )
    magnitude_upper_bound: Optional[float] = request.form.get(
        "magnitude_upper_bound", default=None, type=float
    )
    rate_lower_bound: Optional[float] = request.form.get(
        "rate_lower_bound", default=None, type=float
    )
    rate_upper_bound: Optional[float] = request.form.get(
        "rate_upper_bound", default=None, type=float
    )
    max_fault_count: Optional[int] = request.form.get(
        "max_fault_count", default=None, type=int
    )
    db = nshmdb.NSHMDB(NSHMDB_PATH)
    ruptures = db.query(
        query,
        magnitude_bounds=(magnitude_lower_bound, magnitude_upper_bound),
        rate_bounds=(
            10**rate_lower_bound if rate_lower_bound is not None else None,
            10**rate_upper_bound if rate_upper_bound is not None else None,
        ),
        limit=100,
        fault_count_limit=max_fault_count,
    )
    magnitudes = {
        rupture_id: mag_scaling.a_to_mw_leonard(
            sum(fault.area() for fault in rupture.faults.values()), 4, 3.99, 0
        )
        for rupture_id, rupture in ruptures.items()
    }

    return render_template("ruptures.html", ruptures=ruptures, magnitudes=magnitudes)


@app.route("/download")
def download_ruptures():
    """Serve a CSV file containing all the filtered ruptures."""
    rupture_ids = [
        int(rupture_id) for rupture_id in request.args.get("ruptures").split(",")
    ]
    db = nshmdb.NSHMDB(NSHMDB_PATH)
    ruptures: dict[int, Rupture] = {
        rupture_id: db.get_rupture(rupture_id) for rupture_id in rupture_ids
    }
    with tempfile.NamedTemporaryFile(mode="w") as csv_out:
        writer = csv.DictWriter(
            csv_out, ["Rupture ID", "Magnitude", "Area", "Length", "Rate"]
        )
        writer.writeheader()
        writer.writerows(
            [
                {
                    "Rupture ID": rupture_id,
                    "Magnitude": rupture.magnitude,
                    "Area": rupture.area,
                    "Length": rupture.length,
                    "Rate": rupture.rate,
                }
                for rupture_id, rupture in ruptures.items()
            ]
        )
        download_name = (
            f'ruptures_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv'
        )
        return send_file(
            csv_out.name,
            mimetype="application/x-csv",
            download_name=download_name,
            as_attachment=True,
        )


@app.route("/")
def index() -> str:
    """Serve the index file."""
    return render_template("index.html")


if __name__ == "__main__":
    app.run()
