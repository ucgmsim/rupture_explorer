import csv
import datetime
import json
import os
from io import StringIO
from typing import Optional

import geopandas as gpd
import numpy as np
import plotly.express as px
import shapely
from flask import Flask, Response, make_response, render_template, request, url_for

from nshmdb import nshmdb
from nshmdb.nshmdb import Rupture
from qcore import coordinates
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
    rupture.faults = {
        fault_name: fault
        for fault_name, fault in rupture.faults.items()
        if not fault.geometry.is_empty
    }
    ring = gpd.GeoDataFrame(
        index=list(rupture.faults),
        geometry=[
            shapely.transform(
                fault.geometry,
                lambda coord: coordinates.nztm_to_wgs_depth(coord)[:, ::-1],
            )
            for fault in rupture.faults.values()
        ],
    )
    ring["Name"] = list(rupture.faults)
    ring["Width (km)"] = [int(round(fault.width)) for fault in rupture.faults.values()]
    ring["Length (km)"] = [
        int(round(fault.length)) for fault in rupture.faults.values()
    ]
    ring["Segments"] = [len(fault.planes) for fault in rupture.faults.values()]
    ring["Mean Segment Rupture Rate"] = [
        fault_rates.get(fault_name, 0) / len(fault.planes)
        for fault_name, fault in rupture.faults.items()
    ]
    fig = px.choropleth_map(
        data_frame=ring,
        geojson=json.loads(ring.to_json()),
        locations=ring.index,
        color="Mean Segment Rupture Rate",
        hover_name="Name",
        hover_data={
            "Width (km)": True,
            "Length (km)": True,
            "Segments": True,
            "Mean Segment Rupture Rate": ":.2e",
        },
        opacity=0.5,
        center={"lat": -43, "lon": 172},
        zoom=6,
    )
    fig.update(layout_showlegend=False, layout_coloraxis_showscale=False)

    fig.update_layout(margin={"l": 0, "r": 0, "b": 0, "t": 0})
    # The following replace removes the erroneous request to trust the notebook.
    return fig.to_html(include_plotlyjs=False, full_html=False, div_id="map")


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
    if not query:
        response = make_response("")
        response.headers["HX-Push-Url"] = url_for("index")
        return response

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
    response = make_response(
        render_template("ruptures.html", ruptures=ruptures, magnitudes=magnitudes)
    )
    response.headers["HX-Push-Url"] = url_for(
        "index",
        query=query,
        magnitude_lower_bound=magnitude_lower_bound,
        magnitude_upper_bound=magnitude_upper_bound,
        rate_lower_bound=rate_lower_bound,
        rate_upper_bound=rate_upper_bound,
        max_fault_count=max_fault_count,
    )
    return response


@app.route("/download")
def download():
    """Serve a CSV file containing all the filtered ruptures."""
    rupture_ids = [
        int(rupture_id) for rupture_id in request.args.get("ruptures").split(",")
    ]
    db = nshmdb.NSHMDB(NSHMDB_PATH)
    ruptures: dict[int, Rupture] = {
        rupture_id: db.get_rupture(rupture_id) for rupture_id in rupture_ids
    }
    csv_out = StringIO()
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
    response: Response = make_response(csv_out.getvalue())
    response.mimetype = "application/x-csv"
    response.headers["Content-Disposition"] = (
        f"attachment; filename=ruptures_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')}.csv"
    )
    return response


@app.route("/")
def index() -> str:
    """Serve the index file."""
    query: Optional[str] = request.args.get("query", None, type=str)

    magnitude_lower_bound: Optional[float] = request.args.get(
        "magnitude_lower_bound", default=None, type=float
    )
    magnitude_upper_bound: Optional[float] = request.args.get(
        "magnitude_upper_bound", default=None, type=float
    )
    rate_lower_bound: Optional[float] = request.args.get(
        "rate_lower_bound", default=None, type=float
    )
    rate_upper_bound: Optional[float] = request.args.get(
        "rate_upper_bound", default=None, type=float
    )
    max_fault_count: Optional[int] = request.args.get(
        "max_fault_count", default=None, type=int
    )
    db = nshmdb.NSHMDB(NSHMDB_PATH)
    ruptures = None
    magnitudes = None
    if query:
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
    return render_template(
        "index.html",
        query=query,
        magnitude_lower_bound=magnitude_lower_bound,
        magnitude_upper_bound=magnitude_upper_bound,
        rate_lower_bound=rate_lower_bound,
        rate_upper_bound=rate_upper_bound,
        max_fault_count=max_fault_count,
        ruptures=ruptures,
        magnitudes=magnitudes,
    )


if __name__ == "__main__":
    app.run()
