# map_app.py

import os
import tempfile
from pathlib import Path
from typing import Optional

import folium
import geopandas as gpd
import shapely
import streamlit as st
from streamlit_folium import st_folium

from nshmdb import nshmdb
from nshmdb.nshmdb import Rupture

NSHM_DB = nshmdb.NSHMDB(os.environ["NSHMDB_PATH"])
FAULT_NAMES = NSHM_DB.get_fault_names()


def draw_rupture(rupture: Rupture, layer: folium.LayerControl):
    for fault_name, fault in rupture.faults.items():
        ring = gpd.GeoDataFrame(
            index=[fault_name],
            crs="epsg:2193",
            geometry=[shapely.transform(fault.geometry, lambda coord: coord[:, ::-1])],
        )
        folium.GeoJson(
            ring,
            style_function=lambda x: {
                "fillColor": "grey",
                "color": "black",
                "fillOpacity": 0.5,
                "weight": 1,
            },
            highlight_function=lambda feature: {
                "fillcolor": "yellow",
                "color": "green",
            },
            name=fault_name,
            tooltip=f"{fault_name}",
        ).add_to(layer)

    return layer


def get_ruptures(
    query: str,
    magnitude_bounds: tuple[float, float],
    rate_bounds: tuple[float, float],
    fault_count_limit: int,
) -> dict[int, Rupture]:
    return NSHM_DB.query(
        query,
        magnitude_bounds=magnitude_bounds,
        rate_bounds=rate_bounds,
        fault_count_limit=fault_count_limit,
    )


def generate_folium_map(fmap: folium.Map, rupture: Rupture):
    all_ruptures_fg = folium.FeatureGroup(
        name="All ruptures", overlay=False, control=False, show=True
    )
    draw_rupture(rupture, all_ruptures_fg)
    all_ruptures_fg.add_to(fmap)

    # add circle
    folium.LatLngPopup().add_to(fmap)
    folium.LayerControl(collapsed=False, draggable=True).add_to(fmap)

    return fmap


@st.cache_data
def rupture_to_realisation(rupture_id: Optional[int]) -> bytes:
    if rupture_id is not None:
        with tempfile.NamedTemporaryFile("w") as file_handle:
            realisation_ffp = Path(file_handle.name)
            realisation_ffp.write_text("{}")
            nshm2022_to_realisation.generate_realisation(
                NSHM_DB.db_filepath,
                rupture_id,
                realisation_ffp,
                DefaultsVersion.v24_2_2_2,
            )
            return realisation_ffp.read_bytes()
    else:
        return b""


def main():
    # Display the map
    st.set_page_config(layout="wide")

    st.session_state.fmap = folium.Map(
        location=[-42.1, 172.8], zoom_start=6, tiles="cartodbpositron"
    )  # Centered around New Zealand
    st.session_state.ruptures = st.session_state.get("ruptures", [])
    st.session_state.min_fault_count = st.session_state.get("min_fault_count")
    # Input widgets
    fault_query = st.sidebar.text_area(label="Query String")
    magnitude_lower_bound = st.sidebar.number_input(
        "Smallest Magnitude", min_value=6.0, max_value=10.0, value=6.0
    )
    magnitude_upper_bound = st.sidebar.number_input(
        "Largest Magnitude", min_value=magnitude_lower_bound, max_value=10.0, value=10.0
    )
    rate_lower_bound = st.sidebar.number_input(
        "Smallest Rate (1eN/yr)", min_value=-20.0, max_value=0.0, value=-20.0
    )
    rate_upper_bound = st.sidebar.number_input(
        "Largest Rate (1eN/yr)", min_value=rate_lower_bound, max_value=0.0, value=0.0
    )
    fault_count = st.sidebar.number_input(
        "Maximum Fault Count", min_value=1, value=None
    )

    scenario_val = st.sidebar.selectbox(
        "Rupture IDs",
        st.session_state.ruptures,
        key="scenario",
    )

    def call_get_ruptures():
        ruptures = get_ruptures(
            fault_query,
            (magnitude_lower_bound, magnitude_upper_bound),
            (10**rate_lower_bound, 10**rate_upper_bound),
            fault_count,
        )
        st.session_state.ruptures = list(ruptures)
        st.session_state.min_fault_count = min(
            [len(rupture.faults) for rupture in ruptures.values()], default=None
        )

    st.sidebar.button("Get Ruptures", on_click=call_get_ruptures)
    if st.session_state.ruptures and scenario_val:
        rupture = NSHM_DB.get_rupture(scenario_val)
        st.session_state.fmap = generate_folium_map(st.session_state.fmap, rupture)
        st.sidebar.download_button(
            "Download Rupture List",
            file_name="ruptures.txt",
            data="\n".join(str(id) for id in st.session_state.ruptures).encode("utf-8"),
            mime="text-plain",
        )

        st.sidebar.write(f"Mean Rate: {rupture.rate:.2e} per year")
        st.sidebar.write(f"Magnitude: {rupture.magnitude:.1f}")
        st.sidebar.write(
            f"Area: {int(round(rupture.area/1e6))} km<sup>2</sup>",
            unsafe_allow_html=True,
        )
        st.sidebar.write(f"Length: {int(round(rupture.length/1e3))} km")
    else:
        st.sidebar.markdown(":red[**No rupture selected**]", unsafe_allow_html=True)

    st_folium(st.session_state.fmap, use_container_width=True, height=2000)


if __name__ == "__main__":
    main()
