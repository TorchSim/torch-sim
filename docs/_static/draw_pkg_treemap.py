"""Draw a treemap of the torch_sim package structure.

Run with python docs/_static/draw_pkg_treemap.py
"""

import os

import pymatviz as pmv


module_dir = os.path.dirname(__file__)
pmv.set_plotly_template("plotly_white")

pkg_name = "torch-sim"
fig = pmv.py_pkg_treemap(pkg_name.replace("-", "_"))
fig.layout.title.update(text=f"{pkg_name} Package Structure", font_size=20, x=0.5, y=0.98)
fig.show()
pmv.io.save_and_compress_svg(fig, f"{module_dir}/{pkg_name}-pkg-treemap.svg")
fig.write_html(f"{module_dir}/{pkg_name}-pkg-treemap.html", include_plotlyjs="cdn")
