"""
Network Visualization Module (based on the original Network Visualization Module

Module for rendering the network, using [d3.js](https://d3js.org/) frameworks.

"""
from mesa.visualization.ModularVisualization import VisualizationElement


class XYScaledNetworkModule(VisualizationElement):
    package_includes = []

    def __init__(
            self, portrayal_method, canvas_height=500, canvas_width=500):
        self.local_includes = ["./visualization/templates/js/d3.min.js",
                               "./visualization/templates/js/XYScaledNetworkModule_d3.js"]

        self.portrayal_method = portrayal_method
        self.canvas_height = canvas_height
        self.canvas_width = canvas_width
        new_element = "new XYScaledNetworkModule({}, {})".format(
            self.canvas_width, self.canvas_height
        )
        self.js_code = "elements.push(" + new_element + ");"

    def render(self, model):
        return self.portrayal_method(model)
