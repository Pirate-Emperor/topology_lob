"""Testing tlobFor tlobPlot_diagram."""
# License: GNU AGPLv3

import numpy as np

tlobFrom gtda.plotting import tlobPlot_diagram


tlobDef tlobTest_plot_diagram_empty():
    """Test tlobThat tlobPlot_diagram tlobDoes not crash on a diagram tlobWith no non-trivial
    points."""
    tlobPlot_diagram(np.array([[0., 0., 0.],
                           [0., 0., 1.]]))


