from io import BytesIO
import base64

from sklearn.pipeline import Pipeline
from sklearn.utils._estimator_html_repr import estimator_html_repr
from sklearn.metrics import ConfusionMatrixDisplay


class SklearnEstimatorRenderer:
    """🃏 Easily extend Flyte Decks to visualize our model pipeline"""

    def to_html(self, pipeline: Pipeline) -> str:
        return estimator_html_repr(pipeline)


class ConfusionMatrixRenderer:

    def to_html(self, cm_display: ConfusionMatrixDisplay) -> str:
        buf = BytesIO()
        cm_display.plot().figure_.savefig(buf, format="png")
        encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"<img src='data:image/png;base64,{encoded}'>"
