from .predict_processing import ClassificationPredictprocess
from .predict_processing import TokenPredictprocess
from msnlp.config import TaskName

# Supported data processors
PREDICT_PROCESSOR = {
    TaskName.text_classification: ClassificationPredictprocess,
    TaskName.ner: TokenPredictprocess,
    TaskName.similarity: ClassificationPredictprocess,
    TaskName.sentiment2: ClassificationPredictprocess,
    TaskName.title_content_sent: ClassificationPredictprocess,
}