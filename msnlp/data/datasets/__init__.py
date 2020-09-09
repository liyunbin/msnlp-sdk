from .classification import ClassificationPredictSet
from .token_classification import NerPredictDataset
from msnlp.config import TaskName

# Supported predict dataset
PREDICTSET_DICT = {
    TaskName.text_classification: ClassificationPredictSet,
    TaskName.ner: NerPredictDataset,
    TaskName.similarity: ClassificationPredictSet,
    TaskName.sentiment2: ClassificationPredictSet,
    TaskName.title_content_sent: ClassificationPredictSet,
}