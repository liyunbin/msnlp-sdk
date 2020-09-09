from .classification import (
    ClassificationProcessor,
    SimilarityProcessor,
    TitlecontentProcessor,
)
from .token_classification import TokenClassProcessor
from msnlp.config import TaskName


# Supported processors
DATA_PROCESSOR = {
    TaskName.text_classification: ClassificationProcessor,
    TaskName.similarity: SimilarityProcessor,
    TaskName.sentiment2: ClassificationProcessor,
    TaskName.title_content_sent: TitlecontentProcessor,
    TaskName.ner: TokenClassProcessor,
}
