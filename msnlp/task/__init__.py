
from .classification import Classification
from .token_classification import TokenClassification
from msnlp.config import TaskName
from .word_similarity import WordSimilarity


# Supported task function
TASK_DICT = {
    TaskName.text_classification: Classification,
    TaskName.ner: TokenClassification,
    TaskName.similarity: Classification,
    TaskName.sentiment2: Classification,
    TaskName.title_content_sent: Classification,
    TaskName.word_similarity: WordSimilarity,
}
