"""Small offline mechanisms for chapters 4.1–4.4 and 6.1–6.2.

No model clients or embedding downloads. Fixed vectors test database mechanics,
not semantic retrieval quality. This directory is not an application framework.
"""
from .core import BoundedHistory, BudgetExceeded, TaskState, load_text, split_document
from .vector_memory import VectorMemory

__all__ = ['BoundedHistory', 'BudgetExceeded', 'TaskState', 'load_text',
           'split_document', 'VectorMemory']
