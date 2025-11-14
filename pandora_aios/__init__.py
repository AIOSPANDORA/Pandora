"""
Pandora AIOS - An AI Computer Operating System

A lightweight, AI-powered operating system framework that demonstrates
core OS concepts enhanced with AI capabilities.
"""

__version__ = "0.1.0"
__author__ = "Pandora AIOS Team"

from .kernel import Kernel
from .ai_engine import AIEngine
from .shell import Shell
from .filesystem import FileSystem

__all__ = ["Kernel", "AIEngine", "Shell", "FileSystem"]
