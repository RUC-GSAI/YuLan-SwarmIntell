

import json
import os
import sys
import time
import math
from collections import defaultdict, Counter
import re
import numpy as np
from colorama import init, Fore, Back, Style
from utils.constants import *

#  ANSI Code Stripping Regex
ansi_escape_pattern = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

#  Helper Functions for Metrics
def shannon_entropy(data):
    if not data: return 0.0
    counts = Counter(data)
    total_count = len(data)
    probabilities = [count / total_count for count in counts.values() if count > 0]
    if not probabilities: return 0.0
    return -sum(p * math.log2(p) for p in probabilities)

def manhattan_distance(pos1, pos2):
    """Calculate Manhattan distance between two positions"""
    return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])


def get_visual_width(text):
    return len(ansi_escape_pattern.sub('', text))

def pad_visual_width(text, width):
    current_visual_width = get_visual_width(text)
    padding = max(0, width - current_visual_width)
    return text + ' ' * padding
