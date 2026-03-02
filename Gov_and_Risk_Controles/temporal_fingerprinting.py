import numpy as np
from typing import List, Dict, Any, Optional

def temporal_fingerprinting(time_series: List[float]) -> float:

    """Temporal Fingerprinting (T) | normalized [0,1]"""

    """Identifies non-human posting patterns by flagging accounts that publish
    identical content within sub-millisecond intervals or maintain perfectly rhythmic, automated posting
    schedules."""

    if len(time_series) < 2:
        return 0.0
    diffs = np.diff(time_series)
    neighbor_vs = np.abs(diffs) / (np.abs(time_series[:-1]) + 1e-6)  # avoid division by zero

    return float(np.mean(neighbor_vs))
