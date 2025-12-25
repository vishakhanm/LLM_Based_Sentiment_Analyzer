import numpy as np
from constants import (
    FEED_SEVERITY_METHOD,
    FEED_TOP_K_RATIO,
    FEED_SEVERITY_THRESHOLD,
)

def aggregate_feed_severity(post_scores):
    """
    post_scores: List[float] in [0, 1]
    returns: (feed_severity, is_flagged)
    """
    if not post_scores:
        return 0.0, False

    scores = np.array(post_scores)

    if FEED_SEVERITY_METHOD == "mean":
        feed_severity = scores.mean()

    elif FEED_SEVERITY_METHOD == "max":
        feed_severity = scores.max()

    elif FEED_SEVERITY_METHOD == "top_k_mean":
        k = max(1, int(len(scores) * FEED_TOP_K_RATIO))
        feed_severity = np.sort(scores)[-k:].mean()

    else:
        raise ValueError("Invalid FEED_SEVERITY_METHOD")

    is_flagged = feed_severity >= FEED_SEVERITY_THRESHOLD
    return float(feed_severity), is_flagged