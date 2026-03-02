import numpy as np


"""still need to work on this, but here is the basic structure for the Bayesian Smoothing component of the system. 
It takes in local sentiment and sample size, and applies a smoothing formula to produce a more stable sentiment score, 
especially for counties with low social media activity.
"""

class DataStabilizer:
    def __init__(self, state_baseline_sentiment=0.4):
        """
        state_baseline_sentiment: The average 'background' sentiment 
        across all of Alabama (e.g., 0.4 on a 0-1 scale).
        """
        self.baseline = state_baseline_sentiment
        # 'm' is the strength of our prior belief. 
        # A value of 10 means we need at least 10 posts to fully trust local data.
        self.m_constant = 10 

    def apply_bayesian_smoothing(self, local_sentiment, sample_size):

        """ Formula used: (local_sentiment * N + baseline * M) / (N + M) """

        if sample_size == 0:
            return self.baseline
            
        smoothed_score = (
            (local_sentiment * sample_size) + (self.baseline * self.m_constant)
        ) / (sample_size + self.m_constant)
        
        return round(smoothed_score, 4)




# Example
if __name__ == "__main__":
    stabilizer = DataStabilizer(state_baseline_sentiment=0.4)

    # SCENARIO 1: A rural county with only 2 posts, both very negative (0.9)
    # Without smoothing: 0.90
    rural_raw = 0.9
    rural_n = 2
    rural_smooth = stabilizer.apply_bayesian_smoothing(rural_raw, rural_n)

    # SCENARIO 2: An urban county (Birmingham) with 500 posts, average 0.9
    # Without smoothing: 0.90
    urban_raw = 0.9
    urban_n = 500
    urban_smooth = stabilizer.apply_bayesian_smoothing(urban_raw, urban_n)

    print(f"RURAL (N={rural_n}): Raw={rural_raw} -> Smoothed={rural_smooth}")
    print(f"URBAN (N={urban_n}): Raw={urban_raw} -> Smoothed={urban_smooth}")