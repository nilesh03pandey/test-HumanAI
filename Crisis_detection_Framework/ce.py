import numpy as np

class ConfidenceEngine:
    def __init__(self, k_constant=50):

        """
        k_constant: The 'smoothing' factor from your proposal. 
        Higher k means you need more users to reach high confidence.
        """
        self.k = k_constant

    def calculate_ce(self, n_users, sentiment_scores, bot_probability):

        """
        Implements the CE logic from the ISSR Selection Task.
        """
        # 1. Sample Size Penalty (Density Factor)
        # Prevents high confidence in rural areas with very few users
        density_factor = n_users / (n_users + self.k)

        # 2. Sentiment Variance (Stability)
        # If one person is very happy and another is very sad, Var is high, CE drops.
        if len(sentiment_scores) > 1:
            variance_s = np.var(sentiment_scores)
        else:
            variance_s = 1.0  # Maximum uncertainty for a single data point

        # 3. Noise Factor (Signal Purity)
        # Directly uses the bot/media detection results
        noise_factor = bot_probability 

        # 4. Final CE Calculation
        ce = density_factor * (1 - variance_s) * (1 - noise_factor)
        
        return max(0, round(ce, 4)) # Ensure it doesn't go below 0

# --- Integration Example ---
if __name__ == "__main__":
    engine = ConfidenceEngine(k_constant=50)

    # Scenario A: Rural County (Low N, but consistent sentiment)
    # 5 users, all fairly sad (0.8), no bots
    
    ce_rural = engine.calculate_ce(n_users=5, sentiment_scores=[0.8, 0.82, 0.79, 0.81, 0.8], bot_probability=0.05)

    # Scenario B: Urban County (High N, but high disagreement/noise)
    # 200 users, wildly different feelings, 20% bot activity suspected

    ce_urban = engine.calculate_ce(n_users=200, sentiment_scores=np.random.rand(200), bot_probability=0.20)

    print(f"Rural Confidence: {ce_rural} (Limited by sample size)")
    print(f"Urban Confidence: {ce_urban} (Limited by variance/noise)")