import numpy as np
import json
import datetime
import nltk
from gnews import GNews
from sentence_transformers import SentenceTransformer, util
from rake_nltk import Rake

def setup_nlp_tools():
    for res in ['stopwords', 'punkt', 'punkt_tab']:
        try:
            nltk.data.find(f'tokenizers/{res}' if 'punkt' in res else f'corpora/{res}')
        except LookupError:
            nltk.download(res)

setup_nlp_tools()

class AI4MH_LogicLayer:
    def __init__(self):
        self.news_client = GNews(language='en', country='US', period='7d', max_results=5)
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        self.keyword_extractor = Rake()
        self.k_density = 50 # Smoothing constant from your proposal

    # 1. SCORING ENGINE
    def calculate_metrics(self, sentiment, volume, geo, n_users, var_s, noise):

        # Crisis Index (CI)

        ci = (0.5 * sentiment) + (0.3 * volume) + (0.2 * geo)
        
        # Confidence Estimate (CE)
        density = n_users / (n_users + self.k_density)
        ce = density * (1 - var_s) * (1 - noise)
        
        return round(ci, 2), round(ce, 2)

    # 2. MEDIA SPIKE DETECTION (GNews + SBERT)
    def detect_media_spike(self, county_text):
        self.keyword_extractor.extract_keywords_from_text(county_text)
        keywords = self.keyword_extractor.get_ranked_phrases()[:3]
        
        headlines = []
        for kw in keywords:
            results = self.news_client.get_news(kw)
            headlines.extend([r['title'] for r in results])
        
        if not headlines: return 0.0
        
        # Semantic Correlation
        county_emb = self.embedder.encode(county_text, convert_to_tensor=True)
        news_embs = self.embedder.encode(headlines, convert_to_tensor=True)
        correlation = float(util.cos_sim(county_emb, news_embs).max())
        
        return round(correlation, 2)

    # 3. AUDIT LOGGING
    def log_decision(self, county_id, ci, ce, media_corr, bot_flag):
        log = {
            "log_id": f"AL-2026-{np.random.randint(1000, 9999)}",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "county": county_id,
            "metrics": {"crisis_index": ci, "confidence": ce},
            "risk_flags": {"media_correlation": media_corr, "bot_detected": bot_flag},
            "escalation": "High" if ci >= 0.8 else "Medium" if ci >= 0.6 else "Low"
        }
        print(f"Generated Audit Log: {json.dumps(log, indent=2)}")
        return log


# EXAMPLE
if __name__ == "__main__":
    logic = AI4MH_LogicLayer()
    
    # Simulated Input for a 72-hour window
    county_data = {
        "id": "County_A",
        "text": "Extremely worried about the plant closing. Many people are feeling hopeless today.",
        "stats": {"s": 0.85, "v": 0.90, "g": 0.40, "n": 12, "var": 0.1, "noise": 0.05}
    }

    # Step 1: Check for Media Spikes
    m_corr = logic.detect_media_spike(county_data["text"])
    
    # Step 2: Calculate Scores
    ci, ce = logic.calculate_metrics(
        county_data["stats"]["s"], county_data["stats"]["v"], 
        county_data["stats"]["g"], county_data["stats"]["n"],
        county_data["stats"]["var"], county_data["stats"]["noise"]
    )
    
    # Step 3: Apply Governance Penalty (if media spike > 0.75)
    if m_corr > 0.75: ci *= 0.5 

    # Step 4: Finalize Audit
    logic.log_decision(county_data["id"], ci, ce, m_corr, False)