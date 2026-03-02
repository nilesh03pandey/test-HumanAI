import nltk

# Ensure required NLTK resources are available
try:
    nltk.data.find('corpora/stopwords')
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('stopwords')
    nltk.download('punkt')



from gnews import GNews


from sentence_transformers import SentenceTransformer, util
from rake_nltk import Rake
import numpy as np

class MediaSpikeController:
    def __init__(self):
        # Initialize GNews for step 2 [cite: 147]
        self.google_news = GNews(language='en', country='US', period='7d', max_results=10)
        
        # Initialize SBERT for step 3 [cite: 148]
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Rake for entity/keyword extraction
        self.rake = Rake()

    def step_1_entity_extraction(self, social_media_text):

        """Identify top recurring keywords or nouns."""

        self.rake.extract_keywords_from_text(social_media_text)
        # Get top 5 keywords [cite: 151, 152]
        keywords = self.rake.get_ranked_phrases()[:5]
        return keywords

    def step_2_fetch_context(self, keywords):

        """Query GNews for keywords to find national/state headlines that may be driving local spikes."""

        all_headlines = []
        for kw in keywords:
            news_results = self.google_news.get_news(kw)
            headlines = [item['title'] for item in news_results]
            all_headlines.extend(headlines)
        return list(set(all_headlines)) # Remove duplicates
    


    def step_3_calculate_correlation(self, county_text, news_headlines):

        """Calculate semantic similarity (R²) using Cosine Similarity"""

        if not news_headlines:
            return 0.0

        # Encode the county's social media text and the news headlines into vectors

        county_vec = self.model.encode(county_text, convert_to_tensor=True)
        news_vecs = self.model.encode(news_headlines, convert_to_tensor=True)

        # Calculate cosine similarity between vectors

        cosine_scores = util.cos_sim(county_vec, news_vecs)
        
        # Take the maximum similarity score as the correlation indicator (R²) 

        max_correlation = float(np.max(cosine_scores.cpu().numpy()))
        return max_correlation
    


    def run_media_check(self, county_social_media_batch):

        """Executes the full logic flow"""

        # Step 1: Extract Keywords
        keywords = self.step_1_entity_extraction(county_social_media_batch)
        print(f"Extracted Keywords: {keywords}")

        # Step 2: Fetch News Context
        headlines = self.step_2_fetch_context(keywords)
        print(f"Fetched {len(headlines)} related news headlines.")

        # Step 3: Correlation Calculation
        correlation_score = self.step_3_calculate_correlation(county_social_media_batch, headlines)
        
        # Determine if it is a media-driven spike 

        is_media_driven = correlation_score > 0.70 
        return is_media_driven, correlation_score



# Example Usage
if __name__ == "__main__":
    controller = MediaSpikeController()
    
    # Simulated high-sentiment text from a county
    sample_text = "Heartbroken by the sudden loss of the local community leader. A huge tribute to his life."
    
    is_spike, score = controller.run_media_check(sample_text)
    print(f"Media Correlation Score: {score:.2f}")
    print(f"Is this a media-driven false positive? {is_spike}")