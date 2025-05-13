


from flask import Flask, request, jsonify, send_file
import pandas as pd
from review_clustering import ReviewClustering
import os
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer

nltk.download('averaged_perceptron_tagger_eng')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    
try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')
    
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    
try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

app = Flask(__name__)
clusterer = ReviewClustering()

@app.route('/cluster_reviews', methods=['POST'])
def cluster_reviews():
    try:
        data = request.get_json()

        if not data or 'reviews' not in data:
            return jsonify({"error": "Request must contain a 'reviews' field"}), 400

        reviews = data['reviews']

        # Convert to DataFrame
        df = pd.DataFrame(reviews)
        if 'reviewText' not in df.columns:
            return jsonify({"error": "'reviewText' field is required in each review"}), 400

        # Assign dummy ASIN for internal compatibility
        df["asin"] = "DUMMY_ASIN"
        print(df.head())
        # Run clustering
        clustered_df, topic_words, coherence = clusterer.cluster_reviews(df, "DUMMY_ASIN")

        # Return visualization
        html_path = "./clustering_insights.html"
        if not os.path.exists(html_path):
            return jsonify({"error": "Visualization file not found"}), 500

        return send_file(html_path, as_attachment=True)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
