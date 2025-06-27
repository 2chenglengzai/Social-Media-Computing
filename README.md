# Social-Media-Computing

## Aspect-Based Sentiment Analysis of Online Product Reviews: A Comprehensive NLP Approach for Customer Trend Analysis

## 📱 Project Overview

This project implements a comprehensive sentiment analysis system for iPhone customer reviews using multiple machine learning approaches, aspect-based sentiment analysis (ABSA), and opinion mining techniques. The system analyzes customer sentiments across different product aspects and provides detailed insights into customer opinions.

## 🎯 Objectives

- **Sentiment Classification**: Classify reviews into positive, negative, and neutral sentiments
- **Aspect-Based Analysis**: Analyze sentiments for specific product aspects (battery, screen, performance, camera, design, price)
- **Opinion Mining**: Extract opinion targets and opinion words from reviews
- **Model Comparison**: Compare the performance of multiple ML algorithms
- **Visualization**: Provide a comprehensive visual analysis of results

## 🔧 Features

### Core Functionality
- ✅ **Text Preprocessing Pipeline**: Comprehensive text cleaning and normalization
- ✅ **Sentiment Classification**: Multi-class sentiment prediction
- ✅ **Aspect-Based Sentiment Analysis**: Sentiment analysis for specific product features
- ✅ **Opinion Mining**: Extraction of opinion targets and sentiment words
- ✅ **Model Comparison**: Performance evaluation of multiple algorithms
- ✅ **Visualization Suite**: Rich visualizations and analytics

### Advanced Features
- 🔍 **Dependency Parsing**: Advanced NLP using spaCy
- 📊 **Word Cloud Generation**: Visual representation of frequent terms
- 🎯 **Feature Importance Analysis**: Understanding model decision factors
- 📈 **Comprehensive EDA**: Detailed exploratory data analysis

## 🛠️ Technical Stack

### Core Libraries
```python
pandas                  
numpy
scikit-learn
nltk
spacy
textblob
```

### Visualization
```python
matplotlib
seaborn
wordcloud
```

### Machine Learning Models
```python
RandomForestClassifier 
LogisticRegression
SVM
```

### Deep Learning
```python
tensorflow/keras
transformers
```

## 📋 Requirements

### System Requirements
- Python 3.7+
- Jupyter Notebook or Google Colab
- Minimum 4GB RAM recommended

### Python Dependencies
```bash
pip install pandas numpy scikit-learn nltk spacy textblob
pip install matplotlib seaborn wordcloud
pip install sentence-transformers
pip install tensorflow
pip install transformers
```

### Additional Setup
```bash
# Download NLTK data
python -c "import nltk; nltk.download('all')"

# Download the spaCy model
python -m spacy download en_core_web_sm
```

## 🚀 Getting Started

### 1. Data Preparation
```python
# Place your iPhone reviews CSV file in the appropriate directory
# Expected format: columns for customer name, product, rating, review text, etc.
iphone_data = "path/to/your/iphone.csv"
```

### 2. Run the Notebook
1. Open the notebook in Jupyter or Google Colab
2. Update file paths in the data loading section
3. Execute cells sequentially
4. Monitor outputs and visualizations

### 3. Expected Outputs
- Cleaned dataset with preprocessed text
- Sentiment predictions and probabilities
- Aspect-based sentiment analysis results
- Model performance metrics
- Comprehensive visualizations

## 📊 Dataset Structure

### Input Data Format
```
Column Structure:
- reviewer_name: Customer name
- reviewer_id: Unique customer identifier
- product_name: iPhone model name
- product_id: Product identifier
- rating: Star rating (1-5)
- review_title: Review headline
- review_text: Full review content
- helpful_count: Number of helpful votes
- total_count: Total votes
- date: Review date
```

### Output Data Format
```
Generated Features:
- processed_text: Cleaned review text
- sentiment: Classified sentiment (positive/negative/neutral)
- word_tokens: Tokenized words
- aspects: Identified product aspects
- opinion_pairs: Opinion-target relationships
```

## 🔄 Workflow

### 1. Data Loading & Preprocessing
- Load iPhone reviews dataset
- Handle missing values and data cleaning
- Text normalization and preprocessing
- Feature engineering

### 2. Sentiment Labeling
```python
Rating → Sentiment Mapping:
4-5 stars → Positive
3 stars   → Neutral  
1-2 stars → Negative
```

### 3. Feature Extraction
- TF-IDF vectorization
- Word embeddings (optional)
- N-gram features

### 4. Model Training & Evaluation
- Train multiple ML models
- Cross-validation
- Performance evaluation
- Model comparison

### 5. Advanced Analysis
- Aspect-based sentiment analysis
- Opinion mining and extraction
- Visualization and reporting

## 📈 Results Summary

### Model Performance (Example)
```
Model Performance:
- Random Forest: 90.51% accuracy
- Logistic Regression: 90.08% accuracy
- SVM: 90.59% accuracy
```

### Aspect Analysis (Example)
```
Aspect Sentiment Distribution:
- Battery: 650 positive, 150 negative, 56 neutral
- Screen: 270 positive, 86 negative, 17 neutral
- Performance: 395 positive, 29 negative, 18 neutral
```

## 📁 File Structure

```
iPhone-Sentiment-Analysis/
│
├── notebook.ipynb
├── README.md
├── data/
│   ├── iphone.csv
│   └── iphone_cleaned.csv
├── outputs/
│   ├── predictions.csv
│   ├── comprehensive_analysis.png
│   └── model_results.json
└── requirements.txt   
```

## 🎯 Key Insights

### Sentiment Distribution
- **Positive Reviews**: ~82.7% of all reviews
- **Negative Reviews**: ~13.6% of all reviews  
- **Neutral Reviews**: ~3.7% of all reviews

### Top Aspects Mentioned
1. **Battery**: Most frequently discussed aspect
2. **Performance**: Generally positive feedback
3. **Screen**: Mixed but mostly positive sentiments
4. **Camera**: High satisfaction levels
5. **Design**: Positive reception
6. **Price**: Varied opinions

### Common Opinion Words
- **Positive**: excellent, amazing, great, perfect, love
- **Negative**: terrible, awful, disappointing, poor
- **Neutral**: okay, average, decent

## 🔍 Usage Examples

### Predicting Sentiment for New Reviews
```python
# Load trained model and vectorizer
new_review = "This iPhone has amazing camera quality but poor battery life"
prediction = model.predict(vectorizer.transform([new_review]))
print(f"Predicted sentiment: {prediction}")
```

### Aspect-Based Analysis
```python
# Analyze specific aspects
aspects = absa_analyzer.extract_aspects(new_review)
for aspect in aspects:
    sentiment = absa_analyzer.analyze_aspect_sentiment(new_review, aspect)
    print(f"{aspect}: {sentiment}")
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is for educational and research purposes.

## 👥 Authors

- [Your Name] - Final Year Project / CDS-6344 Assignment

## 🙏 Acknowledgments

- Dataset: Flipkart iPhone Reviews
- Libraries: scikit-learn, NLTK, spaCy, pandas
- Course: CDS-6344 (Data Science/NLP Course)

## 📧 Contact

For questions or support, please contact [your.email@domain.com]

---

**Note**: This project demonstrates comprehensive sentiment analysis techniques including traditional ML, aspect-based analysis, and opinion mining for educational purposes.
