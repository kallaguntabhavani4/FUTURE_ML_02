# 🎫 Customer Support Ticket Classifier

## Project Overview
An NLP-based Machine Learning system that automatically classifies
customer support tickets into categories and assigns priority levels.
Built using Python with Logistic Regression achieving 80% accuracy.

## Business Problem
Support teams waste time manually reading and sorting hundreds of tickets.
This system automatically classifies tickets so agents can focus on
solving problems instead of sorting them.

## Tools Used
- Python, Pandas, NumPy
- Scikit-learn (Logistic Regression, TF-IDF)
- NLTK (Text Preprocessing)
- Matplotlib (Visualization)

## Model Performance
| Metric | Value |
|--------|-------|
| Category Accuracy | 80% |
| Total Tickets | 97 |
| Training Data | 80% |
| Test Data | 20% |

## Categories & Priority Levels
| Category | Priority |
|----------|----------|
| Technical | High / Medium / Low |
| Billing | High / Medium / Low |
| General | High / Medium / Low |

## Key Features
- Text cleaning and stopword removal
- TF-IDF vectorization with bigrams
- Automatic category classification
- Priority level assignment
- Business-friendly dashboard with 4 charts

## Business Impact
- **Support Manager**: Route tickets to right team automatically
- **Agent**: Focus on High priority tickets first
- **Business**: Reduce response time and improve customer satisfaction

## Deliverables
- `ticket_classifier.py` — Complete NLP classification model
- `ticket_classification_dashboard.png` — 4-chart visual dashboard
- `README.md` — Business explanation

## Author
BHAVANI | Future Interns Machine Learning Internship
