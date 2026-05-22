# ============================================
# CUSTOMER SUPPORT TICKET CLASSIFIER
# Task 2 - Future Interns ML Internship
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from nltk.corpus import stopwords
import re
import warnings
warnings.filterwarnings('ignore')

# ============================================
# STEP 1: SIMULATED TICKET DATA
# ============================================

tickets_data = [
    # Technical - High
    ("My system is completely down and I cannot access anything", "Technical", "High"),
    ("Server crashed and all services are unavailable", "Technical", "High"),
    ("Critical bug causing data loss in production", "Technical", "High"),
    ("Application is not loading at all", "Technical", "High"),
    ("Database connection failed completely", "Technical", "High"),
    ("Website is down for all users", "Technical", "High"),
    ("Cannot login to the system urgent help needed", "Technical", "High"),
    ("Payment processing system is broken", "Technical", "High"),
    ("All data has been lost after system crash", "Technical", "High"),
    ("Production server is not responding", "Technical", "High"),
    ("Critical security vulnerability found in system", "Technical", "High"),
    ("Complete system failure affecting all employees", "Technical", "High"),
    ("API is down and causing business loss", "Technical", "High"),
    ("Network is completely unavailable for all users", "Technical", "High"),
    ("Emergency system outage needs immediate fix", "Technical", "High"),
    # Technical - Medium
    ("My computer keeps crashing every few minutes", "Technical", "Medium"),
    ("Software installation failed with error code", "Technical", "Medium"),
    ("App is running very slow and lagging badly", "Technical", "Medium"),
    ("Getting error message when trying to save files", "Technical", "Medium"),
    ("Email client not syncing properly", "Technical", "Medium"),
    ("Video calls keep dropping during meetings", "Technical", "Medium"),
    ("File upload feature not working correctly", "Technical", "Medium"),
    ("Getting timeout errors frequently", "Technical", "Medium"),
    ("Dashboard not loading all data properly", "Technical", "Medium"),
    ("Notifications not appearing on my device", "Technical", "Medium"),
    # Technical - Low
    ("Printer not connecting to my computer", "Technical", "Low"),
    ("How do I update my software to latest version", "Technical", "Low"),
    ("Where can I find the settings menu", "Technical", "Low"),
    ("Need help configuring email settings", "Technical", "Low"),
    ("How do I enable dark mode in the app", "Technical", "Low"),
    ("Where is the keyboard shortcut for copy paste", "Technical", "Low"),
    ("How do I resize the application window", "Technical", "Low"),
    ("Can you help me install the browser extension", "Technical", "Low"),
    ("How do I backup my data regularly", "Technical", "Low"),
    ("Need guidance on using the search feature", "Technical", "Low"),
    # Billing - High
    ("I was charged twice for the same order", "Billing", "High"),
    ("Unauthorized transaction on my account", "Billing", "High"),
    ("Wrong amount deducted from my bank account", "Billing", "High"),
    ("My refund has not been processed after 30 days", "Billing", "High"),
    ("Subscription renewed without my permission", "Billing", "High"),
    ("Fraudulent charge appeared on my credit card", "Billing", "High"),
    ("I was billed for a cancelled subscription", "Billing", "High"),
    ("Double payment taken from my account today", "Billing", "High"),
    ("Incorrect charge of large amount on my card", "Billing", "High"),
    ("Refund promised 2 weeks ago still not received", "Billing", "High"),
    ("Charged full price despite cancellation request", "Billing", "High"),
    ("Payment deducted but order not confirmed", "Billing", "High"),
    # Billing - Medium
    ("Invoice shows incorrect amount", "Billing", "Medium"),
    ("Need copy of my last payment receipt", "Billing", "Medium"),
    ("How do I upgrade my subscription plan", "Billing", "Medium"),
    ("Want to cancel my subscription plan", "Billing", "Medium"),
    ("Need to update my billing address", "Billing", "Medium"),
    ("Can I switch from monthly to annual billing", "Billing", "Medium"),
    ("Tax invoice not generated for last payment", "Billing", "Medium"),
    ("Need breakdown of charges on my invoice", "Billing", "Medium"),
    ("Promo code not applied to my account", "Billing", "Medium"),
    ("Discount not reflecting on my bill", "Billing", "Medium"),
    # Billing - Low
    ("When is my next billing date", "Billing", "Low"),
    ("How do I update my payment method", "Billing", "Low"),
    ("Can I get a discount on annual plan", "Billing", "Low"),
    ("What payment methods do you accept", "Billing", "Low"),
    ("How do I download my invoice", "Billing", "Low"),
    ("What is included in the basic plan", "Billing", "Low"),
    ("How do I view my billing history", "Billing", "Low"),
    ("Is there a free trial available", "Billing", "Low"),
    ("What happens after my trial ends", "Billing", "Low"),
    ("How do I apply a coupon code", "Billing", "Low"),
    # General - Medium
    ("How do I export all my account data", "General", "Medium"),
    ("Need help understanding the analytics dashboard", "General", "Medium"),
    ("How do I add team members to my account", "General", "Medium"),
    ("Can I integrate with third party tools", "General", "Medium"),
    ("How do I enable two factor authentication", "General", "Medium"),
    ("Want to migrate data from old account", "General", "Medium"),
    ("How do I set up automated workflows", "General", "Medium"),
    ("Need help with bulk import of records", "General", "Medium"),
    ("How do I configure user permissions", "General", "Medium"),
    ("Want to set up custom notifications", "General", "Medium"),
    # General - Low
    ("How do I reset my password", "General", "Low"),
    ("Where can I find user documentation", "General", "Low"),
    ("What are your business hours", "General", "Low"),
    ("How do I contact customer support", "General", "Low"),
    ("Can you explain the features of premium plan", "General", "Low"),
    ("How do I change my account settings", "General", "Low"),
    ("I want to update my profile information", "General", "Low"),
    ("What is the maximum file upload size", "General", "Low"),
    ("Is there a mobile app available", "General", "Low"),
    ("How do I delete my account permanently", "General", "Low"),
    ("Where can I find video tutorials", "General", "Low"),
    ("How do I change my username", "General", "Low"),
    ("What languages does the app support", "General", "Low"),
    ("How do I logout from all devices", "General", "Low"),
    ("Can I use the service on multiple devices", "General", "Low"),
    ("How do I subscribe to newsletter", "General", "Low"),
    ("Where do I find terms and conditions", "General", "Low"),
    ("How do I report a problem with the app", "General", "Low"),
    ("Can I share my account with others", "General", "Low"),
    ("How do I change notification preferences", "General", "Low"),
]

df = pd.DataFrame(tickets_data, columns=['Ticket', 'Category', 'Priority'])

print("=" * 55)
print("CUSTOMER SUPPORT TICKET CLASSIFIER")
print("=" * 55)
print(f"Total Tickets: {len(df)}")
print(f"\nCategory Distribution:")
print(df['Category'].value_counts().to_string())
print(f"\nPriority Distribution:")
print(df['Priority'].value_counts().to_string())

# ============================================
# STEP 2: TEXT PREPROCESSING
# ============================================

stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words]
    return ' '.join(tokens)

df['Cleaned_Ticket'] = df['Ticket'].apply(clean_text)

print("\n" + "=" * 55)
print("TEXT PREPROCESSING COMPLETE")
print("=" * 55)
print("Sample cleaned tickets:")
for i in range(3):
    print(f"  Original : {df['Ticket'].iloc[i]}")
    print(f"  Cleaned  : {df['Cleaned_Ticket'].iloc[i]}")
    print()

# ============================================
# STEP 3: TRAIN MODELS
# ============================================

X = df['Cleaned_Ticket']
y_category = df['Category']
y_priority = df['Priority']

X_train, X_test, y_cat_train, y_cat_test = train_test_split(
    X, y_category, test_size=0.2, random_state=42, stratify=y_category
)
_, _, y_pri_train, y_pri_test = train_test_split(
    X, y_priority, test_size=0.2, random_state=42, stratify=y_priority
)

category_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
    ('clf', LogisticRegression(random_state=42, max_iter=1000, C=5))
])
category_pipeline.fit(X_train, y_cat_train)
cat_pred = category_pipeline.predict(X_test)
cat_accuracy = accuracy_score(y_cat_test, cat_pred)

priority_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
    ('clf', LogisticRegression(random_state=42, max_iter=1000, C=5))
])
priority_pipeline.fit(X_train, y_pri_train)
pri_pred = priority_pipeline.predict(X_test)
pri_accuracy = accuracy_score(y_pri_test, pri_pred)

print("=" * 55)
print("CATEGORY CLASSIFIER RESULTS:")
print("=" * 55)
print(f"Accuracy: {cat_accuracy * 100:.2f}%")
print(classification_report(y_cat_test, cat_pred))

print("=" * 55)
print("PRIORITY CLASSIFIER RESULTS:")
print("=" * 55)
print(f"Accuracy: {pri_accuracy * 100:.2f}%")
print(classification_report(y_pri_test, pri_pred))

# ============================================
# STEP 4: LIVE TICKET PREDICTION
# ============================================

def classify_ticket(ticket_text):
    cleaned = clean_text(ticket_text)
    category = category_pipeline.predict([cleaned])[0]
    priority = priority_pipeline.predict([cleaned])[0]
    return category, priority

print("=" * 55)
print("LIVE TICKET CLASSIFICATION TEST:")
print("=" * 55)

test_tickets = [
    "My account has been hacked and I need immediate help",
    "I was charged double for my monthly subscription",
    "How do I change my profile picture",
    "The entire system is down and we are losing money",
    "Can you send me the invoice for last month",
    "Website not loading for any of our customers",
    "How do I reset my password",
    "Refund not received after 3 weeks",
]

for ticket in test_tickets:
    category, priority = classify_ticket(ticket)
    print(f"Ticket  : {ticket}")
    print(f"Result  : Category={category} | Priority={priority}")
    print()

# ============================================
# STEP 5: VISUALIZATION DASHBOARD
# ============================================

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('Support Ticket Classification Dashboard',
             fontsize=16, fontweight='bold')
fig.patch.set_facecolor('#F8F9FA')

# Chart 1: Category Distribution
ax1 = axes[0, 0]
ax1.set_facecolor('#FFFFFF')
cat_counts = df['Category'].value_counts()
colors1 = ['#2196F3', '#4CAF50', '#FF9800']
bars1 = ax1.bar(cat_counts.index, cat_counts.values, color=colors1,
                edgecolor='white', linewidth=1.5)
ax1.set_title('Tickets by Category', fontweight='bold', fontsize=12)
ax1.set_xlabel('Category')
ax1.set_ylabel('Number of Tickets')
ax1.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars1, cat_counts.values):
    ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
             str(val), ha='center', va='bottom', fontweight='bold')

# Chart 2: Priority Distribution
ax2 = axes[0, 1]
ax2.set_facecolor('#FFFFFF')
pri_counts = df['Priority'].value_counts()
colors2 = ['#F44336', '#FF9800', '#4CAF50']
ax2.pie(pri_counts.values, labels=pri_counts.index, colors=colors2,
        autopct='%1.1f%%', startangle=90,
        wedgeprops=dict(edgecolor='white', linewidth=2))
ax2.set_title('Priority Distribution', fontweight='bold', fontsize=12)

# Chart 3: Category vs Priority Heatmap
ax3 = axes[1, 0]
ax3.set_facecolor('#FFFFFF')
heatmap_data = pd.crosstab(df['Category'], df['Priority'])
heatmap_data = heatmap_data.reindex(columns=['High', 'Medium', 'Low'])
im = ax3.imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto')
ax3.set_xticks(range(len(heatmap_data.columns)))
ax3.set_yticks(range(len(heatmap_data.index)))
ax3.set_xticklabels(heatmap_data.columns)
ax3.set_yticklabels(heatmap_data.index)
ax3.set_title('Category vs Priority Heatmap', fontweight='bold', fontsize=12)
for i in range(len(heatmap_data.index)):
    for j in range(len(heatmap_data.columns)):
        ax3.text(j, i, heatmap_data.values[i, j],
                ha='center', va='center', fontweight='bold', fontsize=14)
plt.colorbar(im, ax=ax3)

# Chart 4: Model Accuracy
ax4 = axes[1, 1]
ax4.set_facecolor('#FFFFFF')
models = ['Category\nClassifier', 'Priority\nClassifier']
accuracies = [cat_accuracy * 100, pri_accuracy * 100]
colors4 = ['#3F51B5', '#E91E63']
bars4 = ax4.bar(models, accuracies, color=colors4,
                edgecolor='white', linewidth=1.5, width=0.4)
ax4.set_title('Model Accuracy Comparison', fontweight='bold', fontsize=12)
ax4.set_ylabel('Accuracy (%)')
ax4.set_ylim(0, 110)
ax4.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars4, accuracies):
    ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
             f'{val:.1f}%', ha='center', va='bottom',
             fontweight='bold', fontsize=12)

plt.tight_layout()
plt.savefig('ticket_classification_dashboard.png', dpi=150,
            bbox_inches='tight', facecolor='#F8F9FA')
plt.show()

print("=" * 55)
print("Dashboard saved: ticket_classification_dashboard.png")
print("=" * 55)
print("\nBUSINESS INSIGHTS:")
print(f"  Total Tickets Analyzed : {len(df)}")
print(f"  Category Model Accuracy: {cat_accuracy*100:.2f}%")
print(f"  Priority Model Accuracy: {pri_accuracy*100:.2f}%")
print(f"  Most Common Category   : {df['Category'].value_counts().index[0]}")
print(f"  High Priority Tickets  : {len(df[df['Priority']=='High'])}")
print("\nProject Complete!")
