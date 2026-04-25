import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import Tk, Label, Entry, Button
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# -----------------------------
# STEP 1: Load and Clean Data
# -----------------------------
df = pd.read_excel('C:/Users/gaura/OneDrive/Desktop/learning/NEILIT Data Analytics using python/project/lens_drones1.xlsm')

# Drop rows with invalid/NA values in critical columns
critical_cols = ['publication_date', 'application_date', 'cited_count', 'title',
                 'simple_family_size', 'type', 'priority_numbers']
df = df.dropna(subset=critical_cols)

# Convert date columns to datetime
df['publication_date'] = pd.to_datetime(df['publication_date'], errors='coerce')
df['application_date'] = pd.to_datetime(df['application_date'], errors='coerce')
df = df.dropna(subset=['publication_date', 'application_date'])

# -----------------------------
# STEP 2: Extract Origin Country
# -----------------------------
origin_codes = ['FR','KR','DE','EP','JP','GB','CN','IT','IL','IN','RU','CA','CH']

def extract_country(priority_numbers):
    for code in origin_codes:
        if str(priority_numbers).startswith(code):
            return code
    return None

df['origin_country'] = df['priority_numbers'].apply(extract_country)
df = df.dropna(subset=['origin_country'])

# -----------------------------
# STEP 3: Sampling Strategy
# -----------------------------
types = ['Patent Application', 'Granted Patent', 'Amended Patent',
         'Limited Patent', 'Search report', 'Statutory Invention Registration']

sampled_df = pd.DataFrame()

for t in types:
    for c in origin_codes:
        subset = df[(df['type'] == t) & (df['origin_country'] == c)]
        if len(subset) > 0:
            n = min(len(subset), 15)  # sample up to 15 per combination
            sampled_df = pd.concat([sampled_df, subset.sample(n=n, random_state=42)])

print("Final sample size:", len(sampled_df))

# -----------------------------
# STEP 4: Duration Calculation
# -----------------------------
sampled_df['duration_days'] = (sampled_df['publication_date'] - sampled_df['application_date']).dt.days

avg_duration = sampled_df.groupby('type')['duration_days'].mean().reset_index()

# -----------------------------
# STEP 5: Line Plot (Duration vs Type)
# -----------------------------
plt.figure(figsize=(10,6))
plt.plot(avg_duration['type'], avg_duration['duration_days'], marker='o')
plt.xticks(rotation=45)
plt.ylabel("Average Duration (days)")
plt.title("Average Publication Duration by Patent Type")
plt.tight_layout()
plt.show()

# -----------------------------
# STEP 6: Prediction with KNN
# -----------------------------
# Encode patent type as categorical numbers
type_mapping = {t:i for i,t in enumerate(types)}
sampled_df['type_encoded'] = sampled_df['type'].map(type_mapping)

X = sampled_df[['type_encoded']]
y = sampled_df['duration_days']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)

# -----------------------------
# STEP 7: Tkinter GUI Prediction
# -----------------------------
def predict_duration():
    user_input = entry.get()
    if user_input in type_mapping:
        encoded = np.array([[type_mapping[user_input]]])
        pred = knn.predict(encoded)[0]
        result_label.config(text=f"Predicted duration: {int(pred)} days")
    else:
        result_label.config(text="Invalid type entered.")

root = Tk()
root.title("Patent Duration Predictor")

Label(root, text="Enter Patent Type:").pack()
entry = Entry(root)
entry.pack()

Button(root, text="Predict", command=predict_duration).pack()
result_label = Label(root, text="")
result_label.pack()

root.mainloop()

# -----------------------------
# STEP 8: Jurisdiction Scatter Plot
from sklearn.preprocessing import LabelEncoder

# --- Scatter Plot ---
# Count how many origin_country observations are linked to each jurisdiction
juris_counts = df.groupby('jurisdiction')['origin_country'].count()

# Rescale bubble sizes
max_size = 800
sizes = (juris_counts.values / juris_counts.values.max()) * max_size

# Plot
plt.figure(figsize=(6,4))
plt.scatter(juris_counts.index, [1]*len(juris_counts), s=sizes, alpha=0.6)
for juris, count in juris_counts.items():
    plt.text(juris, 1.05, str(count), ha='center')
plt.xticks(juris_counts.index)
plt.yticks([])
plt.title("Jurisdiction Impact (dot size = origin_country count)")
plt.show()

# --- KNN Classifier ---
# Encode origin_country and jurisdiction
encoder_origin = LabelEncoder()
encoder_juris = LabelEncoder()

df['origin_encoded'] = encoder_origin.fit_transform(df['origin_country'])
y = encoder_juris.fit_transform(df['jurisdiction'])
X = df[['origin_encoded']]

# Train/test split and fit
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

# --- Query block for probability prediction ---
origin_input = "FR"   # change this to any origin_country you want to test
origin_encoded = encoder_origin.transform([origin_input])
proba = knn.predict_proba([origin_encoded])

for juris, p in zip(encoder_juris.classes_, proba[0]):
    print(f"{origin_input} → {juris}: {p:.2f}")









