import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import classification_report, accuracy_score
import re
import ast

# Load the datasets
train_data_path = 'Training (1).csv'
medications_data_path = 'medications.csv'
diets_data_path = 'diets.csv'
workout_data_path = 'workout_df.csv'

medications_data = pd.read_csv(medications_data_path)
train_data = pd.read_csv(train_data_path)
workout_data = pd.read_csv(workout_data_path)
diets_data = pd.read_csv(diets_data_path)

print("Dataset Info:")
print(f"Training data shape: {train_data.shape}")
print(f"Unique diseases in training: {train_data['prognosis'].nunique()}")
print(f"Disease distribution:")
print(train_data['prognosis'].value_counts().head(10))
print(f"Medications data shape: {medications_data.shape}")
print(f"Diets data shape: {diets_data.shape}")
print(f"Workouts data shape: {workout_data.shape}")

# Function to safely parse string representations of lists
def parse_list_string(list_string):
    """Convert string representation of list to actual list"""
    try:
        return ast.literal_eval(list_string)
    except (ValueError, SyntaxError):
        if isinstance(list_string, str):
            cleaned = re.sub(r"[\[\]']", "", list_string)
            return [item.strip() for item in cleaned.split(',')]
        return []

# Function to flatten list strings for TF-IDF
def parse_and_flatten_list_string(list_string):
    """Convert string representation of list to flattened text for TF-IDF"""
    try:
        parsed_list = ast.literal_eval(list_string)
        return ' '.join(parsed_list)
    except (ValueError, SyntaxError):
        if isinstance(list_string, str):
            cleaned = re.sub(r"[\[\]']", "", list_string)
            items = [item.strip() for item in cleaned.split(',')]
            return ' '.join(items)
        return ""

# Prepare data for TF-IDF vectorization
# medications_data['Medication_Text'] = medications_data['Medication'].apply(parse_and_flatten_list_string)
# Prepare data for TF-IDF vectorization with augmented disease names
medications_data['Medication_Text'] = medications_data['Medication'].apply(parse_and_flatten_list_string)
medications_data['Combined_Text'] = medications_data['Disease'] + " " + medications_data['Medication_Text']

diets_data['Diet_Text'] = diets_data['Diet'].apply(parse_and_flatten_list_string)
diets_data['Combined_Text'] = diets_data['Disease'] + " " + diets_data['Diet_Text']

workout_data['Workout_Text'] = workout_data['workout'].astype(str)
workout_data['Combined_Text'] = workout_data['disease'] + " " + workout_data['Workout_Text']


# print("\nSample processed data:")
# print("Medication text sample:", medications_data['Medication_Text'].iloc[0])
# print("Diet text sample:", diets_data['Diet_Text'].iloc[0])
# print("Workout text sample:", workout_data['Workout_Text'].iloc[0])
print("\nSample processed data:")
print("Medication combined text sample:", medications_data['Combined_Text'].iloc[0])
print("Diet combined text sample:", diets_data['Combined_Text'].iloc[0])
print("Workout combined text sample:", workout_data['Combined_Text'].iloc[0])
# Separate features (X) and target (y)
X = train_data.drop('prognosis', axis=1)
y = train_data['prognosis']

print(f"\nFeature matrix shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"Feature sum statistics:")
print(f"Min features per sample: {X.sum(axis=1).min()}")
print(f"Max features per sample: {X.sum(axis=1).max()}")
print(f"Mean features per sample: {X.sum(axis=1).mean():.2f}")

# Encode the target labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print(f"\nLabel encoding info:")
print(f"Number of unique diseases: {len(label_encoder.classes_)}")
print(f"Encoded labels range: {y_encoded.min()} to {y_encoded.max()}")

# Split the dataset with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"\nTrain/Test split:")
print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")

# FIXED: Apply feature scaling and try different SVM parameters
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Try different SVM configurations
print("\nTrying different SVM configurations:")

# Configuration 1: Linear kernel
model_linear = SVC(kernel='linear', C=1.0, random_state=42)
model_linear.fit(X_train_scaled, y_train)
linear_accuracy = model_linear.score(X_test_scaled, y_test)
print(f"Linear SVM Accuracy: {linear_accuracy:.3f}")

# Configuration 2: RBF kernel with different C values
model_rbf = SVC(kernel='rbf', C=10.0, gamma='scale', random_state=42)
model_rbf.fit(X_train_scaled, y_train)
rbf_accuracy = model_rbf.score(X_test_scaled, y_test)
print(f"RBF SVM (C=10) Accuracy: {rbf_accuracy:.3f}")

# Configuration 3: Polynomial kernel
model_poly = SVC(kernel='poly', degree=3, C=1.0, random_state=42)
model_poly.fit(X_train_scaled, y_train)
poly_accuracy = model_poly.score(X_test_scaled, y_test)
print(f"Polynomial SVM Accuracy: {poly_accuracy:.3f}")

# Choose the best model
accuracies = {'linear': linear_accuracy, 'rbf': rbf_accuracy, 'poly': poly_accuracy}
best_model_name = max(accuracies, key=accuracies.get)
best_accuracy = accuracies[best_model_name]

if best_model_name == 'linear':
    best_model = model_linear
elif best_model_name == 'rbf':
    best_model = model_rbf
else:
    best_model = model_poly

print(f"\nBest model: {best_model_name.upper()} SVM with accuracy: {best_accuracy:.3f}")

# Test the model with a few predictions to verify it's working
print(f"\nTesting model predictions on first 5 test samples:")
test_predictions = best_model.predict(X_test_scaled[:5])
test_actual = y_test[:5]
for i in range(5):
    pred_disease = label_encoder.inverse_transform([test_predictions[i]])[0]
    actual_disease = label_encoder.inverse_transform([test_actual[i]])[0]
    print(f"Sample {i+1}: Predicted = {pred_disease}, Actual = {actual_disease}")

# TF-IDF Vectorization for all treatment texts
medications_tfidf = TfidfVectorizer(
    stop_words='english', 
    max_features=1000, 
    ngram_range=(1, 2),
    lowercase=True,
    min_df=1
)
# medications_tfidf_matrix = medications_tfidf.fit_transform(medications_data['Medication_Text'])
medications_tfidf_matrix = medications_tfidf.fit_transform(medications_data['Combined_Text'])

diets_tfidf = TfidfVectorizer(
    stop_words='english', 
    max_features=1000, 
    ngram_range=(1, 2),
    lowercase=True,
    min_df=1
)
# diets_tfidf_matrix = diets_tfidf.fit_transform(diets_data['Diet_Text'])
diets_tfidf_matrix = diets_tfidf.fit_transform(diets_data['Combined_Text'])
# workouts_tfidf_matrix = workouts_tfidf.fit_transform(workout_data['Combined_Text'])

workouts_tfidf = TfidfVectorizer(
    stop_words='english', 
    max_features=1000, 
    ngram_range=(1, 2),
    lowercase=True,
    min_df=1
)
# workouts_tfidf_matrix = workouts_tfidf.fit_transform(workout_data['Workout_Text'])
workouts_tfidf_matrix = workouts_tfidf.fit_transform(workout_data['Combined_Text'])

print("\nTF-IDF Matrices created:")
print(f"Medications TF-IDF shape: {medications_tfidf_matrix.shape}")
print(f"Diets TF-IDF shape: {diets_tfidf_matrix.shape}")
print(f"Workouts TF-IDF shape: {workouts_tfidf_matrix.shape}")

def get_tfidf_recommendations(query_text, tfidf_vectorizer, tfidf_matrix, data_df, 
                             original_column, top_n=5, min_similarity=0.01):
    """
    Pure TF-IDF + Cosine Similarity based recommendations
    """
    try:
        query_vector = tfidf_vectorizer.transform([query_text])
        similarity_scores = cosine_similarity(query_vector, tfidf_matrix).flatten()
        sorted_indices = np.argsort(similarity_scores)[::-1]
        
        recommendations = []
        recommendation_scores = []
        
        for idx in sorted_indices[:top_n*2]:
            if similarity_scores[idx] >= min_similarity:
                if original_column in ['Medication', 'Diet']:
                    original_data = data_df.iloc[idx][original_column]
                    parsed_items = parse_list_string(original_data)
                    
                    for item in parsed_items:
                        if item not in recommendations and item.strip():
                            recommendations.append(item)
                            recommendation_scores.append(similarity_scores[idx])
                            if len(recommendations) >= top_n:
                                break
                else:
                    item = data_df.iloc[idx][original_column]
                    if item not in recommendations and item.strip():
                        recommendations.append(item)
                        recommendation_scores.append(similarity_scores[idx])
            
            if len(recommendations) >= top_n:
                break
        
        if not recommendations:
            print(f"No matches above threshold {min_similarity}, returning top matches...")
            for idx in sorted_indices[:top_n]:
                if original_column in ['Medication', 'Diet']:
                    original_data = data_df.iloc[idx][original_column]
                    parsed_items = parse_list_string(original_data)
                    recommendations.extend(parsed_items[:top_n-len(recommendations)])
                else:
                    recommendations.append(data_df.iloc[idx][original_column])
                
                if len(recommendations) >= top_n:
                    break
        
        return recommendations[:top_n] if recommendations else ["No strong match found."]
            
    except Exception as e:
        print(f"Error in TF-IDF recommendation: {e}")
        return ["No strong match found."]

def recommend_treatment_content_based(symptoms_input, top_n=5, min_similarity=0.01):
    """
    FIXED: Pure TF-IDF + Cosine Similarity based treatment recommendation system
    """
    # Ensure symptoms_input is a numpy array
    if isinstance(symptoms_input, list):
        symptoms_input = np.array(symptoms_input).reshape(1, -1)
    elif isinstance(symptoms_input, pd.DataFrame):
        symptoms_input = symptoms_input.values
    
    # FIXED: Scale the input using the same scaler
    symptoms_input_scaled = scaler.transform(symptoms_input)
    
    # Predict disease using the best SVM model
    predicted_disease_encoded = best_model.predict(symptoms_input_scaled)
    predicted_disease = label_encoder.inverse_transform(predicted_disease_encoded)[0]
    
    print(f"\nPredicted Disease: {predicted_disease}")
    
    # Use the predicted disease name as the query for TF-IDF similarity
    disease_query = predicted_disease.lower().strip()
    
    # Alternative queries to try
    query_variants = [
        disease_query,
        disease_query.replace('_', ' '),
        disease_query.replace(' ', '_'),
        disease_query.replace('(', '').replace(')', ''),
        ' '.join(disease_query.split('_')),
        ' '.join(disease_query.split(' '))
    ]
    
    print(f"Query variants: {query_variants}")
    
    # Get TF-IDF based recommendations for each category
    best_medications = []
    best_diets = []
    best_workouts = []
    
    for query in query_variants:
        if query.strip():
            meds = get_tfidf_recommendations(
                query, medications_tfidf, medications_tfidf_matrix,
                medications_data, 'Medication', top_n, min_similarity
            )
            if meds != ["No strong match found."] and len(meds) > len(best_medications):
                best_medications = meds
            
            diets = get_tfidf_recommendations(
                query, diets_tfidf, diets_tfidf_matrix,
                diets_data, 'Diet', top_n, min_similarity
            )
            if diets != ["No strong match found."] and len(diets) > len(best_diets):
                best_diets = diets
            
            workouts = get_tfidf_recommendations(
                query, workouts_tfidf, workouts_tfidf_matrix,
                workout_data, 'workout', top_n, min_similarity
            )
            if workouts != ["No strong match found."] and len(workouts) > len(best_workouts):
                best_workouts = workouts
    
    final_medications = best_medications if best_medications else ["No strong match found."]
    final_diets = best_diets if best_diets else ["No strong match found."]
    final_workouts = best_workouts if best_workouts else ["No strong match found."]
    
    recommendations = {
        'Disease': predicted_disease,
        'Top Medications': final_medications,
        'Top Diets': final_diets,
        'Top Workouts': final_workouts
    }
    
    return recommendations

def create_symptoms_input(symptom_list):
    """Create symptoms input array from a list of symptom names"""
    symptoms = X.columns.tolist()
    symptoms_input = []
    
    symptom_list_lower = [s.strip().lower().replace(' ', '_') for s in symptom_list]
    
    matched_symptoms = []
    for symptom in symptoms:
        if symptom.lower() in symptom_list_lower:
            symptoms_input.append(1)
            matched_symptoms.append(symptom)
        else:
            symptoms_input.append(0)
    
    print(f"Matched symptoms: {matched_symptoms}")
    print(f"Total symptoms activated: {sum(symptoms_input)}")
    
    return symptoms_input

def get_symptoms_input():
    """FIXED: Better symptom input with fuzzy matching"""
    symptoms = X.columns.tolist()
    
    print("\nAvailable symptoms (first 20):")
    for i, symptom in enumerate(symptoms[:20]):
        print(f"{i+1}. {symptom}")
    print("... and more")
    
    print(f"\nTotal available symptoms: {len(symptoms)}")
    print("Please enter the symptoms as a comma-separated list.")
    print("Tip: Use underscores for multi-word symptoms (e.g., 'skin_rash', 'joint_pain')")
    
    input_symptoms = input("Enter symptoms: ").strip().lower()
    input_symptoms_list = [s.strip() for s in input_symptoms.split(',')]
    
    return create_symptoms_input(input_symptoms_list)

def test_tfidf_recommendations():
    """FIXED: Test TF-IDF recommendations with various symptom combinations"""
    test_cases = [
        ['itching', 'skin_rash', 'nodal_skin_eruptions'],
        ['cough', 'fever', 'breathlessness'],
        ['headache', 'nausea', 'vomiting'],
        ['joint_pain', 'fatigue', 'muscle_wasting'],
        ['chest_pain', 'shortness_of_breath', 'palpitations'],
        ['abdominal_pain', 'diarrhoea', 'nausea'],
        ['high_fever', 'chills', 'sweating']
    ]
    
    print("="*60)
    print("TESTING TF-IDF + COSINE SIMILARITY RECOMMENDATIONS")
    print("="*60)
    
    for i, symptoms in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {symptoms}")
        print("-" * 40)
        
        symptoms_input = create_symptoms_input(symptoms)
        
        if sum(symptoms_input) == 0:
            print("WARNING: No symptoms matched! Skipping this test case.")
            continue
            
        recommendations = recommend_treatment_content_based(symptoms_input)
        
        print(f"Disease: {recommendations['Disease']}")
        print(f"Top Medications: {recommendations['Top Medications']}")
        print(f"Top Diets: {recommendations['Top Diets']}")
        print(f"Top Workouts: {recommendations['Top Workouts']}")

if __name__ == "__main__":
    # Test the system first
    test_tfidf_recommendations()
    
    print("\n" + "="*60)
    print("INTERACTIVE MODE")
    print("="*60)
    
    # Interactive mode
    symptoms_input = get_symptoms_input()
    
    if sum(symptoms_input) == 0:
        print("ERROR: No symptoms were matched. Please check your input.")
    else:
        recommendations = recommend_treatment_content_based(symptoms_input)
        
        print("\nFinal Recommendations (TF-IDF + Cosine Similarity):")
        print(f"Disease: {recommendations['Disease']}")
        print(f"Top Medications: {recommendations['Top Medications']}")
        print(f"Top Diets: {recommendations['Top Diets']}")
        print(f"Top Workouts: {recommendations['Top Workouts']}")