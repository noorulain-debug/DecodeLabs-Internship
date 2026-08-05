import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 1. BUILD THE JOB ROLE DATASET
job_roles = {
    "Data Scientist": "Python SQL Machine Learning Statistics Data Analysis Pandas NumPy Data Visualization",
    "DevOps Engineer": "AWS Docker Kubernetes CI CD Automation Linux Cloud Computing Networking",
    "Backend Developer": "Java Python SQL REST APIs Node.js Databases Software Architecture Git",
    "Frontend Developer": "JavaScript React CSS HTML UI Design Responsive Design Web Development Git",
    "AI/ML Engineer": "Python Machine Learning Deep Learning TensorFlow PyTorch Neural Networks Data Science",
    "Cloud Architect": "AWS Azure Cloud Computing Networking Security Automation Infrastructure DevOps",
    "Cybersecurity Analyst": "Networking Security Linux Encryption Risk Assessment Penetration Testing Automation",
    "Full-Stack Developer": "JavaScript React Node.js SQL REST APIs Git Web Development Databases"
}

df = pd.DataFrame(list(job_roles.items()), columns=["role", "skills_text"])

print("Job Role Dataset:")
print(df)


# 2. THE RECOMMENDATION ENGINE (fully self-contained function)
def recommend_careers(user_skills, job_df, top_n=3):
    """Takes a list of user skills, returns the top_n matching job roles."""
    user_text = " ".join(user_skills)

    # VOCABULARY-MISMATCH CHECK: build the known vocabulary from the JOB
    # ROLES ONLY (never the user's own input) — otherwise the user's words
    # would trivially "recognize" themselves, making the check meaningless.
    vocab_reference = TfidfVectorizer()
    vocab_reference.fit(job_df["skills_text"].tolist())
    known_vocabulary = set(vocab_reference.get_feature_names_out())

    user_words = set(user_text.lower().split())
    recognized_words = user_words & known_vocabulary
    unrecognized_words = user_words - known_vocabulary

    # Combine job roles + user into one shared vocabulary space for scoring
    all_texts = job_df["skills_text"].tolist() + [user_text]

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(all_texts)

    job_vectors = tfidf_matrix[:-1]
    user_vector = tfidf_matrix[-1]

    similarity_scores = cosine_similarity(user_vector, job_vectors)

    result_df = job_df.copy()
    result_df["similarity_score"] = similarity_scores[0]
    ranked = result_df.sort_values(by="similarity_score", ascending=False)

    top_matches = ranked.head(top_n)[["role", "similarity_score"]]

    # Build a warning if recognition was poor (a lightweight Cold Start check)
    warning = None
    if len(recognized_words) == 0:
        warning = (
            "None of your input matched any known skill in our dataset. "
            "These results are not meaningful \u2014 try entering specific skills "
            "(e.g. Python, React, AWS) rather than job titles or unrelated terms."
        )
    elif len(unrecognized_words) > 0:
        warning = (
            f"Note: {', '.join(sorted(unrecognized_words))} "
            f"{'was' if len(unrecognized_words) == 1 else 'were'} not recognized "
            "as a known skill and did not contribute to your matches."
        )

    return top_matches, warning


# 3. INTERACTIVE INPUT LOOP
print("\n=== Tech Stack Recommender ===")
print("Enter at least 3 skills or interests, separated by commas.")
print("Example: Python, Cloud Computing, Automation\n")

raw_input_text = input("Your skills: ")
user_skills = [skill.strip() for skill in raw_input_text.split(",")]

if len(user_skills) < 3:
    print("\nPlease enter at least 3 skills for accurate matching.")
else:
    results, warning = recommend_careers(user_skills, df)
    print(f"\nBased on your skills ({', '.join(user_skills)}), here are your top career matches:\n")
    print(results.to_string(index=False))
    if warning:
        print(f"\n⚠ {warning}")


# 4. STRESS-TEST WITH A DIFFERENT SKILL SET (hardcoded, for comparison)
test_skills = ["JavaScript", "React", "CSS"]
test_results, test_warning = recommend_careers(test_skills, df)
print(f"\n--- Test run with {test_skills} ---")
print(test_results.to_string(index=False))
if test_warning:
    print(f"\n⚠ {test_warning}")