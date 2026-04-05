from flask import Flask, render_template, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

app = Flask(__name__)

# QUESTIONS (with variations)
questions = [
    "documents required for ug admission",
    "admission documents",
    "what documents needed for admission",
    "how to apply for scholarships",
    "scholarship process",
    "where to check results",
    "exam results",
    "how to apply for internships",
    "internship apply",
    "where to download certificates",
    "digilocker documents"
]

# ANSWERS
answers = [
    "You need SSC, Intermediate marks memo, TC, Study Certificate, Aadhar, photos, caste and income certificate.",

    "You need SSC, Intermediate marks memo, TC, Study Certificate, Aadhar, photos, caste and income certificate.",

    "You need SSC, Intermediate marks memo, TC, Study Certificate, Aadhar, photos, caste and income certificate.",

    "Apply through <a href='https://scholarships.gov.in' target='_blank'>National Scholarship Portal</a>.",

    "Apply through <a href='https://scholarships.gov.in' target='_blank'>National Scholarship Portal</a>.",

    "Check results on <a href='https://www.braou.ac.in' target='_blank'>University Website</a>.",

    "Check results on <a href='https://www.braou.ac.in' target='_blank'>University Website</a>.",

    "Apply through <a href='https://internshala.com' target='_blank'>Internshala</a> or <a href='https://linkedin.com' target='_blank'>LinkedIn</a>.",

    "Apply through <a href='https://internshala.com' target='_blank'>Internshala</a> or <a href='https://linkedin.com' target='_blank'>LinkedIn</a>.",

    "Download from <a href='https://digilocker.gov.in' target='_blank'>DigiLocker</a>.",

    "Download from <a href='https://digilocker.gov.in' target='_blank'>DigiLocker</a>."
]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(questions)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get", methods=["POST"])
def chatbot_response():
    user_input = request.json["message"].lower()

    # Greeting handling
    if any(word in user_input for word in ["hi", "hello", "hey"]):
        return jsonify({"reply": "Hello 👋 How can I help you?"})

    if "thank" in user_input:
        return jsonify({"reply": "You're welcome 😊"})

    user_vec = vectorizer.transform([user_input])
    similarity = cosine_similarity(user_vec, X)

    max_score = np.max(similarity)
    index = np.argmax(similarity)

    if max_score < 0.3:
        return jsonify({
            "reply": "Sorry, I didn’t understand. Please try another question."
        })

    return jsonify({"reply": answers[index]})

if __name__ == "__main__":
    app.run(debug=True)
