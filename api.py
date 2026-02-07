from flask import Flask, request, jsonify
from flask_cors import CORS
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer, util
import numpy as np
import os
import mysql.connector 
import json
from datetime import datetime

# Initialize the Web App
app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------
# 1. DATABASE SETUP (Auto-Create & Robust Connect)
# ---------------------------------------------------------
DB_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": ""  # Default XAMPP password is empty
}
DB_NAME = "protected_db"

def create_database_if_not_exists():
    """Connects to MySQL and creates the DB if missing."""
    try:
        # Connect to MySQL Server (No DB selected)
        conn = mysql.connector.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {DB_NAME}")
        conn.close()
        print(f" ✔ Database '{DB_NAME}' checked/created.")
    except Exception as e:
        print(f" ⚠️  MySQL Connection Error: {e}")
        print("    (Make sure XAMPP/WAMP MySQL is running!)")

def get_db_connection():
    """Connects to the specific database"""
    config = DB_CONFIG.copy()
    config["database"] = DB_NAME
    return mysql.connector.connect(**config)

def init_db():
    """Creates the table in MySQL if it doesn't exist."""
    create_database_if_not_exists()
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS assessments (
                id INT AUTO_INCREMENT PRIMARY KEY,
                timestamp VARCHAR(255),
                risk_level VARCHAR(50),
                main_category VARCHAR(255),
                confidence VARCHAR(50),
                full_report TEXT
            )
        ''')
        conn.commit()
        cursor.close()
        conn.close()
        print(" ✔ Table 'assessments' ready.")
    except Exception as e:
        print(f" ❌ Table Init Error: {e}")

# Initialize on startup
init_db()

# ---------------------------------------------------------
# 2. LOAD AI MODELS
# ---------------------------------------------------------
print("\n➡ INITIALIZING PROTECTED AI SYSTEM...")

# A. Load BERTopic
print("   [1/3] Loading BERTopic Model...")
try:
    topic_model = BERTopic.load("vawc_bertopic_model")
    print("   ✔ BERTopic Loaded Successfully!")
except Exception as e:
    print(f"   ❌ Error loading BERTopic: {e}")
    topic_model = None

# B. Load Semantic Engine
print("   [2/3] Loading Semantic Engine...")
try:
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    print("   ✔ Semantic Engine Loaded Successfully!")
except Exception as e:
    print(f"   ❌ Error loading Semantic Engine: {e}")
    embedding_model = None

# ---------------------------------------------------------
# 3. CONFIGURATION
# ---------------------------------------------------------
KEYWORD_TO_CATEGORY = {
    "hit": "Physical Abuse", "slap": "Physical Abuse", "punch": "Physical Abuse", 
    "kick": "Physical Abuse", "weapon": "Physical Abuse", "kill": "Physical Abuse",
    "threat": "Control & Manipulation"
}

TOPIC_MAPPING = {
    2: "Control & Manipulation", 3: "Control & Manipulation", 4: "Neglect & Emotional Withdrawal",
    6: "Control & Manipulation", 8: "Verbal & Emotional Abuse", 1: "Neglect & Emotional Withdrawal",
    5: "Verbal & Emotional Abuse"
}

ANCHOR_SENTENCES = {
    "Control & Manipulation": ["He controls who I see.", "He demands passwords.", "He tracks my location."],
    "Verbal & Emotional Abuse": ["He calls me names.", "He yells and screams.", "He blames me for everything."],
    "Physical Abuse": ["He hurts me physically.", "He pushes and shoves.", "He throws things."],
    "Neglect & Emotional Withdrawal": ["He ignores me for days.", "He isolates me from family."]
}

if embedding_model:
    ANCHOR_EMBEDDINGS = {}
    for cat, sentences in ANCHOR_SENTENCES.items():
        ANCHOR_EMBEDDINGS[cat] = embedding_model.encode(sentences)

CATEGORIES_ADVICE = {
    "Control & Manipulation": "Document incidents. Do not share your location if unsafe.",
    "Verbal & Emotional Abuse": "Prioritize your mental health. Do not engage in escalating arguments.",
    "Neglect & Emotional Withdrawal": "Seek counseling or support from trusted friends.",
    "Physical Abuse": "Go to a safe place immediately. Call 911.",
    "Healthy/Low Risk": "Maintain open communication.",
    "Neutral / Unclassified": "No specific pattern detected."
}

# ---------------------------------------------------------
# 4. DETECTION ENGINE (FIXED)
# ---------------------------------------------------------
def hybrid_detect(text):
    if not text or len(text.strip()) < 5:
        return "Healthy/Low Risk", "Insufficient Data", 0.0

    text_lower = text.lower()

    # Layer 1: Keywords
    for key, val in KEYWORD_TO_CATEGORY.items():
        if key in text_lower:
            return val, "Manual Keyword Match", 1.0

    # Layer 2: BERTopic
    if topic_model:
        topics, probs = topic_model.transform([text])
        if topics[0] in TOPIC_MAPPING:
            # --- FIX: Handle different probability shapes ---
            if probs is not None:
                try:
                    # If probs[0] is iterable (list/array), take max
                    conf = float(max(probs[0]))
                except TypeError:
                    # If probs[0] is a single float, use it directly
                    conf = float(probs[0])
            else:
                conf = 0.5
            
            return TOPIC_MAPPING[topics[0]], f"AI Cluster {topics[0]}", conf

    # Layer 3: Semantic
    if embedding_model:
        user_embedding = embedding_model.encode(text)
        best_cat, high_score = "Neutral / Unclassified", 0.0
        for cat, anchors in ANCHOR_EMBEDDINGS.items():
            score = float(util.cos_sim(user_embedding, anchors).max())
            if score > high_score: high_score, best_cat = score, cat
        if high_score > 0.35: return best_cat, "AI Semantic Similarity", high_score

    return "Healthy/Low Risk", "System Default", 0.0

# ---------------------------------------------------------
# 5. API ROUTES
# ---------------------------------------------------------

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.json
        grouped_answers = data.get('grouped_answers', {})
        
        breakdown = {}
        all_sentences = []
        total_risk_score = 0

        # Analyze Sections
        for section_name, items in grouped_answers.items():
            if not items: continue
            
            section_text = " ".join([item['text'] for item in items])
            section_score = sum(item['weight'] for item in items)
            all_sentences.append(section_text)
            
            cat, method, conf = hybrid_detect(section_text)
            
            # Risk Logic
            if cat == "Physical Abuse": risk, color = "Severe", "red"
            elif section_score >= 6: risk, color = "High", "orange"
            elif section_score >= 2: risk, color = "Moderate", "yellow"
            else: risk, color = "Low", "green"

            breakdown[section_name] = {
                "category": cat, "risk": risk, "color": color, 
                "confidence": f"{round(conf * 100, 1)}%"
            }
            total_risk_score += section_score

        # General Analysis
        full_text = " ".join(all_sentences)
        gen_cat, gen_method, gen_conf = hybrid_detect(full_text)
        
        if gen_cat == "Physical Abuse": gen_risk, gen_color = "Severe", "red"
        elif total_risk_score >= 10: gen_risk, gen_color = "High", "orange"
        elif total_risk_score >= 3: gen_risk, gen_color = "Moderate", "yellow"
        else: gen_risk, gen_color = "Low", "green"

        response_data = {
            "general": {
                "category": gen_cat, "risk": gen_risk, "color": gen_color,
                "advice": CATEGORIES_ADVICE.get(gen_cat, "No advice available."),
                "method": gen_method, "confidence": f"{round(gen_conf * 100, 1)}"
            },
            "breakdown": breakdown
        }

        # --- SAVE TO MYSQL ---
        try:
            conn = get_db_connection()
            cursor = conn.cursor()
            sql = "INSERT INTO assessments (timestamp, risk_level, main_category, confidence, full_report) VALUES (%s, %s, %s, %s, %s)"
            val = (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), gen_risk, gen_cat, f"{round(gen_conf * 100, 1)}%", json.dumps(response_data))
            cursor.execute(sql, val)
            conn.commit()
            cursor.close()
            conn.close()
            print(f" ✔ Saved result to MySQL: {gen_risk}")
        except Exception as e:
            print(f" ❌ DB Error (Data not saved): {e}")

        return jsonify({"status": "success", "result": response_data})

    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/admin/data', methods=['GET'])
def get_admin_data():
    try:
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM assessments ORDER BY id DESC")
        results = cursor.fetchall()
        for row in results:
            if isinstance(row['full_report'], str):
                row['full_report'] = json.loads(row['full_report'])
        cursor.close()
        conn.close()
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    print("\n🔹 ProtectEd Backend Running on http://127.0.0.1:5000")
    app.run(debug=True, use_reloader=False, port=5000)