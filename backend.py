from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from prediction import predict_eeg
import os
import psycopg2

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def get_db():
    return psycopg2.connect(
        dbname="eeg_db",
        user="eeg_user",
        password="NewStrongPassword123",
        host="localhost"
    )


@app.route("/")
def home():
    return render_template("main.html")


@app.route("/result.html")
def result_page():
    return render_template("result.html")


@app.route("/predict", methods=["POST"])
def predict():

    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    # Run ML prediction
    result = predict_eeg(filepath)

    prediction_result = result["prediction"]
    confidence = result["confidence"]
    print("you have reached backend") 
    # Save to PostgreSQL
    conn = get_db()
    cur = conn.cursor()

    print("DB connected successfully")

    cur.execute("""
         INSERT INTO eeg_prediction
            (
                patient_name,
                patient_number,
                prediction_result,
                confidence
            )
            VALUES (%s, %s, %s, %s)
        """, (
            request.form.get("name"),
            request.form.get("phone"),
            prediction_result,
            confidence
        ))

    conn.commit()

    cur.close()
    conn.close()

    
    return jsonify({
        "prediction": prediction_result,
        "confidence": confidence
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)