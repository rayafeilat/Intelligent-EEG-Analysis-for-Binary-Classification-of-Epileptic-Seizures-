def predict_eeg(filepath):

    data = np.load(filepath, allow_pickle=True)

    # safe extraction
    if isinstance(data, np.lib.npyio.NpzFile) and "X" in data:
        X = data["X"]
    else:
        X = data

    X = np.array(X)

    preds = model.predict(X)

    results = []

    for i, p in enumerate(preds):

        probability = float(p[0])

        label = "Seizure" if probability > 0.5 else "Normal"

        results.append({
            "window": i,
            "probability": probability,
            "prediction": label
        })

    # Average probability across all windows
    avg_prob = np.mean([r["probability"] for r in results])

    final_prediction = (
        "Seizure"
        if avg_prob > 0.5
        else "Normal"
    )

    confidence = (
        avg_prob * 100
        if avg_prob > 0.5
        else (1 - avg_prob) * 100
    )

    return {
        "prediction": final_prediction,
        "confidence": round(confidence, 2),
        "windows": results,
        "summary": "Prediction completed"
    }