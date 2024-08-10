from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["GET", "POST"])
def predictor():

    if request.method == "POST":

        model_selection = request.form.get("model")
        if model_selection == "SVM":
            model = joblib.load("./models/svm_model.joblib")
        elif model_selection == "Decision Tree":
            model = joblib.load("./models/decision_tree_model.joblib")
        elif model_selection == "Random Forest":
            model = joblib.load("./models/random_forest_model.joblib")

        radius = request.form.get("mean_radius")
        texture = request.form.get("mean_texture")
        smoothness = request.form.get("mean_smoothness")

        data = np.array([[radius, texture, smoothness]])

        preprocessor = joblib.load("./models/preprocessor.joblib")
        transformed_data = preprocessor.predict(data)

        prediction = model.predict(transformed_data)

        return render_template("result.html", data=prediction)

    return render_template("predictor.html")


if __name__ == "__main__":
    app.run(debug = True)