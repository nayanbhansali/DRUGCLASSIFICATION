from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open("ML2.pkl", "rb"))

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    prob_df = None
    if request.method == 'POST':
        age = int(request.form['age'])
        cholesterol = request.form['cholesterol']
        bp = request.form['bp']
        na_to_k = float(request.form['na_to_k'])
        sex = request.form['sex']

        bin_age = [0, 19, 29, 39, 49, 59, 69, 80]
        category_age = ['<20s', '20s', '30s', '40s', '50s', '60s', '>60s']
        age_binned = pd.cut([age], bins=bin_age, labels=category_age)[0]

        bin_na_to_k = [0, 9, 19, 29, 50]
        category_na_to_k = ['<10', '10-20', '20-30', '>30']
        na_to_k_binned = pd.cut([na_to_k], bins=bin_na_to_k, labels=category_na_to_k)[0]

        data = {
            'Sex': [sex],
            'BP': [bp],
            'Cholesterol': [cholesterol],
            'Age_binned': [age_binned],
            'Na_to_K_binned': [na_to_k_binned]
        }
        df = pd.DataFrame(data)
        df_encoded = pd.get_dummies(df, columns=['Sex', 'BP', 'Cholesterol', 'Age_binned', 'Na_to_K_binned'])
        
        desired_columns_order = [
            'Sex_F', 'Sex_M',
            'BP_HIGH', 'BP_LOW', 'BP_NORMAL',
            'Cholesterol_HIGH', 'Cholesterol_NORMAL',
            'Age_binned_<20s', 'Age_binned_20s', 'Age_binned_30s', 'Age_binned_40s', 'Age_binned_50s', 'Age_binned_60s', 'Age_binned_>60s',
            'Na_to_K_binned_<10', 'Na_to_K_binned_10-20', 'Na_to_K_binned_20-30', 'Na_to_K_binned_>30'
        ]
        df_encoded = df_encoded.reindex(columns=desired_columns_order)
        df_encoded.fillna(0, inplace=True)
        df_encoded_int = df_encoded.astype(int)

        # Predict
        prediction = model.predict(df_encoded)[0]
        prob = model.predict_proba(df_encoded)
        class_labels = model.classes_
        prob_df = pd.DataFrame(prob, columns=class_labels)

    return render_template('index.html', prediction=prediction, prob_df=prob_df)

if __name__ == '__main__':
    if __name__ == "__main__":
        app.run(host='0.0.0.0', port=8080, debug=True)

