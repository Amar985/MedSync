from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Load the model
filename = "savmodel.sav"
plot_model_predictions = pickle.load(open(filename, 'rb'))

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Assuming you have a CSV file  containing your data
        full_df = pd.read_csv('data/consumo_material_clean.xlsx')
        product = int(request.form['product'])
        hospital = request.form['hospital']
        plot_train = False  # Assuming you always want plot_train to be False
        plot_model_predictions(full_df, product, hospital, plot_train)
        return render_template('result.html')
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
