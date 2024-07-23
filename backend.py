from flask import Flask, request, jsonify
import pandas as pd
import requests


app = Flask(__name__)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"})

    if file and file.filename.endswith('.csv'):
        df = pd.read_csv(file)
        # Preprocess the data as needed
        preprocessed_data = preprocess_data(df)
        # Convert to JSON for API submission
        response = send_data_to_api(preprocessed_data)
        return jsonify(response)

    return jsonify({"error": "Invalid file type"})

def preprocess_data(df):
    # checking preprocessing steps
    df['TV'] = df['TV'].fillna(0)
    df['Radio'] = df['Radio'].fillna(0)
    df['Newspaper'] = df['Newspaper'].fillna(0)
    df['Sales'] = df['Sales'].fillna(0)
    return df

def send_data_to_api(df):
    api_url = 'https://api.apistudio.app/postapi/create/si_01_advertisement'
    headers = {'Content-Type': 'application/json'}
    responses = []
    for index, row in df.iterrows():
        json_data = {
            "data": {
                "tv": str(row['TV']),
                "radio": str(row['Radio']),
                "newspaper": str(row['Newspaper']),
                "sales": str(row['Sales'])
            }
        }
        response = requests.post(api_url, json=json_data, headers=headers)
        responses.append(response.json())
    return 





if __name__ == '__main__':
    app.run(debug=True)



    
    
    
        