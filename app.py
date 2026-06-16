from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from google import genai
import pandas as pd
from support import extract_band_ratios_from_csv, calculate_alpha_regions
from compare import compare_qeeg_states
from dotenv import load_dotenv
app = Flask(__name__)

load_dotenv()
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
api_key = os.getenv("GEMINI_API_KEY")

@app.route("/api/files/upload", methods=["POST"])
def upload():

    if "file" not in request.files:
        return {"error": "No file provided"}, 400

    file = request.files["file"]

    filename = secure_filename(file.filename)

    file.save(os.path.join(UPLOAD_FOLDER, filename))

    return {
        "message": "Uploaded successfully",
        "filename": filename
    }, 200

@app.route("/api/analysis/channel", methods=["GET"])
def channelAnalysis():
    filename = request.form["filename"]
    channel=request.form["channel"]
    official_channels = [
        'Fp1', 'Fp2', 'F3', 'F4',
        'F7', 'F8', 'T3', 'T4',
        'C3', 'C4', 'T5', 'T6',
        'P3', 'P4', 'O1', 'O2',
        'Fz', 'Cz', 'Pz'
    ]

    message = extract_band_ratios_from_csv(
        os.path.join(UPLOAD_FOLDER, filename),
        fs=500,
        channel_names=official_channels
    )
    channel_data = message[message["Channel"] == channel]
    return jsonify(
        message=channel_data.to_dict(orient="records")
    ), 200

@app.route("/api/analysis/channel/advice", methods=["GET"])
def channelAdvice():
    filename = request.form["filename"]
    channel=request.form["channel"]
    official_channels = [
        'Fp1', 'Fp2', 'F3', 'F4',
        'F7', 'F8', 'T3', 'T4',
        'C3', 'C4', 'T5', 'T6',
        'P3', 'P4', 'O1', 'O2',
        'Fz', 'Cz', 'Pz'
    ]

    message = extract_band_ratios_from_csv(
        os.path.join(UPLOAD_FOLDER, filename),
        fs=500,
        channel_names=official_channels
    )
    channel_data = message[message["Channel"] == channel]
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=f"You are going to get an analysis of qEEG metrics for the channel {channel}. Explain in a few sentences what course of action should be taken. Write just pure text and dont send any symbols {channel_data}"
    )
    return {
        "message": response.text
    }, 200

@app.route("/api/analysis/region", methods=["GET"])
def regionAnalysis():
    filename = request.form["filename"]
    official_channels = [
        'Fp1', 'Fp2', 'F3', 'F4',
        'F7', 'F8', 'T3', 'T4',
        'C3', 'C4', 'T5', 'T6',
        'P3', 'P4', 'O1', 'O2',
        'Fz', 'Cz', 'Pz'
    ]

    message = extract_band_ratios_from_csv(
        os.path.join(UPLOAD_FOLDER, filename),
        fs=500,
        channel_names=official_channels
    )

    message = message.set_index("Channel")

    regions = calculate_alpha_regions(message)

    return jsonify(
        regions=regions
    ), 200

@app.route("/api/analysis/region/advice", methods=["GET"])
def regionAdvice():
    filename = request.form["filename"]
    official_channels = [
        'Fp1', 'Fp2', 'F3', 'F4',
        'F7', 'F8', 'T3', 'T4',
        'C3', 'C4', 'T5', 'T6',
        'P3', 'P4', 'O1', 'O2',
        'Fz', 'Cz', 'Pz'
    ]

    message = extract_band_ratios_from_csv(
        os.path.join(UPLOAD_FOLDER, filename),
        fs=500,
        channel_names=official_channels
    )

    message = message.set_index("Channel")

    regions = calculate_alpha_regions(message)

    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=f"You are going to get an analysis of qEEG metrics. Explain in a few sentences what course of action should be taken. Write just pure text and dont send any symbols {regions}"
    )

    return {
        "message": response.text
    }, 200

@app.route("/api/analysis/band_ratio", methods=["GET"])
def bandRatioAnalysis():
    filename = request.form["filename"]
    official_channels = [
        'Fp1', 'Fp2', 'F3', 'F4',
        'F7', 'F8', 'T3', 'T4',
        'C3', 'C4', 'T5', 'T6',
        'P3', 'P4', 'O1', 'O2',
        'Fz', 'Cz', 'Pz'
    ]

    message = extract_band_ratios_from_csv(
        os.path.join(UPLOAD_FOLDER, filename),
        fs=500,
        channel_names=official_channels
    )

    return jsonify(
        message=message.to_dict(orient="records")
    ), 200

@app.route("/api/analysis/band_ratio/advice", methods=["GET"])
def bandRatioAdvice():
    filename = request.form["filename"]
    official_channels = [
        'Fp1', 'Fp2', 'F3', 'F4',
        'F7', 'F8', 'T3', 'T4',
        'C3', 'C4', 'T5', 'T6',
        'P3', 'P4', 'O1', 'O2',
        'Fz', 'Cz', 'Pz'
    ]

    message = extract_band_ratios_from_csv(
        os.path.join(UPLOAD_FOLDER, filename),
        fs=500,
        channel_names=official_channels
    )
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=f"You are going to get an analysis of qEEG metrics. Explain in a few sentences what course of action should be taken. Write just pure text and dont send any symbols {message}"
    )

    return {
        "message": response.text
    }, 200

@app.route("/api/analysis/comparison", methods=["GET"])
def comparisonAnalysis():
    filename1 = request.form["filename1"]
    filename2 = request.form["filename2"]
    df1=pd.read_csv(os.path.join(UPLOAD_FOLDER, filename1))
    df2=pd.read_csv(os.path.join(UPLOAD_FOLDER, filename2))
    result=compare_qeeg_states(df1,df2)
    return (result), 200

@app.route("/api/analysis/comparison/advice", methods=["GET"])
def comparisonAdvice():
    filename1 = request.form["filename1"]
    filename2 = request.form["filename2"]
    df1 = pd.read_csv(os.path.join(UPLOAD_FOLDER, filename1))
    df2 = pd.read_csv(os.path.join(UPLOAD_FOLDER, filename2))
    result = compare_qeeg_states(df1, df2)["ratios"]
    client = genai.Client(api_key=api_key)

    response = client.models.generate_content(
        model="gemini-3.1-flash-lite",
        contents=f"You are going to get an analysis of qEEG metrics. Explain in a few sentences what course of action should be taken. Write just pure text and dont send any symbols {result}"
    )

    return {
        "message": response.text
    }, 200



if __name__ == "__main__":
    app.run(debug=True)