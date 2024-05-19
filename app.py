from flask import Flask, request, jsonify, render_template, session
from openai import OpenAI
from utils import play_audio
from dotenv import load_dotenv
import os
import uuid

app = Flask(__name__, static_url_path="/static")
app.secret_key = os.urandom(24)  # Secret key for session management
load_dotenv()

# Get OPENAI_API_KEY from .env file
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload_audio", methods=["POST"])
def upload_audio_route():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Generate a unique filename for the uploaded file
    audio_filename = f"{uuid.uuid4()}.wav"
    audio_path = os.path.join("static", audio_filename)
    file.save(audio_path)

    return (
        jsonify(
            {"message": "File uploaded successfully", "audio_filename": audio_filename}
        ),
        200,
    )

@app.route("/process_audio", methods=["POST"])
def process_audio_route():
    data = request.get_json()
    audio_filename = data.get("audio_filename")
    audio_path = os.path.join("static", audio_filename)

    # Transcribe audio
    with open(audio_path, "rb") as audio_file:
        transcription = client.audio.transcriptions.create(
            model="whisper-1", file=audio_file
        )
    transcription_text = transcription.text

    # Initialize the conversation history if it doesn't exist
    if "conversation_history" not in session:
        session["conversation_history"] = [
            {
                "role": "system",
                "content": "You are Paul, a personal assistant. Please answer in short sentences. Introduce yourself to the user the first time.",
            }
        ]

    # Append the user's message to the conversation history
    session["conversation_history"].append(
        {"role": "user", "content": transcription_text}
    )

    # Get response from GPT model
    response = client.chat.completions.create(
        model="gpt-4o", messages=session["conversation_history"]
    )
    response_text = response.choices[0].message.content

    # Append the assistant's response to the conversation history
    session["conversation_history"].append(
        {"role": "assistant", "content": response_text}
    )

    # Generate speech from response
    speech_filename = f"{uuid.uuid4()}.mp3"
    speech_path = os.path.join("static", speech_filename)

    speech_response = client.audio.speech.create(
        model="tts-1", voice="alloy", input=response_text
    )
    speech_response.stream_to_file(speech_path)

    # Exclude the initial system message from the conversation history sent to the frontend
    filtered_history = [
        message for message in session["conversation_history"] if message["role"] != "system"
    ]

    return (
        jsonify(
            {
                "conversation_history": filtered_history,
                "audio_url": f"/static/{speech_filename}",
            }
        ),
        200,
    )

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
