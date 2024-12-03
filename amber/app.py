from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from twilio.rest import Client

app = Flask(__name__)
CORS(app)

# Twilio configuration (replace with your credentials)
TWILIO_ACCOUNT_SID = "your_account_sid"
TWILIO_AUTH_TOKEN = "your_auth_token"
TWILIO_PHONE_NUMBER = "your_twilio_phone_number"
AUTHORITY_PHONE_NUMBER = "recipient_phone_number"

client = Client(TWILIO_ACCOUNT_SID, TWILIO_AUTH_TOKEN)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/send_alert', methods=['POST'])
def send_alert():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400

        camera_name = data.get('cameraName', 'Unknown Camera')
        people_count = data.get('peopleCount', 0)
        crowd_density = data.get('status', 'unknown')

        message_content = (
            f"Amber Alert from {camera_name}: High crowd density detected! "
            f"Current density: {crowd_density} with {people_count} people."
        )

        message = client.messages.create(
            body=message_content,
            from_=TWILIO_PHONE_NUMBER,
            to=AUTHORITY_PHONE_NUMBER
        )

        return jsonify({'message': 'Alert sent successfully', 'sid': message.sid}), 200

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
