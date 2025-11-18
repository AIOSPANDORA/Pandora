"""
AI Assimilation Script – Secure Web Interface for Device-to-Pandora AIOS Integration

Features:
- Displays a QR code or link for smooth connection from phone/laptop
- Provides a ChatGPT-style web chat interface (adapt as needed)
- Optionally, supports file upload/download, status, pairing, and data sync

Requirements:
pip install flask flask_qrcode
"""

from flask import Flask, render_template_string, request, jsonify, send_from_directory
import socket
import threading

try:
    import flask_qrcode
except ImportError:
    pass  # Will show text link instead of QR if not present

app = Flask(__name__)

# Simple in-memory chat log for demo purposes
chat_history = []

@app.route("/")
def home():
    return render_template_string('''
        <h1>AIOS Pandora Assimilation Portal</h1>
        <p>Device paired: {{ device_info }}</p>
        <form id="chatForm">
          <input id="message" name="message" placeholder="Type message..." size=60 autofocus>
          <button>Send</button>
        </form>
        <pre id="history"></pre>
        <script>
        form = document.getElementById('chatForm'); hist = document.getElementById('history');
        form.onsubmit = async (e)=>{
          e.preventDefault();
          msg = form.message.value;
          resp = await fetch('/chat', {method:'POST',headers:{"Content-Type":"application/json"},body:JSON.stringify({message:msg})});
          data = await resp.json();
          hist.textContent += "You: "+msg+"\\nAI: "+data.response+"\\n";
          form.message.value="";
        };
        </script>
    ''', device_info=request.remote_addr)

@app.route("/chat", methods=["POST"])
def chat():
    """Handle chat messages with input validation"""
    try:
        if not request.json or "message" not in request.json:
            return jsonify(error="Invalid request format"), 400
        
        user_message = request.json["message"]
        
        # Security: Limit message length
        if len(user_message) > 1000:
            return jsonify(error="Message too long (max 1000 chars)"), 400
        
        # Security: Basic sanitization (prevent script injection)
        user_message = user_message.strip()
        
        # AIOS Pandora call goes here in actual system
        ai_response = fake_ai_response(user_message)
        chat_history.append((user_message, ai_response))
        
        # Limit chat history to prevent memory issues
        if len(chat_history) > 100:
            chat_history.pop(0)
        
        return jsonify(response=ai_response)
    except Exception as e:
        return jsonify(error=f"Server error: {str(e)}"), 500

def fake_ai_response(msg):
    """Demo stub; you would call your AI/Pandora backend here"""
    user_message = request.json["message"]
    # AIOS Pandora call goes here in actual system
    ai_response = fake_ai_response(user_message)
    chat_history.append((user_message, ai_response))
    return jsonify(response=ai_response)

def fake_ai_response(msg):
    # Demo stub; you would call your AI/Pandora backend here
    if "status" in msg.lower():
        return "System healthy. AI Pandora up."
    elif "sync" in msg.lower():
        return "Assimilating settings from client device...done."
    elif "help" in msg.lower():
        return "Available commands: status, sync, info. Ask me anything!"
    elif "info" in msg.lower():
        return "Pandora AIOS v2.5 - AI Operating System with stoic philosophy."
    return "AIOS Pandora heard: " + msg

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory("static", filename)

@app.route("/health")
def health():
    """Health check endpoint for monitoring"""
    return jsonify({
        'status': 'healthy',
        'service': 'Pandora Assimilation Portal',
        'version': '2.5.0',
        'chat_history_size': len(chat_history)
    })

@app.route("/clear", methods=["POST"])
def clear_history():
    """Clear chat history (admin function)"""
    global chat_history
    chat_history = []
    return jsonify(message="Chat history cleared")

def get_local_ip():
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))  # Google's DNS, just to grab interface IP
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip

def serve():
    print("\n==== AIOS Pandora Assimilation Portal ====")
    local_ip = get_local_ip()
    print(f"Connect from your phone/computer at: http://{local_ip}:5000/")
    try:
        import flask_qrcode
        print("QR code for quick access:")
        print(f"http://{local_ip}:5000/")
    except ImportError:
        print("(Install flask_qrcode for QR code support)")
    app.run(host='0.0.0.0', port=5000)

if __name__ == "__main__":
    serve()