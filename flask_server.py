from flask import Flask 
from flask import request, jsonify
from PIL import Image
from image_text_replacer import getTextBlocks
import base64 
import io 

app = Flask(__name__)


@app.route('/', methods=['POST'])
def main(): 
    try: 
        data = request.get_json()
        if 'image' not in data: 
            return jsonify({"error", 'No image provided'}), 400 
    
        # Decode base64 image
        image_data = base64.b64decode(data['image'])
        image = Image.open(io.BytesIO(image_data))
        
        # Perform some processing (example: just show the image size)
        print(f"Image received: {image.size}")
        image.show() 

        # Return JSON response
        response_data = {"textBlocks": getTextBlocks(image)}
        print("RESPONSE:\n", response_data)

        return jsonify(response_data)

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=7333)

