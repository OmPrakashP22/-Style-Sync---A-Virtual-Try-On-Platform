import os
import base64
from flask import Flask, request, render_template
from inference import TryOnPipeline

app = Flask(__name__)
base_dir = os.path.dirname(os.path.abspath(__file__))
pipeline = TryOnPipeline(base_dir)

@app.route('/')
def home():
    return render_template("index.html")

@app.route("/preds", methods=['POST'])
def submit():
    cloth = request.files['cloth']
    model = request.files['model']
    background_prompt = request.form.get('background_prompt', '').strip()
    # Save uploads temporarily
    uploads_dir = os.path.join(base_dir, "temp_uploads")
    os.makedirs(uploads_dir, exist_ok=True)
    
    cloth_path = os.path.join(uploads_dir, "uploaded_cloth.jpg")
    model_path = os.path.join(uploads_dir, "uploaded_model.jpg")
    
    cloth.save(cloth_path)
    model.save(model_path)

    # Customization parameters
    custom_kwargs = {}
    if request.form.get('color_transfer') == 'true':
        custom_kwargs['color_transfer'] = True
        custom_kwargs['transfer_mode'] = request.form.get('transfer_mode', 'color')
        custom_kwargs['target_color'] = request.form.get('target_color', '255,255,255')
        custom_kwargs['texture_blend'] = float(request.form.get('texture_blend', '0.6'))
        
        # Handle texture file if uploaded
        if 'texture_file' in request.files and request.files['texture_file'].filename:
            texture_file = request.files['texture_file']
            texture_path = os.path.join(uploads_dir, "texture.jpg")
            texture_file.save(texture_path)
            custom_kwargs['texture_path'] = texture_path

    print("Running Inference Pipeline locally...")
    try:
        # Run pipeline — returns (result_path, critique_text)
        result_path, critique = pipeline.run(
            model_path, cloth_path,
            background_prompt=background_prompt if background_prompt else None,
            **custom_kwargs
        )
        
        # Encode result image to base64 for display
        with open(result_path, "rb") as f:
            encoded_data = base64.b64encode(f.read()).decode()
            
        return render_template('index.html', op=encoded_data, critique=critique,
                               bg_prompt=background_prompt)
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True, port=5000)
