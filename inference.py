import os
import shutil
from PIL import Image
import subprocess
from remove_bg import remove_background_from_image
from cloth_mask import generate_mask

def resize_img(im, shape=(768, 1024)):
    return im.resize(shape)

class TryOnPipeline:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.temp_dir = os.path.join(base_dir, "temp_data")
        self.test_dir = os.path.join(self.temp_dir, "test")
        self.results_dir = os.path.join(base_dir, "static", "results")
        
        self.dirs = [
            os.path.join(self.test_dir, "cloth"),
            os.path.join(self.test_dir, "cloth-mask"),
            os.path.join(self.test_dir, "image"),
            os.path.join(self.test_dir, "image-parse"),
            os.path.join(self.test_dir, "openpose-img"),
            os.path.join(self.test_dir, "openpose-json"),
            self.results_dir
        ]

    def reset_directories(self):
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
        for d in self.dirs:
            os.makedirs(d, exist_ok=True)

    def run(self, model_img_path, cloth_img_path, background_prompt=None, **kwargs):
        self.reset_directories()
        
        # 1. Resize and place cloth
        cloth_im = Image.open(cloth_img_path)
        cloth_im = resize_img(cloth_im, (768, 1024))
        cloth_save_path = os.path.join(self.test_dir, "cloth", "cloth.jpg")
        cloth_im.convert("RGB").save(cloth_save_path)

        # 2. Cloth Mask (U2NET)
        print("-> Generating Cloth Mask")
        generate_mask(os.path.join(self.test_dir, "cloth"), os.path.join(self.test_dir, "cloth-mask"))

        # 3. Model Background Removal
        print("-> Removing Model Background")
        model_save_path = os.path.join(self.test_dir, "image", "model.jpg")
        model_arr, bg_mask = remove_background_from_image(model_img_path, width=768, height=1024, return_mask=True)
        model_pil = Image.fromarray(model_arr)

        # (Background replacement moved to post-processing at Step 6.5)

        model_pil.save(model_save_path)

        # 4. Human Parsing (Self-Correction-Human-Parsing)
        print("-> Running Human Parsing")
        schp_dir = os.path.join(self.base_dir, "Self-Correction-Human-Parsing")
        extractor_script = os.path.join(schp_dir, "simple_extractor.py")
        schp_checkpoint = os.path.join(schp_dir, "checkpoints", "final.pth")
        
        import sys
        subprocess.run([
            sys.executable, extractor_script,
            "--dataset", "lip",
            "--model-restore", schp_checkpoint,
            "--input-dir", os.path.join(self.test_dir, "image"),
            "--output-dir", os.path.join(self.test_dir, "image-parse")
        ], check=True, cwd=schp_dir)

        # 5. OpenPose
        print("-> Running OpenPose")
        openpose_dir = os.path.join(self.base_dir, "openpose")
        openpose_bin = os.path.join(openpose_dir, "bin", "OpenPoseDemo.exe")
        
        if not os.path.exists(openpose_bin):
            raise FileNotFoundError("OpenPoseDemo.exe not found. Please download OpenPose Portable and extract it to virtual-try-on/openpose/")

        # run openpose for json
        subprocess.run([
            openpose_bin,
            "--image_dir", os.path.join(self.test_dir, "image"),
            "--write_json", os.path.join(self.test_dir, "openpose-json"),
            "--display", "0",
            "--render_pose", "0",
            "--hand"
        ], check=True, cwd=openpose_dir)

        # run openpose for img
        subprocess.run([
            openpose_bin,
            "--image_dir", os.path.join(self.test_dir, "image"),
            "--display", "0",
            "--write_images", os.path.join(self.test_dir, "openpose-img"),
            "--hand",
            "--render_pose", "1",
            "--disable_blending", "true"
        ], check=True, cwd=openpose_dir)

        # Write test_pairs.txt
        pairs_file = os.path.join(self.temp_dir, "test_pairs.txt")
        with open(pairs_file, 'w') as f:
            f.write("model.jpg cloth.jpg")

        # 6. GAN Try-On (test.py)
        print("-> Running GAN Try-On")
        test_py = os.path.join(self.base_dir, "test.py")
        
        cmd = [
            sys.executable, test_py,
            "--name", "output",
            "--dataset_dir", self.temp_dir,
            "--dataset_list", "test_pairs.txt",
            "--checkpoint_dir", os.path.join(self.base_dir, "checkpoints"),
            "--save_dir", self.results_dir
        ]
        
        if kwargs.get('color_transfer'):
            cmd.append("--color_transfer")
            if kwargs.get('transfer_mode'):
                cmd.extend(["--transfer_mode", str(kwargs['transfer_mode'])])
            if kwargs.get('target_color'):
                cmd.extend(["--target_color", str(kwargs['target_color'])])
            if kwargs.get('texture_path'):
                cmd.extend(["--texture_path", str(kwargs['texture_path'])])
            if kwargs.get('texture_blend') is not None:
                cmd.extend(["--texture_blend", str(kwargs['texture_blend'])])
                
        subprocess.run(cmd, check=True, cwd=self.base_dir)

        result_img_path = os.path.join(self.results_dir, "output", "model.jpg_cloth.jpg")

        # 6.5. [FEATURE 3] Background Replacement (optional)
        # Done AFTER GAN processing so the GAN has a clean white background to work with!
        if background_prompt and background_prompt.strip():
            print(f"-> Replacing Background: '{background_prompt}'")
            try:
                result_pil = Image.open(result_img_path)
                # Pass the original rembg mask from Step 3
                replaced_pil = self.replace_background(result_pil, bg_mask, background_prompt.strip())
                replaced_pil.save(result_img_path)
            except Exception as e:
                print(f"Background replacement failed (continuing without): {e}")

        # 7. [FEATURE 1] Super-Resolution via Real-ESRGAN
        print("-> Running Super-Resolution (Real-ESRGAN)")
        try:
            result_img_path = self.run_super_resolution(result_img_path)
        except Exception as e:
            print(f"Super-resolution failed (using original output): {e}")

        # 8. [FEATURE 2] Gemini Fashion Critique
        print("-> Getting Gemini Fashion Critique")
        critique = None
        try:
            critique = self.get_fashion_critique(model_img_path, result_img_path)
        except Exception as e:
            print(f"Gemini critique failed: {e}")
            critique = f"⚠️ Gemini API was unable to generate a critique. Please check your API quota or endpoint. (Error: {str(e)[:100]}...)"

        return result_img_path, critique

    # ------------------------------------------------------------------ #
    # FEATURE 1: Super-Resolution (Real-ESRGAN with PIL LANCZOS fallback)
    # ------------------------------------------------------------------ #
    def run_super_resolution(self, img_path):
        """
        Upscale the try-on result using Real-ESRGAN's torch pipeline.
        Falls back to high-quality PIL 2x LANCZOS if torch/basicsr fails.

        BUG FIX: Use os.path.splitext() to build the upscaled path so that
        filenames like 'model.jpg_cloth.jpg' don't get mangled by .replace().
        """
        weights_dir = os.path.join(self.base_dir, "checkpoints", "realesrgan")
        os.makedirs(weights_dir, exist_ok=True)
        model_path = os.path.join(weights_dir, "RealESRGAN_x4plus.pth")

        if not os.path.exists(model_path):
            print("   Downloading Real-ESRGAN weights (~65MB)...")
            import urllib.request
            url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
            urllib.request.urlretrieve(url, model_path)

        # ✅ FIX: derive upscaled path correctly regardless of filename dots
        base, ext = os.path.splitext(img_path)
        upscaled_path = base + "_upscaled" + ext

        try:
            # BUGFIX 1: Monkeypatch torchvision's deleted functional_tensor module
            # This is required because basicsr imports it, but modern torchvision removed it.
            import sys
            import torchvision.transforms.functional as functional
            sys.modules['torchvision.transforms.functional_tensor'] = functional
            
            import torch
            import cv2
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            rrdb = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                           num_block=23, num_grow_ch=32, scale=4)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            upsampler = RealESRGANer(
                scale=4, model_path=model_path, model=rrdb,
                tile=400, tile_pad=10, pre_pad=0,
                half=torch.cuda.is_available(), device=device
            )
            img_cv = cv2.imread(img_path, cv2.IMREAD_COLOR)
            output, _ = upsampler.enhance(img_cv, outscale=2)
            cv2.imwrite(upscaled_path, output)
            print(f"   Real-ESRGAN torch upscale saved: {upscaled_path}")
            return upscaled_path

        except Exception as e:
            # Fallback: PIL 2x LANCZOS — zero extra dependencies
            print(f"   Real-ESRGAN unavailable ({type(e).__name__}: {e}), using PIL 2x LANCZOS fallback.")
            pil_img = Image.open(img_path).convert("RGB")
            w, h = pil_img.size
            pil_up = pil_img.resize((w * 2, h * 2), Image.LANCZOS)
            pil_up.save(upscaled_path, quality=95)
            print(f"   PIL 2x LANCZOS upscale saved: {upscaled_path}")
            return upscaled_path

    # ------------------------------------------------------------------ #
    # FEATURE 2: Gemini Fashion Critique (google-genai SDK)
    # ------------------------------------------------------------------ #
    def get_fashion_critique(self, before_img_path, after_img_path):
        from dotenv import load_dotenv
        load_dotenv(os.path.join(self.base_dir, ".env"))
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("   GEMINI_API_KEY not found in .env — skipping critique.")
            return None

        from google import genai as ggenai
        from PIL import Image as PILImage

        client = ggenai.Client(api_key=api_key)

        before = PILImage.open(before_img_path).convert("RGB")
        after  = PILImage.open(after_img_path).convert("RGB")

        prompt = (
            "You are an expert, highly critical fashion stylist. "
            "The first image shows the person BEFORE the outfit change. "
            "The second image shows the person AFTER virtually trying on a new garment. "
            "Give a very honest, constructive 2-3 sentence fashion critique and styling advice. "
            "Do not just say 'it looks good'. Tell them exactly how it fits, how the color matches their skin tone, "
            "and suggest one specific styling tip or fashion accessory to improve the overall look."
        )

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, before, after]
        )
        return response.text.strip()

    # ------------------------------------------------------------------ #
    def replace_background(self, model_pil, bg_mask_pil, prompt):
        import time
        import io
        import requests
        import numpy as np
        from dotenv import load_dotenv
        
        load_dotenv(os.path.join(self.base_dir, ".env"))
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            print("   HUGGINGFACE_TOKEN not found in .env — skipping background replacement.")
            # FALLBACK: return unchanged image
            return model_pil

        # Create a stunning background using SDXL Text-to-Image
        url = "https://router.huggingface.co/hf-inference/models/stabilityai/stable-diffusion-xl-base-1.0"
        headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}
        
        full_prompt = (
            f"beautiful background scene, {prompt}, masterpiece, 8k resolution, "
            "photorealistic lighting, highly detailed, perfect composition"
        )
        
        data = {
            "inputs": full_prompt,
            "parameters": {"width": 768, "height": 1024}
        }

        print(f"   Calling HF SDXL for background generation...")
        result_bg_bytes = None
        for attempt in range(3):
            try:
                res = requests.post(url, headers=headers, json=data, timeout=60)
                if res.status_code == 200:
                    result_bg_bytes = res.content
                    break
                elif res.status_code == 503:
                    print(f"   Model loading (Attempt {attempt+1})... waiting 5s")
                    time.sleep(5)
                else:
                    print(f"   HF API returned {res.status_code}: {res.text[:100]}")
                    time.sleep(2)
            except Exception as e:
                print(f"   HF Request failed: {e}")
                time.sleep(2)

        if not result_bg_bytes:
            print("   Failed to fetch background from HF. Skipping background swap.")
            return model_pil

        # Composite the person onto the newly generated stable diffusion background
        orig_size = model_pil.size
        try:
            from PIL import ImageFilter
            bg_img = Image.open(io.BytesIO(result_bg_bytes)).convert("RGB").resize(orig_size)
            
            # The bg_mask_pil from remove_bg returns white (255) for background, black (0) for person.
            person_mask = bg_mask_pil.convert("L").resize(orig_size)
            
            # Choke the white fringe: MaxFilter expands the white (background) into the black (person), eroding the person
            person_mask = person_mask.filter(ImageFilter.MaxFilter(3)).filter(ImageFilter.GaussianBlur(2))
            
            # PIL Image.composite uses image1 where mask is 255, and image2 where mask is 0.
            # So image1 = new AI background, image2 = original person.
            composited = Image.composite(bg_img, model_pil.convert("RGB"), person_mask)
            return composited
        except Exception as e:
            print(f"   Compositing failed: {e}")
            return model_pil


if __name__ == "__main__":
    pipeline = TryOnPipeline(os.path.dirname(os.path.abspath(__file__)))