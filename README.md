# Generative AI with Stable Diffusion XL + ControlNet

This project demonstrates how to use **Stable Diffusion XL** along with **ControlNet** for image generation and manipulation.  
It leverages Hugging Face's `diffusers` library, optimized training via `accelerate`, and additional tools like `opencv` for pre-processing inputs (e.g., Canny edge maps).  

The notebook explores creating high-quality AI-generated images, with Rajasthan-themed examples.

---

## 🚀 Project Architecture

### 1. **Setup & Installation**
- Install core libraries:
  - `transformers`, `diffusers`, `accelerate`, `peft`, `bitsandbytes`
  - `controlnet_aux`, `opencv-python-headless`
  - `Pillow`, `torch`, `torchvision`, `python-slugify`
- Ensure the latest **diffusers** is pulled from GitHub for compatibility.

### 2. **Configuration**
- `accelerate` is configured for mixed precision (`fp16`) to optimize GPU performance.
- Google Drive integration available for dataset/model storage.

### 3. **Imports & Dependencies**
- **Core Libraries:** `torch`, `diffusers`, `transformers`
- **Image Processing:** `PIL`, `numpy`, `opencv`
- **Memory Management:** `gc`
- **Utilities:** `os`, `typing.List`

### 4. **Model Pipelines**
- `StableDiffusionXLControlNetPipeline`  
- `StableDiffusionXLImg2ImgPipeline`  
- `ControlNetModel`  
- `AutoencoderKL`  
- `DDPMScheduler`  

These pipelines support **text-to-image** and **image-to-image** workflows.

### 5. **Preprocessing**
- OpenCV used for **Canny edge detection** to guide ControlNet.
- PIL + NumPy used for image loading and transformations.

### 6. **Execution Flow**
1. Install and set up environment.
2. Load pretrained **Stable Diffusion XL** + **ControlNet**.
3. Apply preprocessing (edge detection).
4. Run text/image prompts through the pipeline.
5. Save and visualize results.

# 🖼️ Output Images  

👉 *Six output images will be added here:*  

  
[Final_Image_6](https://github.com/user-attachments/assets/d7d11b6c-559b-44b0-be7c-926af99838dc)
<img width="1024" height="1024" alt="Final_Image_5_Nagpur" src="https://github.com/user-attachments/assets/6eb603d7-ef67-404b-a95c-1ab592956637" />
<img width="1024" height="1024" alt="Final_Image_4_Rajasthan" src="https://github.com/user-attachments/assets/faca6c12-1820-410a-b7ea-3377a246f718" />
<img width="1024" height="1024" alt="Final_Image_3_Rajasthan" src="https://github.com/user-attachments/assets/cefa1593-ecc2-4e02-b073-783e1027fb6b" />
<img width="1024" height="1024" alt="Final_Image_2_Mumbai_Aerial" src="https://github.com/user-attachments/assets/02bbf86a-0c73-41d9-b38b-a0dde07f5874" />
<img width="1024" height="1024" alt="Final_Image_1_Mumbai_Dawn" src="https://github.com/user-attachments/assets/ec107bf3-4330-493f-a978-cf6497fa36cf" />

  
---

## 📦 Dependencies

```bash
pip install accelerate transformers peft bitsandbytes
pip install controlnet_aux opencv-python-headless
pip install Pillow torch torchvision python-slugify
pip install git+https://github.com/huggingface/diffusers.git
