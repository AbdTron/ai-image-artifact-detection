"""
Simple AI Image Detection GUI
Using Tkinter (Python built-in - no external branding)
"""
import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk
import torch

import config
from model import create_model
from dataset import get_val_transforms


class AIDetectorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Image Detector")
        self.root.geometry("800x700")
        self.root.configure(bg="#f0f0f0")
        self.root.resizable(True, True)
        # Remove default icon
        try:
            self.root.iconbitmap(default="")
        except:
            pass
        
        # Load model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = create_model().to(self.device)
        checkpoint = torch.load(config.CHECKPOINT_DIR / "best_model.pth", 
                               map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.transform = get_val_transforms()
        
        # Create styles for colored progress bars
        self.style = ttk.Style()
        self.style.theme_use('default')
        self.style.configure("red.Horizontal.TProgressbar", troughcolor='#ddd', background='#e74c3c')
        self.style.configure("green.Horizontal.TProgressbar", troughcolor='#ddd', background='#2ecc71')
        
        self.setup_ui()
        
    def setup_ui(self):
        # Title
        title = tk.Label(self.root, text="AI Image Detector", 
                        font=("Arial", 24, "bold"), bg="#f0f0f0")
        title.pack(pady=20)
        
        subtitle = tk.Label(self.root, 
                           text="Detect AI-Generated/AI-Edited vs Real Images",
                           font=("Arial", 12), bg="#f0f0f0", fg="#666")
        subtitle.pack()
        
        # Image frame
        self.image_frame = tk.Frame(self.root, bg="#ddd", width=400, height=300)
        self.image_frame.pack(pady=20)
        self.image_frame.pack_propagate(False)
        
        self.image_label = tk.Label(self.image_frame, text="No image loaded",
                                    bg="#ddd", font=("Arial", 14))
        self.image_label.pack(expand=True)
        
        # Buttons
        btn_frame = tk.Frame(self.root, bg="#f0f0f0")
        btn_frame.pack(pady=10)
        
        self.upload_btn = tk.Button(btn_frame, text="Upload Image", 
                                   font=("Arial", 12), command=self.upload_image,
                                   bg="#3498db", fg="white", padx=20, pady=10)
        self.upload_btn.pack(side=tk.LEFT, padx=10)
        
        self.detect_btn = tk.Button(btn_frame, text="Detect", 
                                   font=("Arial", 12), command=self.detect,
                                   bg="#2ecc71", fg="white", padx=20, pady=10,
                                   state=tk.DISABLED)
        self.detect_btn.pack(side=tk.LEFT, padx=10)
        
        # Result frame
        self.result_frame = tk.Frame(self.root, bg="#f0f0f0")
        self.result_frame.pack(pady=20, fill=tk.X, padx=50)
        
        self.result_label = tk.Label(self.result_frame, text="", 
                                    font=("Arial", 18, "bold"), bg="#f0f0f0")
        self.result_label.pack()
        
        self.confidence_label = tk.Label(self.result_frame, text="", 
                                        font=("Arial", 14), bg="#f0f0f0")
        self.confidence_label.pack(pady=5)
        
        # Progress bar for confidence
        self.progress = ttk.Progressbar(self.result_frame, length=400, mode='determinate')
        self.progress.pack(pady=10)
        
        # Details
        self.details_label = tk.Label(self.result_frame, text="", 
                                     font=("Arial", 10), bg="#f0f0f0", fg="#666")
        self.details_label.pack()
        
        self.current_image = None
        
    def upload_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.webp")]
        )
        if file_path:
            # Load and display image
            img = Image.open(file_path).convert("RGB")
            self.current_image = img
            
            # Resize for display
            display_img = img.copy()
            display_img.thumbnail((380, 280))
            photo = ImageTk.PhotoImage(display_img)
            
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo
            
            self.detect_btn.configure(state=tk.NORMAL)
            self.result_label.configure(text="")
            self.confidence_label.configure(text="")
            self.details_label.configure(text="")
            self.progress['value'] = 0
            
    def detect(self):
        if self.current_image is None:
            return
            
        # Transform and predict
        img_tensor = self.transform(self.current_image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred = outputs.argmax(dim=1).item()
        
        ai_conf = probs[0, 0].item() * 100
        real_conf = probs[0, 1].item() * 100
        
        if pred == 0:
            result = "AI-GENERATED / AI-EDITED"
            color = "#e74c3c"
            confidence = ai_conf
        else:
            result = "REAL / AUTHENTIC"
            color = "#2ecc71"
            confidence = real_conf
        
        self.result_label.configure(text=result, fg=color)
        self.confidence_label.configure(text=f"Confidence: {confidence:.1f}%")
        
        # Change progress bar color based on prediction
        if pred == 0:
            self.progress.configure(style="red.Horizontal.TProgressbar")
        else:
            self.progress.configure(style="green.Horizontal.TProgressbar")
        
        self.progress['value'] = confidence
        self.details_label.configure(
            text=f"AI Score: {ai_conf:.1f}% | Real Score: {real_conf:.1f}%"
        )


def main():
    print("Loading model...")
    root = tk.Tk()
    app = AIDetectorGUI(root)
    print("GUI ready!")
    root.mainloop()


if __name__ == "__main__":
    main()
