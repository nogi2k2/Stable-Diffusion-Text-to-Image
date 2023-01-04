import tkinter as tk
import customtkinter as ctk

import torch
from torch.cuda.amp import autocast

from diffusers import StableDiffusionPipeline
from PIL import ImageTk, Image
from authtoken import auth_token

app = tk.Tk()
app.geometry("532x632")
app.title("Stable Diffusion TTI")
ctk.set_appearance_mode("dark")
app.iconbitmap("logo.ico")


#modules
prompt = ctk.CTkEntry(master=app, height=40, width=512, text_color="black", fg_color="white") 
prompt.place(x=10, y=10)


img_display = ctk.CTkLabel(master=app, height=512, width=512)
img_display.place(x=10, y=110)

device=torch.cuda.is_available()
modelid="CompVis/stable-diffusion-v1-4"
pipeline = StableDiffusionPipeline.from_pretrained(modelid, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token)
pipeline.to("cuda")

def SD():
    with autocast(device):
        image=pipeline(prompt.get(), guidance_scale=8.5)['sample'][0]
    image.save('SD_Image.png')
    img = ImageTk.PhotoImage(image)
    img_display.configure(image=img)

txt_button = ctk.CTkButton(master=app, height=40, width=120, text_color="white", fg_color="blue", command=SD)
txt_button.configure(text="Generate")
txt_button.place(x=206, y=60)

app.mainloop()