import requests
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import torch

url = 'https://storage.googleapis.com/sfr-vision-language-research/LAVIS/assets/merlion.png' 
print("Img url: " + url)
image = Image.open(requests.get(url, stream=True).raw).convert('RGB')   

processor = AutoProcessor.from_pretrained("Salesforce/blip2-opt-2.7b")
# by default `from_pretrained` loads the weights in float32
# we load in float16 instead to save memory
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b") 

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def run_prompt(prompt, question=False):
    if question:
        prompt = "Question: " + prompt + " Answer: "
    print("Prompt:" + prompt)
    if prompt == "":
        inputs = processor(image, return_tensors="pt").to(device, torch.float16)
    else:
        inputs = processor(image, text=prompt, return_tensors="pt").to(device, torch.float16)


    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    print("Result: " + generated_text)
    return(generated_text)

run_prompt("") # return the default descriptions
if run_prompt("is this a painting", question=True) in ["yes", "painting"]
    run_prompt("the painting is composed of")
    run_prompt("the colorscheme is")
    run_prompt("the iconography of the painting is")
    run_prompt("the painting invokes feelings of")
    run_prompt("there is golden ratio in the image at")
    run_prompt("in the image is something special at the position of")
    run_prompt("what is the most prominent object in the image")
    run_prompt("what object in the image is missing", question=True)
    run_prompt("which architecture fits to the image", question=True)
    run_prompt("how many elements has the image", question=True)
    run_prompt("what is the prominent feeling attached to the image", question=True)
    run_prompt("the colour scheme is")
    run_prompt("what in the image is scary", question=True)
    run_prompt("in the future the objects in the image will")
    run_prompt("the most interesting object in the image is")
    run_prompt("at the position of a golden ratio there is")
    run_prompt("in the center of the image is")
    run_prompt("the red object in the image is")
    run_prompt("the most frequent colour is")
    run_prompt("you find information about the image in the book with the title")
    run_prompt("three important iconographic features are")
    run_prompt("everyone likes about the image that")
    run_prompt("it is strange in the picture that")
    run_prompt("at the edges of the painting there are")
    run_prompt("in the center of the painting there is")
    run_prompt("what kinds of animals are in the picture", question=True)
    run_prompt("nobody knows that in the image you find")
    run_prompt("it is lovely that there are so many")
