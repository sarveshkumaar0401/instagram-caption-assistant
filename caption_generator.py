import random
import torch
from datasets import load_dataset
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import re


processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load dataset
ds = load_dataset("kkcosmos/instagram-images-with-captions", split="train")

import re

def get_trending_hashtags(context):
    keywords = re.findall(r'\w+', context.lower())
    matching_hashtags = []

    
    subset = ds.shuffle(seed=random.randint(0, 9999)).select(range(10000))

    for item in subset:
        caption = item.get("caption", "").lower()
        if any(kw in caption for kw in keywords):
            hashtags = item.get("hashtags", [])
            if isinstance(hashtags, list):
                matching_hashtags.extend(hashtags)
            elif isinstance(hashtags, str):
                matching_hashtags.extend(hashtags.split())

    if not matching_hashtags:
        fallback = ["#instagood", "#photooftheday", "#love"]
        matching_hashtags = random.sample(fallback, k=min(10, len(fallback)))

    matching_hashtags = list(set(matching_hashtags))
    random.shuffle(matching_hashtags)
    return matching_hashtags[:10]


def generate_captions_and_hashtags(image=None, text=None):
    if image:
        inputs = processor(images=image, return_tensors="pt")
        out = model.generate(**inputs)
        caption = processor.decode(out[0], skip_special_tokens=True)
    elif text:
        caption = text
    else:
        raise ValueError("No input provided")

    hashtags = get_trending_hashtags(caption)
    return {
        "Captions": caption,
        "Hashtags": hashtags
    }
