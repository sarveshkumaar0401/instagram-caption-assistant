import random
from datasets import load_dataset

# Load dataset only once
ds = load_dataset("kkcosmos/instagram-images-with-captions", split="train")

def get_trending_hashtags(context):
    keywords = context.lower().split()
    matching_hashtags = []

    for item in ds.select(range(1000)):
        caption = item.get("caption", "").lower()
        if any(word in caption for word in keywords):
            hashtags = item.get("hashtags", [])
            matching_hashtags.extend(hashtags)

    if not matching_hashtags:
        matching_hashtags = ["#instagood", "#photooftheday", "#love"]

    matching_hashtags = list(set(matching_hashtags))
    random.shuffle(matching_hashtags)
    return matching_hashtags[:10]
