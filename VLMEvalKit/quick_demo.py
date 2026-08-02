# Demo
from vlmeval.config import supported_VLM

model = supported_VLM['NEO-2B-SFT']()
# model = supported_VLM['NEO-9B-SFT']()


# Model Name	Model Weight
# NEO-2B-PT	🤗 NEO-2B-PT HF link
# NEO-2B-MT	🤗 NEO-2B-MT HF link
# NEO-2B-SFT	🤗 NEO-2B-SFT HF link
# NEO-9B-PT	🤗 NEO-9B-PT HF link
# NEO-9B-MT	🤗 NEO-9B-MT HF link
# NEO-9B-SFT	🤗 NEO-9B-SFT HF link

# check model's type:
print(f"Model Type: {type(model)}")
# Forward Single Image
ret = model.generate(['assets/apple_1.png', 'What is in this image?'])
print(ret)
# Forward Multiple Images
ret = model.generate(['assets/apple_1.png', 'assets/apple_2.png', 'How many apples are there in total in the provided images? '])
print(ret)
