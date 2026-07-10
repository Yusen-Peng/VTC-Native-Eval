# Demo
from vlmeval.config import supported_VLM
model = supported_VLM['NEO-2B-SFT']()

# check model's type:
print(f"Model Type: {type(model)}")
# Forward Single Image
ret = model.generate(['assets/apple_1.jpg', 'What is in this image?'])
print(ret)
# Forward Multiple Images
ret = model.generate(['assets/apple_1.jpg', 'assets/apple_2.jpg', 'How many apples are there in total in the provided images? '])
print(ret)
