# Mini-CNN

This repo hosts a compact MobileFaceNet-inspired model (`mini-cnn.py`) for extracting 512D face embeddings along with the associated tester scripts that handle detection, alignment, preprocessing, and enrollment. It’s effectively a stripped-down, efficient variation of MobileFaceNet made to be easy to experiment with and deploy for small-scale recognition tasks.

The notebook (`cnn.ipynb`) shows how the model was originally trained and evaluated, and the `tester/` directory demonstrates how to turn images into normalized embeddings and match them via cosine similarity.
