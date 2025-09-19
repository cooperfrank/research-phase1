"""
This module provides functionality to encode images with OpenAI's CLIP and compute cosine 
similarity matrices.
"""

import torch
import clip
from PIL import Image

class CLIPEmbedder:
    """Class to embed images using OpenAI's CLIP and compute cosine similarity between them."""

    def __init__(self, model_name="ViT-B/32", device="cpu"):
        """
        Initialize CLIPEmbedder with a specified CLIP model and device type.

        Args:
            model_name (str, optional): Name of the CLIP model to load. Defaults to "ViT-B/32".
            device (str, optional): Device to run the model on ("cpu" or "cuda"). Defaults to "cpu".
        """
        self.device = device
        self.model, self.preprocess = clip.load(model_name, device=device)


    def embed_image(self, img_path):
        """
        Encode a single image into a CLIP embedding vector.

        Args:
            image_path (str): Path to the image file.

        Returns:
            torch.Tensor: Normalized embedding of shape (1, embedding_dim)
        """
        img = self.preprocess(Image.open(img_path)).unsqueeze(0).to(self.device)

        with torch.no_grad():
            emb = self.model.encode_image(img)
            emb /= emb.norm(dim=-1, keepdim=True) # Normalize vector
        
        return emb


    def get_image_embeddings(self, img_paths):
        """
        Encode a list of images into CLIP embeddings.

        Args:
            img_paths (list: str): List of image file paths.
        
        Returns:
            torch.Tensor: Tensor of shape (n, embedding_dim) with embeddings.
        """
        return torch.cat([self.embed_image(img_path) for img_path in img_paths], dim=0)


    def calculate_similarity(self, emb1, emb2):
        """
        Compute cosine similarity between two image embeddings.

        Args:
            emb1 (torch.Tensor): Embedding vector of shape (1, embedding_dim).
            emb2 (torch.Tensor): Embedding vector of shape (1, embedding_dim).
        
        Returns:
            float: Cosine similarity score.
        """
        return (emb1 @ emb2.T).item()


    def calculate_similarities(self, embeddings):
        """
        Compute an N x N cosine similarity matrix for a set of embeddings.

        Args:
            embeddings (torch.Tensor): Tensor of shape (N, embedding_dim).

        Returns:
            torch.Tensor: Similarity matrix of shape (N, N).
        """
        return embeddings @ embeddings.T


if __name__ == "__main__":
    image_paths = [
        "collected_screenshots/fullscreen/cloud-connection_1.png", 
        "collected_screenshots/fullscreen/menu-open_1.png",
        "collected_screenshots/fullscreen/menu-open_2.png"
    ]

    embedder = CLIPEmbedder()
    embeddings = embedder.get_image_embeddings(image_paths)
    sim_matrix = embedder.calculate_similarities(embeddings)

    print(sim_matrix)