# Anime Face Generator with GANs

This project implements a Generative Adversarial Network (GAN) to generate anime-style faces using the Anime Face Dataset. The GAN consists of a Generator and a Discriminator, trained adversarially to create realistic anime face images from random noise.

## Output
![image](https://github.com/user-attachments/assets/b6e322aa-cb69-4bb2-992a-403b6b42266e)


## Features

- Downloads and preprocesses the Anime Face Dataset from Kaggle.
- Implements a customizable Generator and Discriminator.
- Visualizes training progress by saving generated images during training.
- Utilizes GPU acceleration for efficient training.

## Prerequisites

- Python 3.8 or higher
- CUDA-enabled GPU (optional for faster training)
- Required Python libraries:
  - `torch`
  - `torchvision`
  - `matplotlib`
  - `opendatasets`
  - `tqdm`

## Training

1. Customize hyperparameters like `batch_size`, `image_size`, `latent_size`, and `epochs` in the script.

2. Run the training script:
   ```bash
   python train.py
   ```

3. The generated images will be saved in the `generated/` folder.

## Model Architecture

### Generator

The Generator uses transposed convolution layers to upsample random noise into high-resolution images.

### Discriminator

The Discriminator is a convolutional neural network that classifies images as real or fake.

## Visualization

- Use `save_samples()` to visualize generated images during training.
- Example visualization:
  ```python
  from torchvision.utils import make_grid
  import matplotlib.pyplot as plt

  fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)
  save_samples(0, fixed_latent, show=True)
  ```

## Results

Generated images improve in quality as training progresses. Saved samples from each epoch can be found in the `generated/` folder.

## Contributing

Feel free to open issues or create pull requests for improvements or bug fixes.

## License

This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgments

- Kaggle for providing the [Anime Face Dataset](https://www.kaggle.com/datasets/splcher/animefacedataset).
- PyTorch for its powerful deep learning framework.

