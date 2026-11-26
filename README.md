# Automatic Image Colorization with Conditional GANs

A deep learning project exploring automatic grayscale-to-color image
translation using **Conditional Generative Adversarial Networks
(CGANs)**. We implemented and compared three generator architectures on
the **COCO 2017** dataset, then selected a baseline **UNet-CGAN** for
the deployed web application because of its stable training behavior and
efficient inference.

```{=html}
<p align="center">
```
`<a href="https://aicolorization.streamlit.app/">`{=html}`<strong>`{=html}Try
the live demo`</strong>`{=html}`</a>`{=html} ·
`<a href="https://github.com/sarrazer24/ai-colorization">`{=html}`<strong>`{=html}Source
code`</strong>`{=html}`</a>`{=html} ·
`<a href="https://huggingface.co/amelakh/best_cgan_V1">`{=html}`<strong>`{=html}Model
weights`</strong>`{=html}`</a>`{=html}
```{=html}
</p>
```
## Overview

Image colorization is an inherently ambiguous computer vision task: a
grayscale image does not uniquely determine the original colors of its
objects and surroundings. This project investigates whether conditional
adversarial learning can generate plausible, visually coherent colors
while preserving the structure of the input image.

Given a grayscale image, the generator predicts a corresponding RGB
image. A discriminator evaluates whether a color image is real or
generated, conditioned on the same grayscale input. The generator is
trained using adversarial and reconstruction objectives.

## Project highlights

-   Implemented and compared **three CGAN variants**.
-   Trained and evaluated the models using selected subsets of **COCO
    2017**.
-   Used **PyTorch** for model development and training.
-   Compared model capacity, validation loss, visual quality, and
    training stability.
-   Deployed the selected UNet-CGAN through an interactive **Streamlit**
    application.

## Architectures

  ---------------------------------------------------------------------------
  Model       Architecture    Training images       Generator Best validation
                                                   parameters  generator loss
  ----------- --------------- --------------- --------------- ---------------
  Model 1     Baseline                  7,000          16.66M          1.9924
              UNet-CGAN                                       

  Model 2     ResUNet-CBAM              7,000          70.05M          1.7926
              with spectral                                   
              normalization                                   

  Model 3     Scaled                    8,000          16.66M          1.9620
              UNet-CGAN                                       
  ---------------------------------------------------------------------------

### Model 1 --- Baseline UNet-CGAN

A UNet encoder-decoder generator with skip connections, paired with a
convolutional discriminator. Skip connections help retain spatial
details while the generator predicts color information.

### Model 2 --- ResUNet-CBAM

An enhanced architecture incorporating residual blocks, a Convolutional
Block Attention Module (CBAM), and spectral normalization in the
discriminator. It achieved the lowest reported validation generator
loss, but training became unstable as the discriminator grew too strong.

### Model 3 --- Scaled UNet-CGAN

The same baseline architecture as Model 1, trained with a larger
training subset (8,000 rather than 7,000 images) and a longer
early-stopping patience. It achieved a modest validation-loss
improvement over Model 1 while retaining stable training behavior.

## Experimental findings

Model 2 achieved the lowest single reported validation generator loss
(**1.7926**), but this result came from its first epoch and was followed
by unstable adversarial training. Its discriminator loss dropped
substantially while generator loss increased.

Model 3 achieved a validation generator loss of **1.9620**, compared
with **1.9924** for Model 1. The improvement from adding 1,000 training
images was modest.

For deployment, **Model 1 was selected** because it offered a practical
balance of stable training, competitive results, and a smaller generator
(approximately 16.66 million parameters). The choice was based on
deployment suitability rather than the lowest isolated validation-loss
value.

## Dataset and preprocessing

The project uses the **COCO 2017** dataset. The experiments use the
following subsets:

  Split          Models 1 & 2   Model 3
  ------------ -------------- ---------
  Training              7,000     8,000
  Validation              500       500
  Test                    500       500

The split was generated using random seed `42`.

Preprocessing steps:

1.  Resize images to **256 × 256** pixels.
2.  Convert the input image to a single grayscale channel.
3.  Keep the target image in RGB format.
4.  Convert images to tensors and normalize pixel values to `[-1, 1]`.

The model receives a one-channel grayscale image and learns to generate
a three-channel RGB image.

Dataset resources: - [COCO official website](https://cocodataset.org/) -
[COCO 2017 dataset on
Kaggle](https://www.kaggle.com/datasets/awsaf49/coco-2017-dataset)

## Training configuration

The reported training configuration includes:

  Setting                       Value
  ----------------------------- --------------------
  Framework                     PyTorch
  Input/output resolution       256 × 256
  Batch size                    32
  Maximum epochs                30
  Optimizer                     AdamW
  Generator learning rate       0.0002
  Discriminator learning rate   0.0002
  Adam β values                 0.5, 0.999
  L1 reconstruction weight      10
  Perceptual loss               Disabled (`λ = 0`)
  Learning-rate scheduler       ReduceLROnPlateau
  Random seed                   42
  Training device               CUDA-enabled GPU
  Precision                     FP32

The generator objective combines adversarial loss and L1 reconstruction
loss. The optional ConvNeXt-Tiny perceptual loss was implemented but
disabled in the reported experiments.

## Try the application

The selected Model 1 checkpoint is served through a Streamlit
application.

**[Open the Automatic Image Colorization
demo](https://aicolorization.streamlit.app/)**

The application allows users to upload a grayscale image and receive a
generated colorized result.

## Repository resources

-   **GitHub repository:**
    [Amina212004/AI_colorization](https://github.com/sarrazer24/ai-colorization)
-   **Streamlit application:**
    [aicolorization.streamlit.app](https://aicolorization.streamlit.app/)
-   **Hugging Face model repository:**
    [amelakh/best_cgan_V1](https://huggingface.co/amelakh/best_cgan_V1)
-   **Model checkpoint:**
    [best_cgan_V1.pth](https://huggingface.co/amelakh/best_cgan_V1/blob/main/best_cgan_V1.pth)
-   **Hugging Face deployment space:**
    [amelakh/Docker](https://huggingface.co/spaces/amelakh/Docker/tree/main)

### Training notebooks

  -------------------------------------------------------------------------------------------------------------------------------------
  Experiment                          Notebook
  ----------------------------------- -------------------------------------------------------------------------------------------------
  Model 1 --- Baseline UNet-CGAN      [Kaggle notebook](https://www.kaggle.com/code/kamifiza/image-colorization-dl-project)

  Model 2 --- ResUNet-CBAM            [Kaggle notebook](https://www.kaggle.com/code/kamifiza/fork-of-image-colorization-dl-project)

  Model 3 --- Scaled UNet-CGAN        [Kaggle
                                      notebook](https://www.kaggle.com/code/kamifiza/fork-of-image-colorization-dl-project-version-2)
  -------------------------------------------------------------------------------------------------------------------------------------

## Run locally

The repository includes a dependency file and deployment configuration.
To set up the project locally:

``` bash
git clone https://github.com/sarrazer24/ai-colorization.git
cd AI_colorization

python -m venv .venv
```

Activate the virtual environment:

**Windows**

``` bash
.venv\Scripts\activate
```

**macOS / Linux**

``` bash
source .venv/bin/activate
```

Install the project dependencies:

``` bash
pip install --upgrade pip
pip install -r requirements.txt
```

The report documents the Streamlit application and its deployment, but
does not specify a verified local application entry-point filename or
exact launch command. Check the repository files for the Streamlit
script and run it with:

``` bash
streamlit run <your_app_file>.py
```

Replace `<your_app_file>.py` with the actual application filename in the
repository.

For training, use the corresponding Kaggle notebooks linked above.
Training requires a CUDA-capable GPU in the reported configuration;
local inference requirements depend on the selected checkpoint and
available hardware.

## Limitations and future work

-   **Color ambiguity:** multiple plausible colorizations may exist for
    the same grayscale image.
-   **Semantic errors:** the models may assign implausible colors when
    object-level context is unclear.
-   **Dataset bias:** models trained on COCO may not generalize equally
    well to specialized domains such as medical or satellite imagery.
-   **GAN stability:** the more complex attention-based model requires
    further tuning to stabilize adversarial training.

Potential next steps include improving GAN training stability, exploring
perceptual and feature-matching losses, evaluating on larger and more
diverse datasets, testing higher resolutions, and investigating
transformer-based or user-guided colorization.

## Team

**Group 1 --- IASD**\
National Higher School of Computer Science, Sidi Bel Abbès\
Academic year: **2025--2026**\
Module: Deep Learning\
Supervisor: **Dr. N. Dif**

Project contributors: - Fezazi Amina Khadidja - Sid Lakhdar Sara
Hafeda - Soummar Inas - Zerguerras Khayra Sarra

## References

This project builds on established work in generative modeling,
image-to-image translation, and attention-based architectures:

1.  Goodfellow et al. (2014). [Generative Adversarial
    Nets](https://arxiv.org/abs/1406.2661).
2.  Isola et al. (2017). [Image-to-Image Translation with Conditional
    Adversarial Networks](https://arxiv.org/abs/1611.07004).
3.  Ronneberger et al. (2015). [U-Net: Convolutional Networks for
    Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597).
4.  Zhang et al. (2016). [Colorful Image
    Colorization](https://arxiv.org/abs/1603.08511).
5.  Woo et al. (2018). [CBAM: Convolutional Block Attention
    Module](https://arxiv.org/abs/1807.06521).
6.  Lin et al. (2014). [Microsoft COCO: Common Objects in
    Context](https://arxiv.org/abs/1405.0312).

------------------------------------------------------------------------

*This README summarizes the project and experimental results documented
in the accompanying Deep Learning project report. For implementation
details, training code, and the exact deployment setup, refer to the
repository and linked notebooks.*
