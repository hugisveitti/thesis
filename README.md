# Generative Neural Networks for Ecosystem Simulation

### Creating a town

![original](images/examples/town/rgb.png) ![gen](images/examples/town/fake_img.png)

### Town to cultivate land

![original](images/examples/town_to_ca/rgb.png) ![gen](images/examples/town_to_ca/fake_img.png)

### Expanding a forest

![original](images/examples/forest/rgb.png) ![gen](images/examples/forest/fake_img.png)

### Cultivated area to herbaceous vegetation

![original](images/examples/hv_to_ca/rgb.png) ![gen](images/examples/hv_to_ca/fake_img.png)

## Code

The main model is in code/model, the inpainting model is in code/inpaint and the landcover model in code/landcover_model. Each folder contains a README.md that explain the models in some detail. More details can be found in the thesis.

## Drawing tool

To run the drawing tool run

```
python main.py
```

but the appropriate models will have to be downloaded and stored in the correct folders.
