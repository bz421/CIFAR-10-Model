**A model that achieves over 95% accuracy on the CIFAR-10 dataset.**

The final model is saved in the `model5-distilled` directory.

The original code used to train a ResNet-152 model is found in `CIFAR10-model5.ipynb`.
It was fine-tuned using `CIFAR10-model5-finetune.ipynb` and then distilled into a ResNet-18 model
using `CIFAR10-model5-distill.ipynb`. The model was trained on Google Colab's T4 GPU utilizing CUDA architecture.

Previous versions of the model are attached in `old-models`. Most of them were built on a conventional CNN architecture.

Additional data gathered from the final model can be found in `misclassifications`.

If you want to test the model yourself with your own images, run `GUI.py`. The possible classes are:
- plane
- car
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

The last uploaded image can be found in `images`.