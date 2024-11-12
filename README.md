## Sloth or Pain au Chocolat: A Deep Learning Classification Model
This project demonstrates the use of deep learning techniques to classify images into one of two categories: "Sloth" or "Pain au Chocolat". The model leverages transfer learning using a pre-trained ResNet-18 model, with fine-tuning and feature extraction techniques to handle the binary image classification task.

### Project Overview
In this project, we apply transfer learning to classify images based on whether they depict a sloth or a pain au chocolat (a French pastry). We make use of a pre-trained convolutional neural network (ResNet-18) and adapt it for our binary classification task by modifying its final layers.

The project includes the following main components:
* Data Preprocessing: We use torchvision transforms to prepare and augment images for training.
* Model Training: We use the pre-trained ResNet-18 model and adapt it for binary classification, using fine-tuning and feature extraction techniques.
* Model Evaluation: The model is evaluated on a validation set, and we monitor performance (accuracy and loss) during training.

### Key learning Points 

### Preprocessing of images involving `pytorch` and `torchvision`.

In deep learning, especially for image classification tasks, preprocessing the input images is a crucial step to ensure that the data is in the right format for the model to process efficiently. The preprocessing involves several transformations, which can be done using torchvision.transforms. In the provided code, two different preprocessing pipelines are defined for the training and validation datasets.

Training Preprocessing (example used): 
```
transforms.RandomResizedCrop(224),
transforms.RandomHorizontalFlip(),
transforms.ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
```
   * RandomResizedCrop(224): Randomly crops the image and resizes it to a fixed size of 224x224 pixels. This helps augment the dataset and prevent overfitting by introducing variability.
   * RandomHorizontalFlip(): Randomly flips the image horizontally with a probability of 0.5. This is another form of data augmentation that introduces variability.
   * ToTensor(): Converts the image into a tensor, which is the format that PyTorch works with. It changes the image from a NumPy array or a PIL image into a PyTorch tensor, which is a multi-dimensional array.
   * Normalize(mean, std): Normalizes the tensor by subtracting the mean and dividing by the standard deviation. The values [0.485, 0.456, 0.406] and [0.229, 0.224, 0.225] are pre-defined for ImageNet, a commonly used dataset for pre-trained models. This normalization step ensures the images have a consistent range of pixel values that the model can process efficiently.

### Different components of a deep learning set-up - namely model, criterion, optimizer and scheduler. 
In deep learning, there are several key components that come together to form a complete model-training pipeline. Here's an overview of these components:

#### Model
* The model defines the architecture that takes in the input data, processes it through layers of computations (such as convolutions, activations, etc.), and outputs the final result. In this example, the pre-trained ResNet-18 model from torchvision is used. ResNet-18 is a convolutional neural network (CNN) with 18 layers, pre-trained on ImageNet, which allows us to leverage its learned features for our specific task.
```
model_conv = torchvision.models.resnet18(pretrained=True)
```
   * The model is initialized with pre-trained weights (pretrained=True), meaning that the model has already learned useful features from a large dataset (ImageNet).
   * After loading the model, the final fully connected layer (model_conv.fc) is modified to suit the binary classification task (i.e., outputting 2 classes instead of the original 1000 classes used in ImageNet).

#### Criterion (Loss Function)
* The loss function defines how well the model's predictions match the true labels. It measures the error (or loss) of the predictions, which is then used to adjust the model's parameters during training. In this example, the CrossEntropyLoss function is used:
```
criterion = nn.CrossEntropyLoss()
```
   * CrossEntropyLoss is commonly used for classification tasks and calculates the difference between the predicted probability distribution and the true label.
   * The model attempts to minimize the loss during training by adjusting its parameters.

#### Optimizer
* The optimizer is responsible for updating the model's parameters based on the computed gradients. In this code, the SGD (Stochastic Gradient Descent) optimizer is used:
```
optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
```
   * SGD is a popular optimization algorithm that iteratively updates the model parameters in the direction that reduces the loss. It does this using the gradients computed from the loss function.
   * The learning rate (lr=0.001) determines how large the step is during each update.
   * The momentum (momentum=0.9) helps accelerate the optimization by smoothing the updates.

#### Learning Rate Scheduler: The learning rate scheduler adjusts the learning rate during training to improve convergence. The StepLR scheduler is used in the code:
```
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)
```
   * StepLR decreases the learning rate by a factor of gamma=0.1 every step_size=7 epochs. This is a form of learning rate decay, which helps the model converge more effectively by reducing the learning rate over time as the model gets closer to an optimal solution.

### Set-up 
1. Installation:
* Download the repo onto your local device and use your favourite IDE in assessing the codebase. 
* access the `notebook.ipynb`

2. set up a virtual environment for the code to run in.

3. install the following packages in your virtual environment: 
```
pip install numpy
pip install matplotlib
pip install torch
pip install torchvision
```

4. run the `notebook.ipynb` file 

### Conclusion
This project provides a simple and effective way to perform binary image classification using transfer learning and the ResNet-18 model. It demonstrates the power of using pre-trained models to tackle new problems with limited data, enabling faster training and better performance.

