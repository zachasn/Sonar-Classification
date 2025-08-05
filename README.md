# Sonar-Classification Report

## Data Preprocessing
I started by loading the sonar dataset which had 207 samples with 60 features each. The last column contained labels - either "M" for metal or "R" for rock.
The main preprocessing step was encoding the labels. The neural network needs numbers, not letters. So i used LabelEncoder() to convert the text labels into numbers. 
This changed "M" to 0 and "R" to 1. Without this step, the model can't train because PyTorch expects numerical targets. I also split the data into three parts: 70% for 
training, 20% for validation, and 10% for testing. The validation set helped me pick the best parameters without touching the test set.

## Model Architectures
I built four different neural network architectures to see which one worked best:
- **Model 1:** Simple 3-layer network with ReLU activation (60→30→15→2)
- **Model 2:** Deeper 5-layer network with Tanh activation (60→30→15→7→3→2)
- **Model 3:** Complex 7-layer network with Sigmoid activation and dropout for regularization
- **TensorBoard integration:** Training metrics visualization and comparison
- **Model 4:** 6-layer network with Tanh activation (60→50→40→20→15→10→2)

I wanted to test different depths and activation functions to see what works for this problem.

## Gradient Descent Algorithms
I tested four different optimizers:
- **SGD** (Stochastic Gradient Descent) - the basic approach
- **Adam** - combines momentum with adaptive learning rates
- **RMSprop** - adjusts learning rates for each parameter
- **Adagrad** - reduces learning rates over time

SGD worked best for most models. This surprised me because Adam usually performs better. But sometimes simple methods work well.

## Hyperparameter Tuning
I tested different combinations of:
- **Learning rates:** 0.001, 0.01, 0.1
- **Batch sizes:** 16, 32, 64
- **Optimizers:** SGD, Adam, RMSprop, Adagrad

I trained each combination for 1000 epochs and picked the settings that gave the lowest validation loss. This took a long time but helped
find the best setup for each model.

## Metrics Used
I mainly used accuracy because it's easy to understand for binary classification. Accuracy tells me what percentage of predictions were correct.

I also tracked cross-entropy loss during training. Loss shows how confident the model is in its predictions. Lower loss usually means better performance.

I could have used other metrics like precision, recall, or F1-score. These would show if the model is better at detecting metal vs rock.
But accuracy worked fine for this balanced dataset where both classes appear roughly equally.

TensorBoard helped me track both accuracy and loss over time. This let me see if models were overfitting or still improving.

## Hyperparameter Search
The system evaluates:
- **Optimizers:** SGD, Adam, RMSprop, Adagrad
- **Learning rates:** 0.001, 0.01, 0.1
- **Batch sizes:** 16, 32, 64
- **Training epochs:** 1000

## Final Model Performance
Model 2 performed best with 90.48% accuracy on the test set. It used:
- 5 layers with Tanh activation
- SGD optimizer with learning rate 0.1
- Batch size 32
- Trained for 628 epochs
  
The other models got:
- **Model 1:** 69.05% accuracy
- **Model 3:** 78.57% accuracy
- **Model 4:** 80.95% accuracy

## Conclusion
The biggest surprise was that Model 2 beat the more complex models. I expected the 7-layer model with dropout to work best, but it actually did worse. This taught me that 
more layers don't always mean better performance. Sometimes simpler models generalize better.

Tanh activation worked really well for this problem. I tried ReLU and Sigmoid too, but Tanh gave the best results. Maybe the smooth gradients helped with the small dataset.

SGD optimizer was surprisingly effective. I expected Adam to perform better but I guess the simpler approach worked for this dataset.

The validation accuracy jumped around a lot during training, which is normal for small and noisy datasets. But overall, the models learned to distinguish between metal and 
rock reasonably well.

I learned that 1000 epochs was probably too many. Most models found their best performance much earlier and I could have saved time by stopping early.

## Future Work
In the future, I'll always remember to try simple models first before building complex ones. They might work just as well and train faster. I'll also remember that different 
activation functions can make a big difference. It's worth testing ReLu, Tanh, and Sigmoid to see what works best for each problem. For small datasets, I'll stick with simple 
optimizers like SGD. Adam and RMSprop might be overkill and could overfit easier.

The hyperparameter tuning method I used here will help with future projects too. Testing multiple combinations systemtically finds better solutions than guessing. 
The train/validation/test split strategy worked well. I'll keep using this approach to avoid overfitting to the test set. I'll also use TensorBoard more often now. 
Seeing the training curves helped me understand what was happening during learning. It's much better than just looking at final numbers. 
