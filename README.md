# Malaria Detection using CNN

## **Overview**<br>
  This project aims to detect malaria from cell images using a Convolutional Neural 
   Network (CNN). The model is trained on a dataset containing images of parasitized and 
   uninfected cells. The notebook includes data preprocessing, model building, training, 
   and evaluation.
## Dataset
The dataset consists of images categorized into two classes:
 - Parasitized
 - Uninfected
 - Each image is a high-resolution scan of a blood smear.
   
## Requirements
To run the notebook, the following Python packages are required:

 - TensorFlow
 - Keras
 - NumPy
 - Matplotlib
 - Scikit-learn
You can install these packages using pip:
```
pip install tensorflow keras numpy matplotlib scikit-learn

```
## Structure
**1.Data Visualization**


![malari visual](https://github.com/Mahmedorabi/Malari_CNN_Model/assets/105740465/c686abb6-ffec-4149-9de2-7c4cd923e3e8)


**2.Data Loading and Preprocessing**

- Load and preprocess the images.
- Split the data into training and testing sets.
  ```
  data_generator=ImageDataGenerator(rescale=1./255,
                                  validation_split=0.2)


  train_data=data_generator.flow_from_directory('D:/Project AI/cell_images',
                                              batch_size=32,
                                              subset='training',
                                              shuffle=True,
                                              class_mode='binary',
                                              target_size=(224,224))

  test_data=data_generator.flow_from_directory('D:/Project AI/cell_images',
                                             class_mode='binary',
                                             shuffle=True,
                                             subset='validation',
                                             target_size=(224,224),
                                             batch_size=1)
  ```
    
**3.Model Building**

- Define the CNN architecture.
- Compile the model with appropriate loss function and optimizer
```
# Model Buliding
model=Sequential()

model.add(Conv2D(filters=32,kernel_size=3,padding='same',activation='relu',input_shape=[224,224,3]))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.20))

model.add(Conv2D(filters=64,kernel_size=3,padding='same',activation='relu'))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='sigmoid'))


# Compile model

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

```
**4.Model Training**

- Train the model on the training set.
- Validate the model on the validation set.
- Using Early Stopping to decrease Overfitting
  ```
  es=EarlyStopping(patience=2)
  model_hist=model.fit(train_data,validation_data=test_data,epochs=4,callbacks=[es])
  ```
  **4.Model Evaluation**

- Evaluate the model performance on the test set.
- Plot training and validation accuracy and loss.

![model malaria](https://github.com/Mahmedorabi/Malari_CNN_Model/assets/105740465/8a2817fd-87cd-43d2-98f0-e812dd2745be)
**5.Prediction**
- Predict the class of a new image.

![predication malaria](https://github.com/Mahmedorabi/Malari_CNN_Model/assets/105740465/b1beeaf7-9e1f-4695-b025-c481f7b4a7e2)


## Usage
**1.Load the dataset**

- Make sure the dataset is in the specified directory structure.
- Modify the paths in the notebook if necessary.
**2.Run the notebook**

- Execute each cell sequentially to preprocess the data, build the model, train it, and evaluate its performance.
  
**3.Predict new images**

- Use the provided functions to predict the class of new images.
## Example
Here is an example of how to predict the class of a new image using the trained model:
```
ef predicate_image(testing_img,Actual_label):
    img_test=image.load_img(testing_img,target_size=(224,224))
    img_arr=image.img_to_array(img_test)/255
    input_image=img_arr.reshape((1,img_arr.shape[0],
                                 img_arr.shape[1],
                                 img_arr.shape[2]))
    
    predict_label=np.argmax(model.predict(input_image))
    predict_class=classes_map[predict_label]
    
    plt.figure(figsize=(10,8))
    plt.imshow(img_arr)
    plt.title(f"Actual Label is: {Actual_label} | Predict Label is: {predict_class}",fontsize=15)
    plt.grid()
    plt.axis('off')
    plt.show()
```
## Notes
- Ensure that the dataset is properly labeled and organized.
- Experiment with different architectures and hyperparameters to improve performance.

## Conclusion
This notebook provides a comprehensive guide to building a CNN model for malaria detection from cell images. It includes data preprocessing, model training, evaluation, and prediction steps.







  


