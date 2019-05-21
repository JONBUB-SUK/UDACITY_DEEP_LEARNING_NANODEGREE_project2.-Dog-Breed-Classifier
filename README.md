# UDACITY_DEEP_LEARNING_NANODEGREE_project2.-Dog-Breed-Classifier

# Introduction

## 1. Abstract

The purpose of this project is to make program that can classify dog breed

And if that picture is human, print the dog breed that most similar

1. make human detection program

I used cv2.CascadeClassifier

2. make dog detection program

I used transfer learning of VGG16 and used it all

3. make classifing dog's breed program

Firstly I created CNN model by scratch but it did not work well

Secondly I used transfer learning by VGG16 but changed last classifier linear model to 133 final classes

4. Make final application all functions put together

App should classify picture as human or dog

If picture is dog, should present dog's breed

If picture is human, should present most similar dog's breed


# Background Learning


### 1. Introduction to Neural Net

<img src="./images/study/Introduction_to_NeuralNet_1.jpg" width="400">
<img src="./images/study/Introduction_to_NeuralNet_2.jpg" width="400">

### 2. Gradient Descent

<img src="./images/study/Gradient_descent_1.jpg" width="400">
<img src="./images/study/Gradient_descent_2.jpg" width="400">

### 3. Training Neural Network

<img src="./images/study/Training_NeuralNet_1.jpg" width="400">
<img src="./images/study/Training_NeuralNet_2.jpg" width="400">
<img src="./images/study/Training_NeuralNet_3.jpg" width="400">
<img src="./images/study/Training_NeuralNet_4.jpg" width="400">

# Flow

## 1. Preparing Data

### 1. Loading and preparing data


```python
data_path = 'Bike-Sharing-Dataset/hour.csv'

rides = pd.read_csv(data_path)

rides.head()
```

<img src="./images/input_data_1.png" width="800">

### 2. Checking out the data

I checked and plotted for 10 days data

```python
rides[:24*10].plot(x='dteday', y='cnt')
```

<img src="./images/input_data_2.png" width="400">

### 3. Dummify variables

For example, month has 12 values (1~12)

But December doesn't mean that it has much valuable than January

Because we put numbers to our Neural Net by x (input layer),
we need to set all variables to equal value (0 or 1)

And that variables are season, weather, month, hour, weekday

```python
dummy_fields = ['season', 'weathersit', 'mnth', 'hr', 'weekday']
for each in dummy_fields:
    dummies = pd.get_dummies(rides[each], prefix=each, drop_first=False)
    rides = pd.concat([rides, dummies], axis=1)

fields_to_drop = ['instant', 'dteday', 'season', 'weathersit', 
                  'weekday', 'atemp', 'mnth', 'workingday', 'hr']
data = rides.drop(fields_to_drop, axis=1)
data.head()
```

<img src="./images/dummify_variables.png" width="800">

### 4. Scaling target variables

For example, number of rental for an hour can be 0 or even 10,000

It can have too much difference and it can make distortion for Neuaral Net calculation

So we have to scaling to equal range that have 0 mean and 1 standard deviation

That variables are total rental number, registered number, casueal number, temperature, humidity, windspeed

```python
quant_features = ['casual', 'registered', 'cnt', 'temp', 'hum', 'windspeed']
# Store scalings in a dictionary so we can convert back later
scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std
```

<img src="./images/scaling_target_variables.png" width="800">

### 5. Splitting data into training, validation, testing sets

```python
test_data = data[-21*24:]

data = data[:-21*24]

target_fields = ['cnt', 'casual', 'registered']

features, targets = data.drop(target_fields, axis=1), data[target_fields]

test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]

train_features, train_targets = features[:-60*24], targets[:-60*24]

val_features, val_targets = features[-60*24:], targets[-60*24:]
```

<img src="./images/test_features.png" width="800">
<img src="./images/test_targets.png" width="200">


## 2. Training , valiation, test

I did not included source code, please check .py files

Below is flow of parameter tunnings

### 1. Learning Rate = 0.1, Hidden Nodes = 2, Iteration = 100 (orininal set)

<img src="./images/loss/loss_train_val_1.png" width="400">
<img src="./images/result/result_1_lr=0.1,hidden=2,iteration=100.png" width="800">

We can see validation loss is down getting down at loss graph

That means it doesn't have correct architecture

So I raised the hidden nodes

### 2. Learning Rate = 0.1, Hidden Nodes = 100, Iteration = 100

<img src="./images/loss/loss_train_val_2.png" width="400">
<img src="./images/result/result_2_lr=0.1,hidden=100,iteration=100.png" width="800">

This time but validation loss exploded

So I lowered the hidden nodes

### 3. Learning Rate = 0.1, Hidden Nodes = 50, Iteration = 100

<img src="./images/loss/loss_train_val_3.png" width="400">
<img src="./images/result/result_3_lr=0.1,hidden=50,iteration=100.png" width="800">

This time, validation loss is not exploded but increased slowely

So I lowered the hidden nodes more

### 4. Learning Rate = 0.1, Hidden Nodes = 25, Iteration = 100

<img src="./images/loss/loss_train_val_4.png" width="400">
<img src="./images/result/result_4_lr=0.1,hidden=25,iteration=100.png" width="800">

Thie time, validation loss is slowly getting down

So I lowered the hidden nodes more

### 5. Learning Rate = 0.1, Hidden Nodes = 13, Iteration = 100

<img src="./images/loss/loss_train_val_5.png" width="400">
<img src="./images/result/result_5_lr=0.1,hidden=13,iteration=100.png" width="800">

This time, validation loss increased slowely

So I raised the hidden nodes a little

### 6. Learning Rate = 0.1, Hidden Nodes = 20, Iteration = 100

<img src="./images/loss/loss_train_val_6.png" width="400">
<img src="./images/result/result_6_lr=0.1,hidden=20,iteration=100.png" width="800">

This time again, validation loss increased slowely

So I raised the hidden nodes a little

### 7. Learning Rate = 0.1, Hidden Nodes = 30, Iteration = 100

<img src="./images/loss/loss_train_val_7.png" width="400">
<img src="./images/result/result_7_lr=0.1,hidden=30,iteration=100.png" width="800">

This time again, validation loss increased slowely

So I concluded 25 is adaquate

And next time, I raised learning rate to decrease error more fastly

### 8. Learning Rate = 0.5, Hidden Nodes = 25, Iteration = 100

<img src="./images/loss/loss_train_val_8.png" width="400">
<img src="./images/result/result_8_lr=0.5,hidden=25,iteration=100.png" width="800">

It was effective

So I raised more

### 9. Learning Rate = 1.0, Hidden Nodes = 25, Iteration = 100

<img src="./images/loss/loss_train_val_9.png" width="400">
<img src="./images/result/result_9_lr=1.0,hidden=25,iteration=100.png" width="800">

It excluded

So I concluded learning rate = 0.5 is adaquate

Now its time to iteration

I just put very high number to iteration

### 10. Learning Rate = 0.5, Hidden Nodes = 25, Iteration = 7000

<img src="./images/loss/loss_train_val_10.png" width="400">
<img src="./images/result/result_10_lr=0.5,hidden=25,iteration=7000.png" width="800">

Seeing loss graph, we can find val loss is inclined after 6500 iterations

So I concluded iteration = 6500 is adaquate

### 11. Learning Rate = 0.5, Hidden Nodes = 25, Iteration = 6500

<img src="./images/loss/loss_train_val_11.png" width="400">
<img src="./images/result/result_11_lr=0.5,hidden=25,iteration=6500.png" width="800">



# Conclusion & Discussion

### 1. Meaning

I already used Keras and TensorFlow library at Self Driving Car Nanodegree program

Especially at Keras, it was possible to implement not only simple Neural Net but also complex and famous architectures

even if I don't understand principle of Neural Net

At that time, I thought I understood everything of Neural Net

But this time was very good chance for me to studying Neural Net

especially the mathmatical principle of Forward propagation, Backpropagation

### 2. About architecture

This project confined architectures to just 1 hidden layer Neural Net

So there were only 3 variables I can adjust

It was learning rate, number of hidden nodes, iterations

Of course on the contrary, thanks to it, it was good time to feel the effects of that variables

But if I can change architecture more complex, I will get better result



