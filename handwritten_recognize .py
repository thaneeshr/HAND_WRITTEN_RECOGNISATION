#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install pillow mist ')


# In[2]:


from PIL import Image
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix


# In[6]:


import mnist


# In[4]:


get_ipython().system('pip install mnist')


# In[8]:


x_train =mnist.train_images()
y_train=mnist.train_labels()


# In[9]:


x_test=mnist.test_images()
y_test=mnist.test_labels()


# In[10]:


print(x_train)
print(y_train)
print(x_test)
print(y_test)


# In[11]:


print(x_train.shape)


# In[12]:


print(x_train.ndim)


# In[15]:


x_train=x_train.reshape((-1,28*28))
x_test=x_test.reshape((-1,28*28))


# In[16]:


x_train


# In[17]:


x_train[0]


# In[18]:


x_train=(x_train/256)
x_test=(x_test/256)


# In[20]:


x_train[0]


# In[21]:


clf =MLPClassifier(solver='adam',activation='relu',hidden_layer_sizes=(64,64))


# In[22]:


clf.fit(x_train,y_train)


# In[23]:


prediction=clf.predict(x_test)


# In[24]:


acc=confusion_matrix(y_test,prediction)


# In[25]:


acc


# In[27]:


def accuracy(cm):
    diagonal=cm.trace()
    elements=cm.sum()
    return diagonal/elements
    


# In[28]:


print(accuracy(acc))


# In[31]:


from PIL import Image

img = Image.open('four.png')

data = list(img.getdata())
for i in range(len(data)):
    data[i] = 255 - data[i]
print(data)


# In[32]:


data=np.array(data)/256
data


# In[34]:


p=clf.predict([data])
print(p)


# In[36]:


from PIL import Image

img = Image.open('third.png')

data = list(img.getdata())
for i in range(len(data)):
    data[i] = 255 - data[i]
print(data)


# In[37]:


p=clf.predict([data])
print(p)


# In[ ]:




