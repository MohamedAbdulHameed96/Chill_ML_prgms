  
pip install numpy

  
import numpy as np

  
one_D_array = np.array([1,2,3,4,5])

  
one_D_array

  
two_D_array = np.array([[1, 2,3], [3, 4, 5]])    

  
print(one_D_array.shape)
print(two_D_array.shape)

  
two_D_array.shape

  
#array reshaping
print(two_D_array.reshape(3,2))

  
#checking dimensions
print(two_D_array.ndim)

  
two_D_array

  
#Accessing elements array position
print(two_D_array[1,2])

  
#Slicing
print(two_D_array[1:])

  
print(two_D_array[:,1:])


