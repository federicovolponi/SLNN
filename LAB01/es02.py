#Part 1

Dataset2 = h5py.File('/content/drive/MyDrive/Colab Notebooks/Federico/Computer LAB 1/Lab1_Ex_2_wine.hdf5')

Data = np.array(Dataset2.get('Dataset'))

Train_Set = Data[:200,:] 
Test_Set = Data[200:,:] 

#To be completed by the student
accuracy_test = []
accuracy_train = []
prediction_test = 0
prediction_train = 0
n_features = 11
#Function for calculating euclidean distances (2 dimensions)
def euclidean_distance(p, q):
  dist = np.sqrt(np.sum(np.square(p - q)))
  return dist

#Function for K-NN Algorithm
def K_NNClassifier(sample, k):
  dist = []
  neighbors = []
  neighbors_index = []
  #Create a vector with all the distances
  for i in range(len(Train_Set)):
    dist.append(euclidean_distance(sample, Train_Set[i, :n_features]))

  neighbors_index = np.argsort(dist)
  for i in range(k):
    neighbors.append(Train_Set[neighbors_index[i],n_features]) #Vector containing the classes of the k-nearest elements
  
  classes,counts = np.unique(neighbors, return_counts=True) #Evaluate the majority class
  return classes[np.argmax(counts)]

#Calculate correct prediction for test and train sets 
for k in range(1,200,2):
  prediction_test = 0
  prediction_train = 0
  for i in range(len(Test_Set)):
    if K_NNClassifier(Test_Set[i, :n_features], k) == Test_Set[i, n_features]:
      prediction_test += 1
    if K_NNClassifier(Train_Set[i, :n_features], k) == Train_Set[i, n_features]:
      prediction_train += 1
  accuracy_test.append(prediction_test/len(Test_Set))
  accuracy_train.append(prediction_train/len(Train_Set))

k = range(1,200, 2)
fig, ax = plt.subplots()
ax.plot(k, accuracy_test, "b", label="Test accuracy")
ax.plot(k, accuracy_train, "r", label="Train accuracy")
plt.xlabel("K")
plt.ylabel("Accuracy")
ax.legend()
plt.grid()
plt.show()
