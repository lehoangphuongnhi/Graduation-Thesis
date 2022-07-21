import numpy as np 


def generate_x (n, dim):
  """Generate the features (x).
  
  Args:
    - n: the number of samples
    - dim: the number of features (feature dimensions)
    
  Returns:
    - x: (n x dim) data points sample from N(0, 1)
  """
  x = np.random.randn(n, dim)
  return x


def generate_y (x, data_type):
  """Generate corresponding label (y) given feature (x).
  
  Args:
    - x: features
    - data_type: synthetic data type (syn1 to syn6)
  Returns:
    - y: corresponding labels
  """
  # number of samples
  n = x.shape[0]
    
  # Logit computation
  if data_type == 'syn1':
    logit = np.exp(x[:, 0]*x[:, 1])
  elif data_type == 'syn2':       
    logit = np.exp(np.sum(x[:, 2:6]**2, axis = 1) - 4.0) 
  elif data_type == 'syn3':
    logit = np.exp(-10 * np.sin(0.2*x[:, 6]) + abs(x[:, 7]) + \
                   x[:, 8] + np.exp(-x[:, 9])  - 2.4)     
  elif data_type == 'syn4':
    logit1 = np.exp(x[:, 0]*x[:, 1])
    logit2 = np.exp(np.sum(x[:, 2:6]**2, axis = 1) - 4.0) 
  elif data_type == 'syn5':
    logit1 = np.exp(x[:, 0]*x[:, 1])
    logit2 = np.exp(-10 * np.sin(0.2*x[:, 6]) + abs(x[:, 7]) + \
                    x[:, 8] + np.exp(-x[:, 9]) - 2.4) 
  elif data_type == 'syn6':
    logit1 = np.exp(np.sum(x[:,2:6]**2, axis = 1) - 4.0) 
    logit2 = np.exp(-10 * np.sin(0.2*x[:, 6]) + abs(x[:, 7]) + \
                    x[:, 8] + np.exp(-x[:, 9]) - 2.4) 
    
  # For syn4, syn5 and syn6 only
  if data_type in ['syn4', 'syn5', 'syn6']:
    # Based on X[:,10], combine two logits        
    idx1 = (x[:, 10]< 0)*1
    idx2 = (x[:, 10]>=0)*1    
    logit = logit1 * idx1 + logit2 * idx2    
        
  # Compute P(Y=0|X)
  prob_0 = np.reshape((logit / (1+logit)), [n, 1])
    
  # Sampling process
  y = np.zeros([n, 2])
  y[:, 0] = np.reshape(np.random.binomial(1, prob_0), [n,])
  y[:, 1] = 1-y[:, 0]

  return y


def generate_ground_truth(x, data_type):
  """Generate ground truth feature importance corresponding to the data type
     and feature.
  
  Args:
    - x: features
    - data_type: synthetic data type (syn1 to syn6)
  Returns:
    - ground_truth: corresponding ground truth feature importance
  """

  # Number of samples and features
  n, d = x.shape

  # Output initialization
  ground_truth = np.zeros([n, d])
        
  # For each data_type
  if data_type == 'syn1':
    ground_truth[:, :2] = 1
  elif data_type == 'syn2':
    ground_truth[:, 2:6] = 1
  elif data_type == 'syn3':
    ground_truth[:, 6:10] = 1
        
  # Index for syn4, syn5 and syn6
  if data_type in ['syn4', 'syn5', 'syn6']:        
    idx1 = np.where(x[:, 10]<0)[0]
    idx2 = np.where(x[:, 10]>=0)[0]
    ground_truth[:, 10] = 1
        
  if data_type == 'syn4':        
    ground_truth[idx1, :2] = 1
    ground_truth[idx2, 2:6] = 1
  elif data_type == 'syn5':        
    ground_truth[idx1, :2] = 1
    ground_truth[idx2, 6:10] = 1
  elif data_type == 'syn6':        
    ground_truth[idx1, 2:6] = 1
    ground_truth[idx2, 6:10] = 1
        
  return ground_truth

    
def generate_dataset(n = 10000, dim = 11, data_type = 'syn1', seed = 0):
  """Generate dataset (x, y, ground_truth).
  
  Args:
    - n: the number of samples
    - dim: the number of dimensions
    - data_type: synthetic data type (syn1 to syn6)
    - seed: random seed
    
  Returns:
    - x: features
    - y: labels
    - ground_truth: ground truth feature importance
  """

  # Seed
  np.random.seed(seed)

  # x generation
  x = generate_x(n, dim)
  # y generation
  y = generate_y(x, data_type)
  # ground truth generation
  ground_truth = generate_ground_truth(x, data_type)
  
  return x, y, ground_truth