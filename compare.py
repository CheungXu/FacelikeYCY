import cv2,os,sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#os.environ['CUDA_VISIBLE_DEVICES']='0'

import tensorflow as tf
import numpy as np
import scipy.misc
import cv2
import facenet

class FaceDistence(object):
  def __init__(self,img_size = 200, model_dir = os.path.join('.','model','20180512-110547.pb')):
    self.__image_size = 200
    self.__model_dir = model_dir
    
    #Get Session And Load Model
    tf.Graph().as_default()
    self.__sess = tf.Session()
    facenet.load_model(self.__model_dir)
    
    #Create Placeholder
    self.__images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    self.__embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    self.__phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    self.__embedding_size = self.__embeddings.get_shape()[1]
    
  def __enter__(self):
    return self
    
  def __exit__(self, exec_type, exec_value, exec_trace):
    self.__image_size = 0
    self.__model_dir = ''
    self.__sess.close()
    
  def compare(self, src_path, dst_path):
    #Load Image and Preprocess
    src_img = scipy.misc.imread(src_path, mode='RGB')
    src_img = cv2.resize(src_img, (self.__image_size, self.__image_size), interpolation=cv2.INTER_CUBIC)
    src_img = facenet.prewhiten(src_img)
    dst_img = scipy.misc.imread(dst_path, mode='RGB')
    dst_img = cv2.resize(dst_img, (self.__image_size, self.__image_size), interpolation=cv2.INTER_CUBIC)
    dst_img = facenet.prewhiten(dst_img)
    
    #Use Facenet to Calculate Face Feature Vector
    src_vector = np.zeros(self.__embedding_size)
    src_reshape = src_img.reshape(-1,self.__image_size,self.__image_size,3)
    src_vector = self.__sess.run(self.__embeddings, feed_dict={self.__images_placeholder: src_reshape, self.__phase_train_placeholder: False })[0]
    dst_vector = np.zeros(self.__embedding_size)
    dst_reshape = dst_img.reshape(-1,self.__image_size,self.__image_size,3)
    dst_vector = self.__sess.run(self.__embeddings, feed_dict={self.__images_placeholder: dst_reshape, self.__phase_train_placeholder: False })[0]
    
    #Calculate Face Vector Distence
    dist = np.sqrt(np.sum(np.square(src_vector-dst_vector)))
    
    return dist.item()



if __name__ == '__main__':
    src_path = ''
    dst_path = ''
    min_dist = 0.75
    max_dist = 1.40
    if len(sys.argv) < 3:
        src_path = os.path.join('.','data','AF2.jpg')
        dst_path = os.path.join('.','data','ycy.jpg')
    
    with FaceDistence() as fd:
        dist = fd.compare(src_path, dst_path)
        print(dist)
        dist = max_dist if dist>max_dist else dist
        dist = min_dist if dist<min_dist else dist
        score = 100.0-100*(dist-min_dist)/(max_dist-min_dist)
        print('Similarity Percent: %.2f' % (score))
            

  








































