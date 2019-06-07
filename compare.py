import cv2,os,sys
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['CUDA_VISIBLE_DEVICES']='0'

import tensorflow as tf
import numpy as np
import scipy.misc
import cv2
import facenet
import detect_face


class FaceAlign(object):
    def __init__(self, model_dir = os.path.join('.','model')):
        self.__min_size = 100
        self.__threshold = [ 0.6, 0.7, 0.7 ]
        self.__factor = 0.709
        tf.Graph().as_default()
        self.__sess = tf.Session()
        with self.__sess.as_default():
            self.__pnet, self.__rnet, self.__onet = detect_face.create_mtcnn(self.__sess, model_dir)
            
    def detect(self, image):
        margin = 44
        image_size = 200
        img_size = np.asarray(image.shape)[0:2]
        bounding_boxes, _ = detect_face.detect_face(image, self.__min_size, self.__pnet, self.__rnet, self.__onet, self.__threshold, self.__factor)
        if len(bounding_boxes) < 1:
            return None
        det = np.squeeze(bounding_boxes[0,0:4])
        box_length = np.maximum(det[3]-det[1], det[2]-det[0])
        center = [np.floor((det[3]-det[1])/2)+det[1], np.floor((det[2]-det[0])/2)+det[0]]
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(center[1]-box_length/2,0)
        bb[1] = np.maximum(center[0]-box_length/2,0)
        bb[2] = np.minimum(center[1]+box_length/2,img_size[1])
        bb[3] = np.minimum(center[0]+box_length/2,img_size[0])
        """
        bb[0] = np.maximum(det[0]-margin/2, 0)
        bb[1] = np.maximum(det[1]-margin/2, 0)
        bb[2] = np.minimum(det[2]+margin/2, img_size[1])
        bb[3] = np.minimum(det[3]+margin/2, img_size[0])
        """
        cropped = image[bb[1]:bb[3],bb[0]:bb[2],:]
        aligned = scipy.misc.imresize(cropped, (image_size, image_size), interp='bilinear')
        
        return aligned
    
    def close(self):
        self.__pnet = None
        self.__rnet = None
        self.__onet = None
        self.__sess.close()
        
        
class FaceDistence(object):
    def __init__(self,img_size = 200, model_dir = os.path.join('.','model','20180512-110547.pb')):
        self.__image_size = img_size
        self.__model_dir = model_dir
    
        #Get Session And Load Model
        tf.Graph().as_default()
        self.__sess = tf.Session()
        with self.__sess.as_default():
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
        
    def close(self):
        self.__image_size = 0
        self.__model_dir = ''
        self.__sess.close() 
    
    def compare(self, src_img, dst_img):
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

class Facelike(object):
    def __init__(self):
        self.__image_size = 200
        self.__max_dist = 1.50
        self.__min_dist = 0.60
        self.__fa = FaceAlign()
        self.__fd = FaceDistence(self.__image_size)

    
    def __enter__(self):
        return self

    def __exit__(self, exec_type, exec_value, exec_trace):
        self.__fa.close()
        self.__fd.close()
    
    def face_similar(self, src_img, dst_img, detect = True, resized = True, prewhiten = True):
        if detect:
            src_img = self.__fa.detect(src_img)
            dst_img = self.__fa.detect(dst_img)
        
        if resized:
            src_img = cv2.resize(src_img, (self.__image_size, self.__image_size), interpolation=cv2.INTER_CUBIC)
            dst_img = cv2.resize(dst_img, (self.__image_size, self.__image_size), interpolation=cv2.INTER_CUBIC)
            
        if prewhiten:
            src_img = facenet.prewhiten(src_img)
            dst_img = facenet.prewhiten(dst_img)
        
        dist = self.__fd.compare(src_img, dst_img)
        
        dist = self.__max_dist if dist>self.__max_dist else dist
        dist = self.__min_dist if dist<self.__min_dist else dist
        score = 100.0-100*(dist-self.__min_dist)/(self.__max_dist-self.__min_dist)
        
        return dist, score
                
    def face_similar_with_path(self, src_path, dst_path):
        #Load Image, Detect Face and Preprocess
        src_img = scipy.misc.imread(src_path, mode='RGB')
        src_img = self.__fa.detect(src_img)
        src_img = cv2.resize(src_img, (self.__image_size, self.__image_size), interpolation=cv2.INTER_CUBIC)
        src_img = facenet.prewhiten(src_img)
        dst_img = scipy.misc.imread(dst_path, mode='RGB')
        dst_img = self.__fa.detect(dst_img)
        dst_img = cv2.resize(dst_img, (self.__image_size, self.__image_size), interpolation=cv2.INTER_CUBIC)
        dst_img = facenet.prewhiten(dst_img)        
        
        #Face Distance
        dist = self.__fd.compare(src_img, dst_img)
        #print('Face Distance: %.2f' % dist)
        
        #Trans To Percentage
        dist = self.__max_dist if dist>self.__max_dist else dist
        dist = self.__min_dist if dist<self.__min_dist else dist
        score = 100.0-100*(dist-self.__min_dist)/(self.__max_dist-self.__min_dist)
        #print('Similarity Percent: %.2f' % (score))
        
        return dist, score

flags = tf.app.flags
flags.DEFINE_string('src_path', '', 'Path of First Compared Image')
flags.DEFINE_string('dst_path', '', 'Path of Second Compared Image')
flags.DEFINE_integer('image_size', 200, 'Size of Images(Max 200)')
FLAGS = flags.FLAGS

def main(_):
    #Face Like Testing
    print(flags.FLAGS.__flags)
    if not os.path.exists(FLAGS.src_path) or not os.path.exists(FLAGS.dst_path):
        print('ERROR: Images Path Does Not Exists !')
        return
    src_path = FLAGS.src_path
    dst_path = FLAGS.dst_path
    with Facelike() as fl:
        dist, score = fl.face_similar_with_path(src_path, dst_path)
    
if __name__ == '__main__':
    tf.app.run()
    """
    #Distance Testing Without Align
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
    """
    """
    #Detect Testing
    fa = FaceAlign()
    img = cv2.imread('./data/1.jpg',1)
    croped_img, align_img = fa.align(img)
    cv2.imshow('1',img)
    cv2.imshow('2',align_img)
    cv2.imshow('3',croped_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('ycy.jpg',align_img)
    """
    """
    #Face Like Testing
    pp.print(flags.FLAGS.__flags)
    if not os.path.exists(FLAGS.src_path) or not os.path.exists(FLAGS.dst_path):
        print('ERROR: Images Path Does Not Exists !')
        return
    src_path = FLAGS.src_path
    dst_path = FLAGS.dst_path
    dst_path = os.path.join('.','YCYData','ycy.jpg')
    with Facelike() as fl:
        files = os.listdir(os.path.join('.','YCYData'))
        for f in files:
            src_path = os.path.join('.','YCYData',f)
            print('==================%s================' % f)
            dist, score = fl.face_similar(src_path, dst_path)
    """
                 
            

  








































