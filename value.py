from skimage.measure import compare_ssim,compare_mse,compare_psnr
import cv2,os
import tensorflow as tf
import numpy as np
import scipy.misc
import cv2
import facenet
import seaborn as sns
import matplotlib.pyplot as plt  

class value(object):
  def __init__(self,src_path,dst_path,img_size):
    self.src_path = src_path
    self.dst_path = dst_path
    self.image_size = 200
    self.file_list = [fn for fn in os.listdir(src_path) if '.jpg' in fn]
    if not self.file_list == [fn for fn in os.listdir(dst_path) if '.jpg' in fn]:
       print("File Error !")

  def SSIM(self, img1, img2):
    score = compare_ssim(img1, img2, win_size=5, multichannel=True, gaussian_weights=True, sigma=1.5, use_sample_covariance=False)
    #print("SSIM: {}".format(score))
    return score

  def MSE(self, img1,img2):
    mse = compare_mse(img1, img2)
    #print("MSE:  {}".format(mse))
    return mse
  
  def PSNR(self, img1,img2):
    score = compare_psnr(img1,img2)
    #print("PSNR: {}".format(score))
    return score

  def FACENET(self, modeldir):
    
    tf.Graph().as_default()
    sess = tf.Session()
    facenet.load_model(modeldir)
    images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
    embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
    phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
    embedding_size = embeddings.get_shape()[1]

    value = []
    total = len(self.file_list)
    num = 0
    for fn in self.file_list:
      num += 1
      print(str(num)+'/'+str(total),fn)

      scaled_reshape = []
      image1 = scipy.misc.imread(self.src_path+fn, mode='RGB')
      image1 = cv2.resize(image1, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)

      image1 = facenet.prewhiten(image1)

      scaled_reshape.append(image1.reshape(-1,self.image_size,self.image_size,3))
      emb_array1 = np.zeros((1, embedding_size))
      emb_array1[0, :] = sess.run(embeddings, feed_dict={images_placeholder: scaled_reshape[0], phase_train_placeholder: False })[0]

      image2 = scipy.misc.imread(self.dst_path+fn, mode='RGB')
      image2 = cv2.resize(image2, (self.image_size, self.image_size), interpolation=cv2.INTER_CUBIC)
      image2 = facenet.prewhiten(image2)
      scaled_reshape.append(image2.reshape(-1,self.image_size,self.image_size,3))
      emb_array2 = np.zeros((1, embedding_size))
      emb_array2[0, :] = sess.run(embeddings, feed_dict={images_placeholder: scaled_reshape[1], phase_train_placeholder: False })[0]
      
      dist = np.sqrt(np.sum(np.square(emb_array1[0]-emb_array2[0])))

      value.append(dist.item())

    return value
  
  def maxminavg(self,value):
    max_v = -10000
    min_v = 10000
    summ = 0.0
    for v in value:
      if v > max_v:
        max_v = v
      if v < min_v:
        min_v = v
      summ += v
    return (str(max_v), str(min_v), str(summ/len(value)))

  def value(self, modeldir):
    mse = []
    psnr = []
    ssim = []
    for fn in self.file_list:
      imageA = cv2.imread(self.src_path+fn,1)
      imageB = cv2.imread(self.dst_path+fn,1)
      mse.append(self.MSE(imageA,imageB))
      psnr.append(self.PSNR(imageA,imageB))
      ssim.append(self.SSIM(imageA,imageB))
    facenet = self.FACENET(modeldir)
    with open(self.dst_path+'value_res.txt','w') as f:
      for i in range(len(self.file_list)):
        f.write(self.file_list[i]+' '+str(mse[i])+' '+str(psnr[i])+' '+str(ssim[i])+' '+str(facenet[i])+'\n')
    with open(self.dst_path+'statistic_res.txt','w') as f:
      res = self.maxminavg(mse)
      f.write('MSE  '+res[0]+' '+res[1]+' '+res[2]+'\n')
      res = self.maxminavg(psnr)
      f.write('PSNR '+res[0]+' '+res[1]+' '+res[2]+'\n')
      res = self.maxminavg(ssim)
      f.write('SSIM '+res[0]+' '+res[1]+' '+res[2]+'\n')
      res = self.maxminavg(facenet)
      f.write('FCNT '+res[0]+' '+res[1]+' '+res[2]+'\n')


def crop(img_path,dst_path,img_size):
  img = cv2.imread(img_path)
  index = int(img.shape[0]/img_size)
  num = 0
  for i in range(index):
    for j in range(index):
      num += 1
      tmp = img[j*img_size:(j+1)*img_size,i*img_size:(i+1)*img_size,:]
      cv2.imwrite(dst_path+str(num)+'.jpg',tmp)

def peast_crop(mask_img,gene_img,dst_path,img_size):
  mask_im = cv2.imread(mask_img,1)
  gene_im = cv2.imread(gene_img,1)
  index = int(mask_im.shape[0]/img_size)
  num = 0
  for i in range(index):
    for j in range(index):
      num += 1
      mask_tmp = mask_im[j*img_size:(j+1)*img_size,i*img_size:(i+1)*img_size,:]
      gene_tmp = gene_im[j*img_size:(j+1)*img_size,i*img_size:(i+1)*img_size,:]
      #mask_tmp[28:36,21:29,:] = gene_tmp[28:36,21:29,:]
      mask_tmp[32:96,18:82,:] = gene_tmp[32:96,18:82,:]
      if not os.path.exists(dst_path):
        os.mkdir(dst_path)
      cv2.imwrite(dst_path+str(num)+'.jpg',mask_tmp)

def crop_all(path, img_size):
  crop(path+'real_img.png',path+'/real/',img_size)
  crop(path+'mask_img.png',path+'/mask/',img_size)
  crop(path+'gene_img.png',path+'/generate/',img_size)
  peast_crop(path+'mask_img.png',path+'gene_img.png',path+'/recover/',img_size)

def value_all(path, img_size):
  mask = value(path+'/real/',path+'/mask/',img_size)
  mask.value('./model/20170512-110547.pb')
  
  gene = value(path+'/real/',path+'/generate/',img_size)
  gene.value('./model/20170512-110547.pb')

  reco =  value(path+'/real/',path+'/recover/',img_size)
  reco.value('./model/20170512-110547.pb')

  real =  value(path+'/real/',path+'/real/',img_size)
  real.value('./model/20170512-110547.pb')

def show(path, score_num):
  score = ['MSE','PSNR','SSIM','FACENET']
  classes = ['mask','generate','recover']
  record = []
  name = []
  for c in classes:
    with open(path+c+'/value_res.txt','r') as f:
      lines = f.readlines()
    for line in lines:
      record.append(float(line.strip().split()[score_num]))
      name.append(c)

  sns.boxplot(x = name,y = record)  
  plt.savefig(path+score[score_num-1]+'.jpg')
  plt.show()

def compare_models(path, model_name, score_num):
  score = ['MSE','PSNR','SSIM','FACENET']
  classes = ['generate', 'recover']
  record = []
  name = []
  for c in classes:
    for i in range(len(path)):
      with open(path[i]+c+'/value_res.txt','r') as f:
        lines = f.readlines()
      for line in lines:
        record.append(float(line.strip().split()[score_num]))
        name.append(model_name[i]+c)


  sns.boxplot(x = name,y = record)  
  plt.savefig('./'+score[score_num-1]+'.jpg')
  plt.show()




if __name__ == '__main__':
  #crop_all('./3.31/',128)
  crop('./5.13/train_output08_12657.png','./5.13/generator/',128)
  crop('./5.13/train_input.png','./5.13/input/',128)
  #crop('./3.31/gene_img.png','./3.31/generate/',128)
  #peast_crop('./3.31/mask_img.png','./3.31/gene_img.png','./3.31/recover/',128)
  #value_all('./3.31',128)
  #show('./3.31/',1)
  #show('./3.31/',2)
  #show('./3.31/',3)
  #show('./3.31/',4)
  #compare('./paper_net/','./3.14/',1)
  #compare('./paper_net/','./3.14/',2)
  #compare('./paper_net/','./3.14/',3)
  #compare('./paper_net/','./3.14/',4)

  








































