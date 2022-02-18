import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import os
import datetime


###########################
device = cv2.ocl.Device_getDefault()
print(f"Vendor ID: {device.vendorID()}")
print(f"Vendor name: {device.vendorName()}")
print(f"Name: {device.name()}")

print("Have OpenCL:")
print(cv2.ocl.haveOpenCL())
cv2.ocl.setUseOpenCL(True)
print("Using OpenCL:")
print(cv2.ocl.useOpenCL())
################################
ISOTIMEFORMAT = '%Y%m%d_%H%M%S'

def HOG(img, bin_n=180):


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.GaussianBlur(gray, (11, 11), 3)
    gray = cv2.normalize(gray, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    gx = cv2.Sobel(gray, -1, 1, 0)
    gy = -cv2.Sobel(gray, -1, 0, 1)

    mag, rad = cv2.cartToPolar(gx, gy)

    mag = cv2.normalize(mag, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    _, mag_filter = cv2.threshold(mag, 0.25, 1, cv2.THRESH_BINARY)
    mag_filter = cv2.erode(mag_filter, kernel, iterations=1)
    mag_filter = cv2.normalize(mag_filter, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)


    ang = np.int32(rad*180/np.pi)

    ang = np.where(ang == 360, 0, ang)
    ang = np.where(ang >= 180, ang-180, ang)

    hist = np.bincount(ang.ravel(), mag_filter.ravel(), bin_n)

    return np.reshape(gy,(gy.shape[0],gy.shape[1],1)) , np.array(hist, np.float32), mag, mag_filter


def VV(filename,scale):
    
    if(filename=="camera"):
        theTime = datetime.datetime.now().strftime(ISOTIMEFORMAT)
        csv_name="./camera_"+str(theTime)+".csv"
        video_name = "./camera_"+str(theTime)+ ".mp4"
        cap = cv2.VideoCapture(CAMERA_NO, cv2.CAP_DSHOW)

    else:
        csv_name="./VV_"+filename[2:]+".csv"
        video_name="./VV_Video_" + filename[2:] + ".mp4"
        cap =cv2.VideoCapture(filename)


    if (cap.isOpened()== False):
      print("Error opening video  file")

    width = int(cap.get(3)) 
    height = int(cap.get(4))

    VV_all=[]

    fig = plt.figure(figsize=((2*width)/(100*scale),(height*0.6/(100*scale))))
    ax = fig.add_subplot(1,1,1)

    x=np.arange(0,180,1)

    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    video_out = cv2.VideoWriter(video_name,
                                fourcc,
                                cap.get(cv2.CAP_PROP_FPS),
                                (int((2*width)/ scale), int((height*2.6)/ scale))
                                )
    VV_m_1=90
    frame_t=0
    while(cap.isOpened()):

      ret, input_img = cap.read()

      if ret == True:

          input_img = cv2.resize(input_img, (int(input_img.shape[1] / scale), int(input_img.shape[0] / scale)))

          hog_x_img, hog_hist_180,hog_mag, hog_mag_filter= HOG(img=input_img)
#
          VV_best_1=np.argsort(hog_hist_180[30:151])[-1]
          VV_best_2=np.argsort(hog_hist_180[30:151])[-2]
          VV_best_3=np.argsort(hog_hist_180[30:151])[-3]

          Sum_VV_samples=hog_hist_180[VV_best_1]+hog_hist_180[VV_best_2]+hog_hist_180[VV_best_3]
          VV=(VV_best_1*(hog_hist_180[VV_best_1]/Sum_VV_samples)
               +VV_best_2*(hog_hist_180[VV_best_2]/Sum_VV_samples)
               +VV_best_3*(hog_hist_180[VV_best_3]/Sum_VV_samples))+30
          VV=(0.7*VV+0.3*VV_m_1)
          VV_all = np.append(VV_all, VV)
          VV_m_1=VV

          VV_rad = np.radians(VV)

          center=[int(input_img.shape[1]/2),int(input_img.shape[0]/2)]
          M=cv2.getRotationMatrix2D(center,90-VV,1)
          Caribreated_img=cv2.warpAffine(input_img,M,(input_img.shape[1],input_img.shape[0]))

          ax.set_ylim(0, 0.2)
          ax.set_xticks([0, 30, 60, 90, 120, 150, 180])
          ax.set_xlim([180,0])

          ax.bar(x, hog_hist_180 / np.sum(hog_hist_180),width=2, align="center")

          ax.vlines(VV,0,0.2,color=(0,1,0))
          ax.vlines(30, 0, 0.2, color="black")
          ax.vlines(150, 0, 0.2, color="black")


          fig.canvas.draw()
          graph=np.array(fig.canvas.renderer.buffer_rgba())[:,:,0:3]

          ax.cla()

          try:
              cv2.putText(input_img, " VV_dig=" + str(int(VV)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 2,
                        cv2.LINE_AA)
          except:
              pass

          cv2.line(input_img, (0, int(input_img.shape[0]/2)), (input_img.shape[1], int(input_img.shape[0]/2)), (0, 0, 0), thickness=2, lineType=cv2.LINE_4)
          cv2.line(input_img, (int(input_img.shape[1]/2), int(input_img.shape[0]/2)), (int(input_img.shape[1]/2), 0), (0, 0, 0), thickness=2, lineType=cv2.LINE_4)

          try:
              cv2.arrowedLine(input_img, (int(input_img.shape[1]/2), int(input_img.shape[0]/2)),
                              (int(input_img.shape[1]/2)+int((input_img.shape[0]/3)*np.cos(VV_rad)),
                               int(input_img.shape[0] / 2)-int((input_img.shape[0]/3)*np.sin(VV_rad))), (0, 255, 0), thickness=4)

              cv2.ellipse(input_img,( int(input_img.shape[1]/2), int(input_img.shape[0]/2)), (int(input_img.shape[0]/6), int(input_img.shape[0]/6)), 0, 0, -VV, (0, 255, 0),thickness=2)

          except:
              pass

          cv2.line(Caribreated_img, (0, int(Caribreated_img.shape[0] / 2)), (Caribreated_img.shape[1], int(Caribreated_img.shape[0] / 2)),
                   (0, 0, 0), thickness=2, lineType=cv2.LINE_4)

          imghstack = np.hstack((input_img, Caribreated_img))
          img2 = np.hstack((hog_mag, hog_mag_filter))

          hoghstack = np.zeros_like(imghstack)
          hoghstack[:, :, 0] = img2.astype("float32")*255
          hoghstack[:, :, 1] = img2.astype("float32")*255
          hoghstack[:, :, 2] = img2.astype("float32")*255

          imgvstack = np.vstack((imghstack, hoghstack))
          imgvstack = np.vstack((imgvstack, graph))

          frame_t=frame_t+1

          cv2.imshow('Video', imgvstack)
          video_out.write(imgvstack)
          c = cv2.waitKey(1)
          if c == 27:
              break
      else:
          break


    VV_all_rad=(VV_all.astype("float32")*np.pi/180)
    VV_x_all = 9.8 * np.cos(VV_all_rad)
    VV_y_all = 9.8 * np.sin(VV_all_rad)



    save_data = pd.DataFrame({'VV_acc_x[m/s^2]':VV_x_all, 'VV_acc_y[m/s^2]':VV_y_all, 'VV_acc_rad':VV_all_rad,'VV_acc_dig':VV_all})
    print(csv_name)
    save_data.to_csv(csv_name)
    cap.release()

    video_out.release()

    cv2.destroyAllWindows()




if __name__ == "__main__":


    ################
    CAMERA=False
    CAMERA_NO=0
    scale=1
    kernel = np.ones((3, 3), np.uint8)
    ##############

    if CAMERA==False:
        scale = 2
        filename="./test.mp4"
        VV(filename,scale)

    elif CAMERA==True:

        filename="camera"
        VV(filename, scale)