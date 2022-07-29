# coding=utf-8
import os
import time
from picamera import PiCamera
import numpy as np
import PIL.Image
import PIL.ImageTk
from tkinter import *
from tkinter.messagebox import askyesno
import cv2
from picamera.array import PiRGBArray
from threading import Thread
from datetime import datetime

#        Images/videos acquisition & treatment software for rasberry pi and picamera            #
#                                      Copyright Alain Paillou 2018-2022                                           #

############################################################################
############################################################################

# Choose your directories for images and videos
image_path = '/home/SkyPi/Images'
video_path= '/home/SkyPi/Videos'

# Initialisation des constantes d'exposition mode rapide
exp_min=1 #ms
exp_max=500 #ms
exp_delta=1 #ms
exp_interval=50

#  Initialisation des paramètres fixés par les boites scalebar
val_resolution = 4
mode_BIN=1
res_x_max = 3200
res_y_max = 2400
res_cam_x = 1024
res_cam_y = 768
cam_displ_x = 1266
cam_displ_y = 950
val_exposition = exp_min #  temps exposition en ms
frame_rate = 1 // (val_exposition / 1000) # division avec résultat entier
if frame_rate > 30 :
    frame_rate = 30
val_sharpen_hard = 0
val_denoise = 2
val_histo_min = 0
val_histo_max = 255
val_contrast_CLAHE = 2
val_phi = 1.0
val_theta = 1.0
text_info1 = "Test information"
val_nb_captures = 1
nb_cap_video =0
val_nb_capt_video = 5000
val_nb_darks = 5
dispo_dark = 'Dark NON dispo'


# Initialisation des filtres soft
flag_2DConv = 0
flag_average = 0
flag_gaussian = 0
flag_bilateral = 0
flag_stop_acquisition = 0
flag_full_res = 0
flag_sharpen_soft1 = 0
flag_unsharp_mask = 0
flag_denoise_soft = 0
flag_histogram_equalize = 0
flag_histogram_stretch = 0
flag_histogram_phitheta = 0
flag_contrast_CLAHE = 0
flag_noir_blanc = 0
flag_acquisition_en_cours = False
flag_autorise_acquisition = False
flag_image_disponible = False
flag_premier_demarrage = True
flag_BIN2 = False
flag_cap_pic = False
flag_sub_dark = False
flag_cap_video = False



camera = PiCamera()
rawCapture=PiRGBArray(camera)
image_affichee = np.empty ((cam_displ_x,cam_displ_y,3), dtype=np.uint8)


def init_camera() :
    global camera
    camera.sensor_mode = 0
    camera.resolution = (res_cam_x, res_cam_y)
    camera.framerate = frame_rate
    camera.drc_strength = 'off'
    camera.image_denoise = False
    camera.image_effect = 'none'
    camera.awb_mode="auto"
    camera.sharpness=0
    camera.vflip = 0
    camera.hflip = 0
    choix_ISO_LOW()
    choix_BIN1()
    mode_acq_rapide()

def quitter() :
    global camera,flag_autorise_acquisition
    flag_autorise_acquisition = False
    time.sleep(1)
    camera.close()
    fenetre_principale.quit()
    
def start_acquisition() :
    global flag_autorise_acquisition,thread_1,flag_stop_acquisition
    flag_autorise_acquisition = True
    flag_stop_acquisition = False
    thread_1 = acquisition("1")
    thread_1.start()
  
def rebin(arr, new_shape):
    shape = (new_shape[0], arr.shape[0] // new_shape[0], new_shape[1], arr.shape[1] // new_shape[1])
    return arr.reshape(shape).mean(-1).mean(1)
             
  
class acquisition(Thread) :
    def __init__(self,lettre) :
        Thread.__init__(self)
        
    def run(self) :
        global camera,traitement, img_cam,cadre_image,rawCapture,image_affichee,image_brut,flag_acquisition_en_cours,flag_autorise_acquisition,flag_image_disponible,frame,flag_capture_off,flag_noir_blanc,flag_BIN2,flag_sub_dark,Master_Dark
        while flag_autorise_acquisition == True :
            if res_cam_x > 1920 :
                if flag_stop_acquisition == False :
                    flag_acquisition_en_cours = True
                    flag_image_disponible = False
                    try :
                        camera.capture(rawCapture, format="bgr")
                        image_brut=rawCapture.array
                        if flag_noir_blanc == 0 :
                            if flag_BIN2 == False :
                                image_brut=cv2.cvtColor(image_brut, cv2.COLOR_BGR2RGB)
                            else :
                                image_brut=cv2.cvtColor(image_brut, cv2.COLOR_BGR2RGB)
                                img_b,img_g,img_r = cv2.split(image_brut)
                                Y,X  = img_b.shape
                                X_BIN = np.int16(Y/2)
                                Y_BIN = np.int16(X/2)
                                image_tampon_b = np.zeros([X_BIN,Y_BIN],dtype='int16')
                                image_tampon_g = np.zeros([X_BIN,Y_BIN],dtype='int16')
                                image_tampon_r = np.zeros([X_BIN,Y_BIN],dtype='int16')
                                image_tampon_b = np.int16(rebin(img_b, (X_BIN,Y_BIN)))
                                image_tampon_g = np.int16(rebin(img_g, (X_BIN,Y_BIN)))
                                image_tampon_r = np.int16(rebin(img_r, (X_BIN,Y_BIN)))
                                image_tampon_b = np.int16(image_tampon_b) * 4
                                image_tampon_g = np.int16(image_tampon_g) * 4
                                image_tampon_r = np.int16(image_tampon_r) * 4
                                image_tampon_b[image_tampon_b > 255] =255
                                image_tampon_g[image_tampon_g > 255] =255
                                image_tampon_r[image_tampon_r > 255] =255
                                img_b=np.uint8(image_tampon_b)
                                img_g=np.uint8(image_tampon_g)
                                img_r=np.uint8(image_tampon_r)
                                image_brut=cv2.merge((img_b,img_g,img_r))
                        else :
                            if flag_BIN2 == False :
                                image_brut=cv2.cvtColor(image_brut, cv2.COLOR_BGR2GRAY)
                            else :
                                image_brut=cv2.cvtColor(image_brut, cv2.COLOR_BGR2GRAY)
                                Y,X  = image_brut.shape
                                X_BIN = np.int16(Y/2)
                                Y_BIN = np.int16(X/2)
                                image_tampon = np.zeros([X_BIN,Y_BIN],dtype='int16')
                                image_tampon = np.int16(rebin(image_brut, (X_BIN,Y_BIN)))
                                image_tampon = np.int16(image_tampon) * 4
                                image_tampon[image_tampon > 255] =255
                                image_brut=np.uint8(image_tampon)
                        rawCapture.truncate(0)
                        if flag_sub_dark == True :
                            image_brut = image_brut - Master_Dark
                        flag_image_disponible = True
                        flag_acquisition_en_cours = False
                    except ValueError :
                        time.sleep(0.05)
            else :
                if flag_stop_acquisition == False :
                    flag_acquisition_en_cours = True
                    flag_image_disponible = False
                    try :
                        for frame in camera.capture_continuous(rawCapture, format='bgr', use_video_port=True) :
                            image_brut_tmp=frame.array
                            if flag_noir_blanc == 0 :
                                if flag_BIN2 == False :
                                    image_brut=cv2.cvtColor(image_brut_tmp, cv2.COLOR_BGR2RGB)
                                else :
                                    image_brut=cv2.cvtColor(image_brut_tmp, cv2.COLOR_BGR2RGB)
                                    img_b,img_g,img_r = cv2.split(image_brut)
                                    Y,X  = img_b.shape
                                    X_BIN = np.int16(Y/2)
                                    Y_BIN = np.int16(X/2)
                                    image_tampon_b = np.zeros([X_BIN,Y_BIN],dtype='int16')
                                    image_tampon_g = np.zeros([X_BIN,Y_BIN],dtype='int16')
                                    image_tampon_r = np.zeros([X_BIN,Y_BIN],dtype='int16')
                                    image_tampon_b = np.int16(rebin(img_b, (X_BIN,Y_BIN)))
                                    image_tampon_g = np.int16(rebin(img_g, (X_BIN,Y_BIN)))
                                    image_tampon_r = np.int16(rebin(img_r, (X_BIN,Y_BIN)))
                                    image_tampon_b = np.int16(image_tampon_b) * 4
                                    image_tampon_g = np.int16(image_tampon_g) * 4
                                    image_tampon_r = np.int16(image_tampon_r) * 4
                                    image_tampon_b[image_tampon_b > 255] =255
                                    image_tampon_g[image_tampon_g > 255] =255
                                    image_tampon_r[image_tampon_r > 255] =255
                                    img_b=np.uint8(image_tampon_b)
                                    img_g=np.uint8(image_tampon_g)
                                    img_r=np.uint8(image_tampon_r)
                                    image_brut=cv2.merge((img_b,img_g,img_r))
                            else :
                                if flag_BIN2 == False :
                                    image_brut=cv2.cvtColor(image_brut_tmp, cv2.COLOR_BGR2GRAY)
                                else :
                                    image_brut_tmp=cv2.cvtColor(image_brut_tmp, cv2.COLOR_BGR2GRAY)
                                    Y,X  = image_brut_tmp.shape
                                    X_BIN = np.int16(Y/2)
                                    Y_BIN = np.int16(X/2)
                                    image_tampon = np.zeros([X_BIN,Y_BIN],dtype='int16')
                                    image_tampon = np.int16(rebin(image_brut_tmp, (X_BIN,Y_BIN)))
                                    image_tampon = np.int16(image_tampon) * 4
                                    image_tampon[image_tampon > 255] =255
                                    image_brut=np.uint8(image_tampon)
                            rawCapture.truncate(0)
                            if flag_sub_dark == True :
                                image_brut = image_brut - Master_Dark
                            flag_image_disponible = True
                            flag_acquisition_en_cours = False
                            break
                    except ValueError :
                        time.sleep(0.05)
            flag_acquisition_en_cours = False
            time.sleep(0.02)

    def stop(self) :
        global flag_start_acquisition
        flag_start_acquisition = False
        
    def reprise(self) :
        global flag_start_acquisition
        flag_start_acquisition = True
        
                
    
def refresh() :
    global camera,traitement, img_cam,cadre_image,rawCapture,image_affichee,image_brut,flag_image_disponible,thread_1,flag_acquisition_en_cours,flag_autorise_acquisition,flag_premier_demarrage,flag_BIN2
    if flag_premier_demarrage == True :
        flag_premier_demarrage = False
        start_acquisition()
    if flag_image_disponible == True :
        if flag_stop_acquisition == False :
            application_filtrage()
            img_cam=PIL.Image.fromarray(image_brut)
            if res_cam_x > cam_displ_x and flag_full_res == 0 and flag_BIN2 == False :
                cadre_image.im=img_cam.resize((cam_displ_x,cam_displ_y), PIL.Image.NEAREST)
            else :
                if res_cam_x/2 > cam_displ_x and flag_full_res == 0 and flag_BIN2 == True :
                    cadre_image.im=img_cam.resize((cam_displ_x,cam_displ_y), PIL.Image.NEAREST)
                else :
                    cadre_image.im = img_cam
            cadre_image.photo=PIL.ImageTk.PhotoImage(cadre_image.im)
            cadre_image.create_image(cam_displ_x/2,cam_displ_y/2, image=cadre_image.photo)
            rawCapture.truncate(0)
        else :
            application_filtrage()
            img_cam=PIL.Image.fromarray(img_brut_tmp)
            if res_cam_x > cam_displ_x and flag_full_res == 0 :
                cadre_image.im=img_cam.resize((cam_displ_x,cam_displ_y), PIL.Image.NEAREST)
            else :
                cadre_image.im = img_cam
            cadre_image.photo=PIL.ImageTk.PhotoImage(cadre_image.im)
            cadre_image.create_image(cam_displ_x/2,cam_displ_y/2, image=cadre_image.photo)
            rawCapture.truncate(0)
    fenetre_principale.after(5, refresh)


def application_filtrage() :
    global image_brut, img_brut_tmp,val_denoise,val_histo_min,val_histo_max,flag_cap_pic,flag_traitement,val_contrast_CLAHE,flag_histogram_phitheta
    if flag_stop_acquisition == False : 
        if flag_2DConv == 1 :
            kern_2DConv=np.ones((5,5), np.float32)/25
            image_brut=cv2.filter2D(image_brut,-1,kern_2DConv) # Application filtre 2D convolution
        if flag_average == 1 :
            image_brut=cv2.blur(image_brut,(5,5)) # Application filtre average
        if flag_gaussian == 1 :
            image_brut=cv2.GaussianBlur(image_brut,(5,5),0) # Application filtre gaussien
        if flag_bilateral == 1 :
            image_brut=cv2.bilateralFilter(image_brut,9,75,75) # Application filtre bilateral
        if flag_denoise_soft == 1 :
            param=float(val_denoise)
            if flag_noir_blanc == 0 :
                image_brut=cv2.fastNlMeansDenoisingColored(image_brut, None, param, param, 3, 5) # application filtre denoise software colour
            else :
                image_brut=cv2.fastNlMeansDenoising(image_brut, None, param, 3, 5) # application filtre denoise software N&B
                time.sleep(0.01)
        if flag_sharpen_soft1 == 1 :
            kern_sharp_soft1 = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],dtype=np.int8) # application filtre logiciel sharpen 1
            image_brut=cv2.filter2D(image_brut,-1,kern_sharp_soft1)
        if flag_unsharp_mask == 1 :
            gaussian_3 = cv2.GaussianBlur(image_brut, (9,9),10,0)
            #img_tempo1 = image_brut
            image_brut = cv2.addWeighted(image_brut,1.5,gaussian_3,-0.5,0, image_brut)
        if flag_histogram_equalize ==1 :
            if flag_noir_blanc == 0 :
                img_b,img_g,img_r = cv2.split(image_brut)
                img_b=cv2.equalizeHist(img_b)
                img_g=cv2.equalizeHist(img_g)
                img_r=cv2.equalizeHist(img_r)
                image_brut=cv2.merge((img_b,img_g,img_r))
            else :
                image_brut=cv2.equalizeHist(image_brut)
        if flag_histogram_stretch == 1 :
            im_tempo = np.int16(image_brut)
            im_tempo=(im_tempo-val_histo_min)*(255/(val_histo_max-val_histo_min))
            im_tempo[im_tempo > 255] = 255
            image_brut = np.uint8(im_tempo)
        if flag_histogram_phitheta == 1 :
            im_tempo = np.int16(image_brut)
            im_tempo=(255/val_phi)*(im_tempo/(255/val_theta))**0.5
            im_tempo[im_tempo > 255] = 255
            image_brut = np.uint8(im_tempo)
        if flag_contrast_CLAHE ==1 :
            if flag_noir_blanc == 0 :
                img_b,img_g,img_r = cv2.split(image_brut)
                clahe = cv2.createCLAHE(clipLimit=val_contrast_CLAHE, tileGridSize=(8,8))
                img_b=clahe.apply(img_b)
                img_g=clahe.apply(img_g)
                img_r=clahe.apply(img_r)
                image_brut=cv2.merge((img_b,img_g,img_r))
            else :
                clahe = cv2.createCLAHE(clipLimit=val_contrast_CLAHE, tileGridSize=(8,8))
                image_brut = clahe.apply(image_brut)
        if flag_cap_pic == True:
            pic_capture()
        if flag_cap_video == True :
            video_capture()
    else :        
        img_brut_tmp=image_brut
        if flag_2DConv == 1 :
            kern_2DConv=np.ones((5,5), np.float32)/25
            img_brut_tmp=cv2.filter2D(img_brut_tmp,-1,kern_2DConv) # Application filtre 2D convolution
        if flag_average == 1 :
            img_brut_tmp=cv2.blur(img_brut_tmp,(5,5)) # Application filtre average
        if flag_gaussian == 1 :
            img_brut_tmp=cv2.GaussianBlur(img_brut_tmp,(5,5),0) # Application filtre gaussien
        if flag_bilateral == 1 :
            img_brut_tmp=cv2.bilateralFilter(img_brut_tmp,9,75,75) # Application filtre bilateral
        if flag_denoise_soft == 1 :
            param=float(val_denoise)
            if flag_noir_blanc == 0 :
                img_brut_tmp=cv2.fastNlMeansDenoisingColored(img_brut_tmp, None, param, 3, 5) # application filtre denoise software colour
            else :
                img_brut_tmp=cv2.fastNlMeansDenoising(img_brut_tmp, None, param, 3, 5) # application filtre denoise software N&B
                time.sleep(0.01)
        if flag_sharpen_soft1 == 1 :
            kern_sharp_soft1 = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]],dtype=np.int8) # application filtre logiciel sharpen 1
            img_brut_tmp=cv2.filter2D(img_brut_tmp,-1,kern_sharp_soft1)
        #if flag_unsharp_mask == 1 :
        #    gaussian_3 = cv2.GaussianBlur(img_brut_tmp, (9,9),10,0)
        #    img_tempo1 = cv2.addWeighted(img_brut_tmp,1.5,gaussian_3,-0.5,0, img_brut_tmp)
        #    img_brut_tmp = img_tempo1
        if flag_histogram_equalize ==1 :
            if flag_noir_blanc == 0 :
                img_b,img_g,img_r = cv2.split(img_brut_tmp)
                img_b=cv2.equalizeHist(img_b)
                img_g=cv2.equalizeHist(img_g)
                img_r=cv2.equalizeHist(img_r)
                img_brut_tmp=cv2.merge((img_b,img_g,img_r))
            else :
                img_brut_tmp=cv2.equalizeHist(img_brut_tmp)    
        if flag_histogram_stretch == 1 :
            im_tempo = np.int16(img_brut_tmp)
            im_tempo=(im_tempo-val_histo_min)*(255/(val_histo_max-val_histo_min))
            im_tempo[im_tempo > 255] = 255
            img_brut_tmp = np.uint8(im_tempo)
        if flag_histogram_phitheta == 1 :
            im_tempo = np.int16(img_brut_tmp)
            im_tempo=(255/val_phi)*(im_tempo/(255/val_theta))**0.5
            im_tempo[im_tempo > 255] = 255
            img_brut_tmp = np.uint8(im_tempo)
        if flag_contrast_CLAHE ==1 :
            if flag_noir_blanc == 0 :
                img_b,img_g,img_r = cv2.split(img_brut_tmp)
                clahe = cv2.createCLAHE(clipLimit=val_contrast_CLAHE, tileGridSize=(8,8))
                img_b=clahe.apply(img_b)
                img_g=clahe.apply(img_g)
                img_r=clahe.apply(img_r)
                img_brut_tmp=cv2.merge((img_b,img_g,img_r))
            else :
                clahe = cv2.createCLAHE(clipLimit=val_contrast_CLAHE, tileGridSize=(8,8))
                img_brut_tmp = clahe.apply(img_brut_tmp)
        if flag_cap_pic == True:
            pic_capture()
            
                        
def pic_capture() :
    global start,nb_pic_cap,nb_acq_pic,labelInfo1,flag_cap_pic,nb_cap_pic,image_path
    if nb_cap_pic <= val_nb_captures :
        nom_fichier = start.strftime('PIC%Y%m%d_%H%M%S_%f_') + "%03d" % nb_cap_pic + '.jpg'
        if image_brut.ndim == 3 :
            cv2.imwrite(os.path.join(image_path,nom_fichier), cv2.cvtColor(image_brut, cv2.COLOR_BGR2RGB), [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        else :
             cv2.imwrite(nom_fichier, image_brut, [int(cv2.IMWRITE_JPEG_QUALITY), 90])   
        labelInfo1 = Label (cadre, text = "capture n° "+ nom_fichier)
        labelInfo1.place(anchor="w", x=xLI1,y=yLI1)
        nb_cap_pic += 1
    else :
        flag_cap_pic = False
        labelInfo1 = Label (cadre, text = "                                                                                                        ") 
        labelInfo1.place(anchor="w", x=xLI1,y=yLI1)
        labelInfo1 = Label (cadre, text = " Capture pictures terminee")
        labelInfo1.place(anchor="w", x=xLI1,y=yLI1)
    
def start_pic_capture() :
    global nb_acq_pic,flag_cap_pic,nb_cap_pic,start
    flag_cap_pic = True
    nb_cap_pic =1
    start = datetime.now()
    
 
def stop_pic_capture() :
    global nb_cap_pic
    nb_cap_pic = val_nb_capt_video +1

def video_capture() :
    global image_brut,start_video,nb_cap_video,nb_acq_video,labelInfo1,flag_cap_video,video_path,val_nb_capt_video,video,echelle11
    if nb_cap_video == 1 :
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        nom_video = start_video.strftime('VID%Y%m%d_%H%M%S') + '.avi'
        if image_brut.ndim == 3 :
            height,width,layers = image_brut.shape
            video = cv2.VideoWriter(os.path.join(video_path,nom_video), fourcc, 25, (width, height), isColor = True)
        else :
            height,width = image_brut.shape
            video = cv2.VideoWriter(os.path.join(video_path,nom_video), fourcc, 25, (width, height), isColor = False)
        labelInfo1 = Label (cadre, text = "                                                                                                        ") 
        labelInfo1.place(anchor="w", x=xLI1,y=yLI1)
        labelInfo1 = Label (cadre, text = " Acquisition vidéo en cours")
        labelInfo1.place(anchor="w", x=xLI1,y=yLI1)
    if nb_cap_video <= val_nb_capt_video :
        video.write(image_brut)
        if nb_cap_video % 10 == 0 :
            labelInfo1 = Label (cadre, text = " frame : " + str (nb_cap_video) + "                            ")
            labelInfo1.place(anchor="w", x=xLI1,y=yLI1)
        nb_cap_video += 1
    else :
        video.release()
        flag_cap_video = False
        labelInfo1 = Label (cadre, text = " Acquisition vidéo terminee     ")
        labelInfo1.place(anchor="w", x=xLI1,y=yLI1)
        
def start_video_capture() :
    global nb_cap_video,flag_cap_video,start_video,val_nb_capt_video
    flag_cap_video = True
    nb_cap_video =1
    if val_nb_capt_video == 0 :
        val_nb_capt_video = 10000
    start_video = datetime.now()
    
 
def stop_video_capture() :
    global nb_cap_video,val_nb_capt_video
    nb_cap_video = val_nb_capt_video +1
    

# Fonctions récupération des paramètres grace aux scalebars
def mode_acq_rapide() :
    global camera,exp_min, exp_max, exp_delta,exp_interval,echelle1,val_exposition,frame_rate,flag_acquisition_en_cours,flag_stop_acquisition,flag_sub_dark,dispo_dark,labelInfoDark
    flag_stop_acquisition=True
    flag_sub_dark = False
    dispo_dark = 'Dark NON dispo '
    labelInfoDark = Label (cadre, text = dispo_dark)
    labelInfoDark.place(anchor="w", x=xdark+90,y=ydark)
    choix_sub_dark.set(0)
    while flag_acquisition_en_cours == True : # on attend que les acquisitions soient interrompues
        time.sleep(0.1)
    exp_min=1 #ms
    exp_max=400 #ms
    exp_delta=1 #ms
    exp_interval=50
    val_exposition=exp_min
    echelle1 = Scale (cadre, from_ = exp_min, to = exp_max , command= valeur_exposition, orient=HORIZONTAL, length = 400, width = 10, resolution = exp_delta, label="",showvalue=1,tickinterval=exp_interval,sliderlength=20)
    echelle1.set (val_exposition)
    echelle1.place(anchor="w", x=xS1,y=yS1)
    frame_rate = 1 / (val_exposition / 1000)
    if frame_rate > 30 :
        frame_rate = 30
    camera.framerate = frame_rate
    camera.framerate_delta = 5
    camera.shutter_speed= val_exposition*1000
    time.sleep(0.1)
    flag_stop_acquisition=False
    
def mode_acq_lente() :
    global camera,exp_min, exp_max, exp_delta,exp_interval,echelle1,val_exposition,frame_rate,flag_acquisition_en_cours,flag_stop_acquisition,flag_sub_dark,dispo_dark,labelInfoDark
    flag_stop_acquisition=True
    flag_sub_dark = False
    dispo_dark = 'Dark NON dispo '
    labelInfoDark = Label (cadre, text = dispo_dark)
    labelInfoDark.place(anchor="w", x=xdark+90,y=ydark)
    choix_sub_dark.set(0)
    while flag_acquisition_en_cours == True : # on attend que les acquisitions soient interrompues
        time.sleep(0.1)
    exp_min=400 #ms
    exp_max=8000 #ms
    exp_delta=250 #ms
    exp_interval=2000
    val_exposition=exp_min
    echelle1 = Scale (cadre, from_ = exp_min, to = exp_max , command= valeur_exposition, orient=HORIZONTAL, length = 400, width = 10, resolution = exp_delta, label="",showvalue=1,tickinterval=exp_interval,sliderlength=20)
    echelle1.set (val_exposition)
    echelle1.place(anchor="w", x=xS1,y=yS1)
    frame_rate = 1 / (val_exposition / 1000)
    if frame_rate > 30 :
        frame_rate = 30
    if frame_rate < 0.1 :
        frame_rate=0.1
    print("framrate : ",frame_rate)
    camera.framerate = frame_rate
    camera.framerate_delta = 1
    camera.shutter_speed= val_exposition*1000
    time.sleep(0.1)
    flag_stop_acquisition=False

def valeur_exposition (event=None) :
    global camera,val_exposition, frame_rate,camera,echelle1,val_resolution,flag_stop_acquisition,flag_sub_dark,dispo_dark,labelInfoDark
    flag_stop_acquisition=True
    flag_sub_dark = False
    dispo_dark = 'Dark NON dispo '
    labelInfoDark = Label (cadre, text = dispo_dark)
    labelInfoDark.place(anchor="w", x=xdark+90,y=ydark)
    choix_sub_dark.set(0)
    #while flag_acquisition_en_cours == True : # on attend que les acquisitions soient interrompues
    #    time.sleep(0.1)
    val_exposition = echelle1.get()
    frame_rate = 1 / (val_exposition / 1000)
    if frame_rate > 30 :
        frame_rate = 30
    if frame_rate < 0.1 :
        frame_rate=0.1
    camera.framerate = frame_rate
    print("framerate : ",frame_rate)
    camera.shutter_speed = val_exposition*1000
    time.sleep(0.05)
    flag_stop_acquisition=False

def choix_ISO_LOW() :
    global camera,flag_acquisition_en_cours,flag_stop_acquisition,flag_sub_dark,dispo_dark,labelInfoDark
    flag_stop_acquisition=True
    flag_sub_dark = False
    dispo_dark = 'Dark NON dispo '
    labelInfoDark = Label (cadre, text = dispo_dark)
    labelInfoDark.place(anchor="w", x=xdark+90,y=ydark)
    choix_sub_dark.set(0)
    while flag_acquisition_en_cours == True : # on attend que les acquisitions soient interrompues
        time.sleep(0.1)
    camera.exposure_mode = 'sports'
    camera.ISO = 100
    time.sleep(0.5)
    camera.exposure_mode = 'off'
    flag_stop_acquisition=False

def choix_ISO_MEDIUM() :
    global camera,flag_acquisition_en_cours,flag_stop_acquisition,flag_sub_dark,dispo_dark,labelInfoDark
    flag_stop_acquisition=True
    flag_sub_dark = False
    dispo_dark = 'Dark NON dispo '
    labelInfoDark = Label (cadre, text = dispo_dark)
    labelInfoDark.place(anchor="w", x=xdark+90,y=ydark)
    choix_sub_dark.set(0)
    while flag_acquisition_en_cours == True : # on attend que les acquisitions soient interrompues
        time.sleep(0.1)
    camera.exposure_mode = 'sports'
    camera.ISO = 400
    time.sleep(2)
    camera.exposure_mode = 'off'
    flag_stop_acquisition=False

def choix_ISO_HIGH() :
    global camera,flag_acquisition_en_cours,flag_stop_acquisition,flag_sub_dark,dispo_dark,labelInfoDark
    flag_stop_acquisition=True
    flag_sub_dark = False
    dispo_dark = 'Dark NON dispo '
    labelInfoDark = Label (cadre, text = dispo_dark)
    labelInfoDark.place(anchor="w", x=xdark+90,y=ydark)
    choix_sub_dark.set(0)
    while flag_acquisition_en_cours == True : # on attend que les acquisitions soient interrompues
        time.sleep(0.1)
    camera.exposure_mode = 'sports'
    camera.ISO = 800
    time.sleep(4)
    camera.exposure_mode = 'off'
    flag_stop_acquisition=False

def choix_BIN1(event=None) :
    global camera,val_resolution,echelle3,flag_acquisition_en_cours,flag_stop_acquisition,flag_BIN2,flag_sub_dark,dispo_dark,labelInfoDark
    flag_BIN2=False
    flag_stop_acquisition=True
    flag_sub_dark = False
    dispo_dark = 'Dark NON dispo '
    labelInfoDark = Label (cadre, text = dispo_dark)
    labelInfoDark.place(anchor="w", x=xdark+90,y=ydark)
    choix_sub_dark.set(0)
    while flag_acquisition_en_cours == True : # on attend que les acquisitions soient interrompues
        time.sleep(0.1)
    val_resolution = echelle3.get()
    camera.sensor_mode = 0
    time.sleep(0.2)
    echelle3 = Scale (cadre, from_ = 1, to = 9, command= choix_resolution_camera, orient=HORIZONTAL, length = 250, width = 10, resolution = 1, label="",showvalue=1,tickinterval=1,sliderlength=20)
    echelle3.set (val_resolution)
    echelle3.place(anchor="w", x=xS3,y=yS3)
    time.sleep(0.2)
    choix_resolution_camera()
    print('BIN1 selectionne')
    flag_stop_acquisition=False

def choix_BIN2(event=None) :
    global camera,echelle3,val_resolution,flag_acquisition_en_cours,flag_stop_acquisition,choix_noir_blanc,flag_noir_blanc,flag_BIN2,flag_sub_dark,dispo_dark,labelInfoDark
    flag_sub_dark = False
    dispo_dark = 'Dark NON dispo '
    labelInfoDark = Label (cadre, text = dispo_dark)
    labelInfoDark.place(anchor="w", x=xdark+90,y=ydark)
    choix_sub_dark.set(0) 
    #choix_noir_blanc.set(1)
    #flag_noir_blanc = 1
    flag_BIN2 = True
    print('BIN2 selectionne')

def choix_resolution_camera(event=None) :
    global camera,traitement,val_resolution,res_cam_x,res_cam_y, img_cam,rawCapture,res_x_max,res_y_max,echelle3,flag_image_disponible,flag_acquisition_en_cours,flag_stop_acquisition,flag_sub_dark,dispo_dark,labelInfoDark
    flag_stop_acquisition=True
    flag_sub_dark = False
    dispo_dark = 'Dark NON dispo '
    labelInfoDark = Label (cadre, text = dispo_dark)
    labelInfoDark.place(anchor="w", x=xdark+90,y=ydark)
    choix_sub_dark.set(0)
    while flag_acquisition_en_cours == True : # on attend que les acquisitions soient interrompues
        time.sleep(0.1)
    val_resolution = echelle3.get()
    print (val_resolution)
    if val_resolution == 1 :
        res_cam_x,res_cam_y = 320,240
    elif val_resolution == 2 :
        res_cam_x, res_cam_y = 640, 480
    elif val_resolution == 3 :
        res_cam_x, res_cam_y = 800,608
    elif val_resolution == 4 :
        res_cam_x, res_cam_y = 1024, 768
    elif val_resolution == 5 :
        res_cam_x, res_cam_y = 1280, 960
    elif val_resolution == 6 :
        res_cam_x, res_cam_y = 1600,1200
    elif val_resolution == 7 :
        res_cam_x, res_cam_y = 1920,1440
    elif val_resolution == 8 :
        res_cam_x, res_cam_y = 2560,1920
    else :
        res_cam_x, res_cam_y = 3200, 2400
    #time.sleep(0.1)
    camera.resolution = (res_cam_x, res_cam_y)
    rawCapture=PiRGBArray(camera)
    #time.sleep(0.1)
    print("resolution camera = ",res_cam_x," ",res_cam_y)
    #valeur_exposition()
    flag_stop_acquisition=False
    time.sleep(0.1)

def choix_valeur_denoise(event=None) :
    global val_denoise
    val_denoise=echelle4.get()
    if val_denoise == 0 :
        val_denoise += 1       

def commande_flipV() :
    global camera
    if choix_flipV.get() == 0 :
        camera.vflip = 0
    else :
        camera.vflip = 1

def commande_flipH() :
    global camera
    if choix_flipH.get() == 0 :
        camera.hflip = 0
    else :
        camera.hflip = 1

def commande_img_Neg() :
    global camera
    if choix_img_Neg.get() == 0 :
        camera.image_effect = 'none'
    else :
        camera.image_effect = 'negative'

def commande_img_Denoise_Hard() :
    global camera
    if choix_img_Denoise_Hard.get() == 0 :
        camera.image_denoise = False
    else :
        camera.image_denoise = True
        
def commande_2DConvol() :
    global flag_2DConv
    if choix_2DConv.get() == 0 :
        flag_2DConv = 0
    else :
        flag_2DConv = 1
        
def commande_average() :
    global flag_average
    if choix_average.get() == 0 :
        flag_average = 0
    else :
        flag_average = 1
        
def commande_gaussian() :
    global flag_gaussian
    if choix_gaussian.get() == 0 :
        flag_gaussian = 0
    else :
        flag_gaussian = 1
        
def commande_bilateral() :
    global flag_bilateral
    if choix_bilateral.get() == 0 :
        flag_bilateral = 0
    else :
        flag_bilateral = 1

def commande_stop_acquisition() :
    global flag_stop_acquisition
    if choix_stop_acquisition.get() == 0 :
        flag_stop_acquisition = 0
    else :
        flag_stop_acquisition = 1
        
def commande_mode_full_res() :
    global flag_full_res
    if choix_mode_full_res.get() == 0 :
        flag_full_res = 0
    else :
        flag_full_res = 1
        
def choix_sharpen_hard(event=None) :
    global camera,val_sharpen_hard,echelle2
    val_sharpen_hard=echelle2.get()
    camera.sharpness=val_sharpen_hard

def choix_valeur_CLAHE(event=None) :
    global val_contrast_CLAHE,echelle9
    val_contrast_CLAHE=echelle9.get()
    print(val_contrast_CLAHE)

def commande_sharpen_soft1() :
    global flag_sharpen_soft1
    if choix_sharpen_soft1.get() == 0 :
        flag_sharpen_soft1 = 0
    else :
        flag_sharpen_soft1 = 1
        
def commande_unsharp_mask() :
    global flag_unsharp_mask
    if choix_unsharp_mask.get() == 0 :
        flag_unsharp_mask = 0
    else :
        flag_unsharp_mask = 1

def commande_denoise_soft() :
    global flag_denoise_soft
    if choix_denoise_soft.get() == 0 :
        flag_denoise_soft = 0
    else :
        flag_denoise_soft = 1

def commande_histogram_equalize() :
    global flag_histogram_equalize
    if choix_histogram_equalize.get() == 0 :
        flag_histogram_equalize = 0
    else :
        flag_histogram_equalize = 1

def choix_histo_min(event=None) :
    global camera,val_histo_min,echelle5
    val_histo_min=echelle5.get()
    
def choix_phi(event=None) :
    global val_phi,echelle12
    val_phi=echelle12.get()

def choix_theta(event=None) :
    global val_theta,echelle13
    val_theta=echelle13.get()
    
def choix_histo_max(event=None) :
    global camera,val_histo_max,echelle6
    val_histo_max=echelle6.get()
    #print("Histo Maxi : ",val_histo_max)
    
    
def commande_histogram_stretch() :
    global flag_histogram_stretch
    if choix_histogram_stretch.get() == 0 :
        flag_histogram_stretch = 0
    else :
        flag_histogram_stretch = 1
    
def commande_histogram_phitheta() :
    global flag_histogram_phitheta
    if choix_histogram_phitheta.get() == 0 :
        flag_histogram_phitheta = 0
    else :
        flag_histogram_phitheta = 1
      
def commande_contrast_CLAHE() :
    global flag_contrast_CLAHE
    if choix_contrast_CLAHE.get() == 0 :
        flag_contrast_CLAHE= 0
    else :
        flag_contrast_CLAHE = 1
      
def commande_noir_blanc() :
    global flag_noir_blanc,flag_sub_dark,dispo_dark,labelInfoDark
    if choix_noir_blanc.get() == 0 :
        flag_noir_blanc = 0
        print("forcage N&B OFF")
    else :
        flag_noir_blanc = 1
        print("forcage N&B ON")
    flag_sub_dark = False
    dispo_dark = 'Dark NON dispo '
    labelInfoDark = Label (cadre, text = dispo_dark)
    labelInfoDark.place(anchor="w", x=xdark+90,y=ydark)
    choix_sub_dark.set(0)

def choix_nb_captures(event=None) :
    global val_nb_captures
    val_nb_captures=echelle8.get()

def choix_nb_video(event=None) :
    global val_nb_capt_video
    val_nb_capt_video=echelle11.get()

def commande_sub_dark() :
    global flag_sub_dark,dispo_dark
    if choix_sub_dark.get() == 0 :
        flag_sub_dark = False
        dispo_dark = 'Dark NON dispo '
        print(dispo_dark)
    else :
        flag_sub_dark = True
        dispo_dark = 'Dark Disponible '
        print(dispo_dark)

def start_cap_dark() :
    global flag_sub_dark,dispo_dark,labelInfoDark,flag_stop_acquisition,flag_acquisition_en_cours,labelInfo1,val_nb_darks,text_info1,xLI1,yLI1,Master_Dark
    if askyesno("Cover the Lens", "Dark acquisition continue ?") :
        print("on continue")
        flag_stop_acquisition=True
        text_info1 = "Initialisation capture DARK"
        labelInfo1 = Label (cadre, text = text_info1)
        labelInfo1.place(anchor="w", x=xLI1,y=yLI1)
        while flag_acquisition_en_cours == True : # on attend que les acquisitions soient interrompues
            time.sleep(0.1)
        try :
            num_dark = 1
            while num_dark <= val_nb_darks :
                text_info1 = "Capture Dark n°" +  "%02d" % num_dark
                labelInfo1 = Label (cadre, text = text_info1)
                labelInfo1.place(anchor="w", x=xLI1,y=yLI1)
                print(text_info1)
                time.sleep(1)
                camera.capture(rawCapture, format="bgr")
                if num_dark == 1 :
                    dark_tempo=np.int16(rawCapture.array)
                else :
                    dark_tempo= dark_tempo + rawCapture.array
                num_dark += 1
                rawCapture.truncate(0)
            text_info1 = "Calcul Master Dark"
            labelInfo1 = Label (cadre, text = text_info1)
            labelInfo1.place(anchor="w", x=xLI1,y=yLI1)
            print(text_info1)
            time.sleep(1)
            dark_tempo = dark_tempo // val_nb_darks
            dark_tempo[dark_tempo > 255] = 255
            dark_tempo[dark_tempo < 0] = 0
            Mean_dark = np.uint8(dark_tempo)
            print("Mean dark ok")
            if flag_noir_blanc == 0 :
                Master_Dark=cv2.cvtColor(Mean_dark, cv2.COLOR_BGR2RGB)
                print('master dark ok BIN 1 colour')
            else :
                if flag_BIN2 == False :
                    Master_Dark=cv2.cvtColor(Mean_dark, cv2.COLOR_BGR2GRAY)
                    print('master dark ok BIN 1 mono')
                else :
                    image_brut=cv2.cvtColor(Mean_dark, cv2.COLOR_BGR2GRAY)
                    Y,X  = image_brut.shape
                    X_BIN = np.int16(Y/2)
                    Y_BIN = np.int16(X/2)
                    image_tampon = np.zeros([X_BIN,Y_BIN],dtype='int16')
                    image_tampon = np.int16(rebin(image_brut, (X_BIN,Y_BIN)))
                    image_tampon = np.int16(image_tampon) * 4
                    image_tampon[image_tampon > 255] =255
                    Master_Dark=np.uint8(image_tampon)
                    print('master dark ok BIN 2 mono')
        except ValueError :
            print("erreur creation Dark")
            time.sleep(0.05)
        #flag_sub_dark = True
        dispo_dark = 'Dark disponible '
        labelInfoDark = Label (cadre, text = dispo_dark)
        labelInfoDark.place(anchor="w", x=xdark+90,y=ydark)
        flag_stop_acquisition=False
    else :
        print("on arrete")
        flag_sub_dark = False
        dispo_dark = 'Dark NON dispo '
        labelInfoDark = Label (cadre, text = dispo_dark)
        labelInfoDark.place(anchor="w", x=xdark+90,y=ydark)
    
def choix_nb_darks(event=None) :
    global val_nb_darks, echelle10
    val_nb_darks=echelle10.get()
    

# définition fenetre principale
fenetre_principale = Tk ()
w, h = fenetre_principale.winfo_screenwidth(), fenetre_principale.winfo_screenheight()-20
fenetre_principale.geometry("%dx%d+0+0" % (w, h))
fenetre_principale.protocol("WM_DELETE_WINDOW", quitter)
fenetre_principale.title("Pi Camera")

# Création cadre général
cadre = Frame (fenetre_principale, width = 1800 , heigh = 950)
cadre.pack ()

mode_acq = IntVar()
mode_acq.set(1) # Initialisation du mode d'acquisition a Rapide

choix_ISO = IntVar ()
choix_ISO.set(1) # Initialisation ISO LOW sur choix LOW MEDIUM HIGH

choix_bin = IntVar ()
choix_bin.set(1) # Initialisation BIN 1 sur choix 1, 2 ou 3


choix_flipV = IntVar ()
choix_flipV.set(0) # Initialisation Flip V inactif

choix_flipH = IntVar ()
choix_flipH.set(0) # Initialisation Flip H inactif

choix_img_Neg = IntVar ()
choix_img_Neg.set(0) # Initialisation image en négatif inactif

choix_img_Denoise_Hard = IntVar ()
choix_img_Denoise_Hard.set(0) # Initialisation denoise image hard inactif

choix_2DConv = IntVar()
choix_2DConv.set(0) # intialisation filtre 2D convolution inactif

choix_average = IntVar()
choix_average.set(0) # initialisation filtre average inactif

choix_gaussian = IntVar()
choix_gaussian.set(0) # initialisation filtre gaussien inactif

choix_bilateral = IntVar()
choix_bilateral.set(0) # Initialisation filtre Median inactif

choix_stop_acquisition = IntVar()
choix_stop_acquisition.set(0) # Initialisation stop acquisition inactif

choix_mode_full_res = IntVar()
choix_mode_full_res.set(0) # Initialisation mode full resolution inactif

choix_sharpen_soft1 = IntVar()
choix_sharpen_soft1.set(0) # initialisation mode sharpen software 1 inactif

choix_unsharp_mask = IntVar()
choix_unsharp_mask.set(0) # initialisation mode unsharp mask inactif

choix_denoise_soft = IntVar()
choix_denoise_soft.set(0) # initialisation mode denoise software inactif

choix_histogram_equalize = IntVar()
choix_histogram_equalize.set(0) # initialisation mode histogram equalize inactif

choix_histogram_stretch = IntVar()
choix_histogram_stretch.set(0) # initialisation mode histogram stretch inactif

choix_histogram_phitheta = IntVar()
choix_histogram_phitheta.set(0) # initialisation mode histogram Phi Theta inactif

choix_contrast_CLAHE = IntVar()
choix_contrast_CLAHE.set(0) # initialisation mode contraste CLAHE inactif

choix_noir_blanc = IntVar()
choix_noir_blanc.set(0) # initialisation mode hoir et blanc inactif

choix_sub_dark = IntVar()
choix_sub_dark.set(0) # Initialisation sub dark inactif

# initialisation des boites scrolbar, buttonradio et checkbutton

xBRB=1500 # Sélection mode BIN 1, 2 ou 3 bouton radio
yBRB=20

xCBFNB=1680 # Check box force N&B
yCBFNB=20

xCBSA=1400 # Sélection mode stop acquisition
yCBSA=45

xS3=1550 # Choix résolution
yS3=80

xBRS=1480 # Sélection sensibilite bouton radio
yBRS=120

xBRMA=1530 # Mode acquisition bouton radio
yBRMA=150

xS1=1400 # Durée exposition en ms
yS1=185

xCBFV=1650 # Sélection Flip V
yCBFV=220

xCBFH=1650 # Sélection Flip H
yCBFH=220

xCBIN=1430 # Sélection Image Negative
yCBIN=220

xCBIDH=1430 # Sélection Image Denoise Hard
yCBIDH=220

xS2=1550 # Sélection parametrage Image Sharpen Hard
yS2=255

xCB2DC = 1450 # Selection 2D convolution filter
yCB2DC=300

xCBAV=1200 # Selection Average filter
yCBAV=300

xCBGA= 1200 # Sélection filter Gaussian
yCBGA=300

xCBBL=1200 # Selection filtre bilateral
yCBBL=300

xCBSS1=1450 # Selection filtre sharpen software 1
yCBSS1=330

xCBDS=1630 # Selection filtre Denoise software
yCBDS=300

xS4=1710 # Sélection parametrage Image denoise soft
yS4=300

xCBHPT = 1310 # selection histogramme Phi Theta
yCBHPT = 410

xCBHS = 1310 # selection histogramme stretch
yCBHS = 360

xCBCC = 1310 # selection histogramme Contrat CLAHE
yCBCC = 460

xS5=1460 # Sélection parametrages histo mini et maxi
yS5=360

xS8=1430 # Sélection parametrages nb acquisition pictures
yS8=800

xS9=1460 # Sélection parametrages contrast CLAHE
yS9=460

xS10=1430 # Sélection nombre de darks
yS10=740

xS11=1430 # Sélection nombre de video
yS11=870

xdark = 1600 # Sélection sub dark
ydark = 740

xLI1 = 1300 # label info 1
yLI1 = 930

xS12=1460 # Sélection parametrages histo phi theta
yS12=410

# Choix forcage N&B
CBFNB = Checkbutton(cadre,text="Force N&B", variable=choix_noir_blanc,command=commande_noir_blanc,onvalue = 1, offvalue = 0)
CBFNB.place(anchor="w",x=xCBFNB, y=yCBFNB)

#Choix histogramme phi theta
CBHPT = Checkbutton(cadre,text="Histo Phi Theta", variable=choix_histogram_phitheta,command=commande_histogram_phitheta,onvalue = 1, offvalue = 0)
CBHPT.place(anchor="w",x=xCBHPT, y=yCBHPT)

# Choix histogramme stretch
CBHS = Checkbutton(cadre,text="Histogram Stretch", variable=choix_histogram_stretch,command=commande_histogram_stretch,onvalue = 1, offvalue = 0)
CBHS.place(anchor="w",x=xCBHS, y=yCBHS)

# Choix contrast CLAHE
CBCC = Checkbutton(cadre,text="Contrast CLAHE", variable=choix_contrast_CLAHE,command=commande_contrast_CLAHE,onvalue = 1, offvalue = 0)
CBCC.place(anchor="w",x=xCBCC, y=yCBCC)

# Choix filtre 2D convolution
CB2DC = Checkbutton(cadre,text="2D convol", variable=choix_2DConv,command=commande_2DConvol,onvalue = 1, offvalue = 0)
CB2DC.place(anchor="w",x=xCB2DC+90, y=yCB2DC)

# Choix filtre average
CBAV = Checkbutton(cadre,text="Average", variable=choix_average,command=commande_average,onvalue = 1, offvalue = 0)
CBAV.place(anchor="w",x=xCBAV+90, y=yCBAV)

# Choix filtre gaussien
CBGA = Checkbutton(cadre,text="Gaussian", variable=choix_gaussian,command=commande_gaussian,onvalue = 1, offvalue = 0)
CBGA.place(anchor="w",x=xCBGA+170, y=yCBGA)

# Choix filtre Bilateral
CBBL = Checkbutton(cadre,text="Bilateral", variable=choix_bilateral,command=commande_bilateral,onvalue = 1, offvalue = 0)
CBBL.place(anchor="w",x=xCBBL+250, y=yCBBL)


# Choix du mode flip Vertical
CBFV = Checkbutton(cadre,text="Flip V", variable=choix_flipV,command=commande_flipV,onvalue = 1, offvalue = 0)
CBFV.place(anchor="w",x=xCBFV, y=yCBFV)

# Choix du mode flip Horizontal
CBFH = Checkbutton(cadre,text="Flip H", variable=choix_flipH,command=commande_flipH,onvalue = 1, offvalue = 0)
CBFH.place(anchor="w",x=xCBFH+60, y=yCBFH)

# Choix du mode image en négatif
CBIN = Checkbutton(cadre,text="Image Neg", variable=choix_img_Neg,command=commande_img_Neg,onvalue = 1, offvalue = 0)
CBIN.place(anchor="w",x=xCBIN, y=yCBIN)

# Choix du mode Denoise hardware
CBIDH = Checkbutton(cadre,text="Denoise Hard", variable=choix_img_Denoise_Hard,command=commande_img_Denoise_Hard,onvalue = 1, offvalue = 0)
CBIDH.place(anchor="w",x=xCBIDH+100, y=yCBIDH)

# Choix stop acquisition
CBSA = Checkbutton(cadre,text="Pause acquisition", variable=choix_stop_acquisition,command=commande_stop_acquisition,onvalue = 1, offvalue = 0)
CBSA.place(anchor="w",x=xCBSA+260, y=yCBSA)

CBMFR = Checkbutton(cadre,text="Full Res", variable=choix_mode_full_res,command=commande_mode_full_res,onvalue = 1, offvalue = 0)
CBMFR.place(anchor="w",x=xCBSA+180, y=yCBSA)

CBSS1 = Checkbutton(cadre,text="Sharpen", variable=choix_sharpen_soft1,command=commande_sharpen_soft1,onvalue = 1, offvalue = 0)
CBSS1.place(anchor="w",x=xCBSS1+90, y=yCBSS1)

CBUSM = Checkbutton(cadre,text="UnSharp Mask", variable=choix_unsharp_mask,command=commande_unsharp_mask,onvalue = 1, offvalue = 0)
CBUSM.place(anchor="w",x=xCBSS1+180, y=yCBSS1)

CBDS = Checkbutton(cadre,text="Denoise", variable=choix_denoise_soft,command=commande_denoise_soft,onvalue = 1, offvalue = 0)
CBDS.place(anchor="w",x=xCBDS, y=yCBDS)

# Choix du mode d'aquisition
labelMode_Acq = Label (cadre, text = "Mode acquisition")
labelMode_Acq.place (anchor="w",x=xBRMA, y=yBRMA)
RBMA1 = Radiobutton(cadre,text="Rapide", variable=mode_acq,command=mode_acq_rapide,value=1)
RBMA1.place(anchor="w",x=xBRMA+120, y=yBRMA)
RBMA2 = Radiobutton(cadre,text="Lent", variable=mode_acq,command=mode_acq_lente,value=2)
RBMA2.place(anchor="w",x=xBRMA+200, y=yBRMA)

# Choix du mode de sensibilité ISO - LOW=100 MEDIUM=400 HIGH=800
labelSensibilite = Label (cadre, text = "Sensibilité camera")
labelSensibilite.place (anchor="w",x=xBRS, y=yBRS)
RBS1 = Radiobutton(cadre,text="LOW", variable=choix_ISO,command=choix_ISO_LOW,value=1)
RBS1.place(anchor="w",x=xBRS+120, y=yBRS)
RBS2 = Radiobutton(cadre,text="MEDIUM", variable=choix_ISO,command=choix_ISO_MEDIUM,value=2)
RBS2.place(anchor="w",x=xBRS+170, y=yBRS)
RBS3 = Radiobutton(cadre,text="HIGH", variable=choix_ISO,command=choix_ISO_HIGH,value=3)
RBS3.place(anchor="w",x=xBRS+250, y=yBRS)

# Choix du mode BINNING - 1, 2 ou 3
labelBIN = Label (cadre, text = "BINNING : ")
labelBIN.place (anchor="w",x=xBRB, y=yBRB)
RBB1 = Radiobutton(cadre,text="BIN1", variable=choix_bin,command=choix_BIN1,value=1)
RBB1.place(anchor="w",x=xBRB+70, y=yBRB)
RBB2 = Radiobutton(cadre,text="BIN2", variable=choix_bin,command=choix_BIN2,value=2)
RBB2.place(anchor="w",x=xBRB+120, y=yBRB)


labelParam1 = Label (cadre, text = "Exposition ms")
labelParam1.place(anchor="e", x=xS1,y=yS1)
echelle1 = Scale (cadre, from_ = exp_min, to = exp_max , command= valeur_exposition, orient=HORIZONTAL, length = 400, width = 10, resolution = exp_delta, label="",showvalue=1,tickinterval=exp_interval,sliderlength=20)
echelle1.set (val_exposition)
echelle1.place(anchor="w", x=xS1,y=yS1)

labelParam2 = Label (cadre, text = "Sharpen Picam")
labelParam2.place(anchor="e", x=xS2,y=yS2)
echelle2 = Scale (cadre, from_ = -100, to = 100, command= choix_sharpen_hard, orient=HORIZONTAL, length = 250, width = 10, resolution = 25, label="",showvalue=1,tickinterval=25,sliderlength=20)
echelle2.set (val_sharpen_hard)
echelle2.place(anchor="w", x=xS2,y=yS2)

labelParam3 = Label (cadre, text = "Resolution")
labelParam3.place(anchor="e", x=xS3,y=yS3)
echelle3 = Scale (cadre, from_ = 1, to = 9, command= choix_resolution_camera, orient=HORIZONTAL, length = 250, width = 10, resolution = 1, label="",showvalue=1,tickinterval=1,sliderlength=20)
echelle3.set (val_resolution)
echelle3.place(anchor="w", x=xS3,y=yS3)

#labelParam4 = Label (cadre, text = "") # choix valeur denoise software
#labelParam4.place(anchor="e", x=xS4,y=yS4)
echelle4 = Scale (cadre, from_ = 1, to = 9, command= choix_valeur_denoise, orient=HORIZONTAL, length = 90, width = 10, resolution = 1, label="",showvalue=1,tickinterval=2,sliderlength=20)
echelle4.set (val_denoise)
echelle4.place(anchor="w", x=xS4,y=yS4)

labelParam5 = Label (cadre, text = "Min") # choix valeur histogramme strech minimum
labelParam5.place(anchor="w", x=xS5,y=yS5)
echelle5 = Scale (cadre, from_ = 0, to = 200, command= choix_histo_min, orient=HORIZONTAL, length = 130, width = 10, resolution = 5, label="",showvalue=1,tickinterval=50,sliderlength=20)
echelle5.set (val_histo_min)
echelle5.place(anchor="w", x=xS5+30,y=yS5)

labelParam6 = Label (cadre, text = "Max") # choix valeur histogramme strech maximum
labelParam6.place(anchor="w", x=xS5+180,y=yS5)
echelle6 = Scale (cadre, from_ = 205, to = 255, command= choix_histo_max, orient=HORIZONTAL, length = 130, width = 10, resolution = 5, label="",showvalue=1,tickinterval=25,sliderlength=20)
echelle6.set (val_histo_max)
echelle6.place(anchor="w", x=xS5+210,y=yS5)

#labelParam8 = Label (cadre, text = "") # choix nombre captures images
#labelParam8.place(anchor="e", x=xS8,y=yS8)
echelle8 = Scale (cadre, from_ = 1, to = 251, command= choix_nb_captures, orient=HORIZONTAL, length = 350, width = 10, resolution = 1, label="",showvalue=1,tickinterval=25,sliderlength=20)
echelle8.set (val_nb_captures)
echelle8.place(anchor="w", x=xS8,y=yS8)

labelParam9 = Label (cadre, text = "Clip") # choix valeur contrate CLAHE
labelParam9.place(anchor="w", x=xS9,y=yS9)
echelle9 = Scale (cadre, from_ = 2, to = 5, command= choix_valeur_CLAHE, orient=HORIZONTAL, length = 200, width = 10, resolution = 0.5, label="",showvalue=1,tickinterval=1,sliderlength=20)
echelle9.set (val_contrast_CLAHE)
echelle9.place(anchor="w", x=xS9+30,y=yS9)

#labelParam10 = Label (cadre, text = "") # choix nombre de darks
#labelParam10.place(anchor="e", x=xS10,y=yS10)
echelle10 = Scale (cadre, from_ = 5, to = 30, command= choix_nb_darks, orient=HORIZONTAL, length = 150, width = 10, resolution =1, label="",showvalue=1,tickinterval=5,sliderlength=20)
echelle10.set (val_nb_darks)
echelle10.place(anchor="w", x=xS10,y=yS10)

#labelParam11 = Label (cadre, text = "") # choix nombre captures video
#labelParam11.place(anchor="e", x=xS11,y=yS11)
echelle11 = Scale (cadre, from_ = 0, to = 1000, command= choix_nb_video, orient=HORIZONTAL, length = 350, width = 10, resolution = 10, label="",showvalue=1,tickinterval=200,sliderlength=20)
echelle11.set (val_nb_capt_video)
echelle11.place(anchor="w", x=xS11,y=yS11)

labelParam12 = Label (cadre, text = "Phi") # choix valeur histogramme Phi Theta
labelParam12.place(anchor="w", x=xS12,y=yS12)
echelle12 = Scale (cadre, from_ = 0.1, to = 1, command= choix_phi, orient=HORIZONTAL, length = 130, width = 10, resolution = 0.05, label="",showvalue=1,tickinterval=0.3,sliderlength=20)
echelle12.set (val_phi)
echelle12.place(anchor="w", x=xS12+30,y=yS12)

labelParam13 = Label (cadre, text = "Theta") # choix valeur histogramme Phi Theta
labelParam13.place(anchor="w", x=xS12+170,y=yS12)
echelle13 = Scale (cadre, from_ = 0.1, to = 1, command= choix_theta, orient=HORIZONTAL, length = 130, width = 10, resolution = 0.05, label="",showvalue=1,tickinterval=0.3,sliderlength=20)
echelle13.set (val_theta)
echelle13.place(anchor="w", x=xS12+210,y=yS12)



# Choix appliquer dark
CBAD = Checkbutton(cadre,text="Sub Dark", variable=choix_sub_dark,command=commande_sub_dark,onvalue = 1, offvalue = 0)
CBAD.place(anchor="w",x=xdark, y=ydark)

labelInfoDark = Label (cadre, text = dispo_dark) # label info Dark
labelInfoDark.place(anchor="w", x=xdark+90,y=ydark)

labelInfo1 = Label (cadre, text = text_info1) # label info n°1
labelInfo1.place(anchor="w", x=xLI1,y=yLI1)

Button (fenetre_principale, text = "Cap Dark", command = start_cap_dark).place(x=1380,y=740, anchor=CENTER)

Button (fenetre_principale, text = "Start Pic Cap", command = start_pic_capture).place(x=1380,y=785, anchor=CENTER)
Button (fenetre_principale, text = "Stop Pic Cap", command = stop_pic_capture).place(x=1380,y=815, anchor=CENTER)

Button (fenetre_principale, text = "REC Video", command = start_video_capture).place(x=1380,y=855, anchor=CENTER)
Button (fenetre_principale, text = "Stop Video", command = stop_video_capture).place(x=1380,y=885, anchor=CENTER)

Button (fenetre_principale, text = "Quitter", command = quitter).place(x=1750,y=925, anchor=CENTER)

cadre_image = Canvas (cadre, width = cam_displ_x, height = cam_displ_y, bg = "dark grey")
cadre_image.place(anchor="w", x=20,y=cam_displ_y/2+5)



# Initialisation camera en mode rapide, ISO LOW, BIN1
init_camera()

fenetre_principale.after(200, refresh)

fenetre_principale.mainloop()
fenetre_principale.destroy()




