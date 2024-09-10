import cv2
import numpy as np
import time
import os
import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
from os import makedirs
from os.path import join, exists
from csv import DictReader
from matplotlib import pyplot as plt

mp_hands = mp.solutions.hands

'''
        FONCTION DE MODIFICATION DU SON WINDOWS

'''
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

def set_master_volume(volume):
    from comtypes import CLSCTX_ALL
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume_object = interface.QueryInterface(IAudioEndpointVolume)
    volume_object.SetMasterVolumeLevelScalar(volume, None)
    
def get_master_volume():
    from comtypes import CLSCTX_ALL
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume_object = interface.QueryInterface(IAudioEndpointVolume)
    return volume_object.GetMasterVolumeLevelScalar()
    
    
    

def lm_extraction(hands_res):
    '''
            EXTRACTION LES LANDMARKS DE LA MAINS

    '''

    # initialisation des lm ###############################################
    hand_lm = []
    if hands_res.multi_hand_landmarks:
        # on parcours toutes les detection ###############################################
        hand_landmarks = hands_res.multi_hand_landmarks[0]
        # Extraire les coordonnées x, y et z de chaque landmark
        for landmark in hand_landmarks.landmark:
            hand_lm.append([landmark.x, landmark.y, landmark.z])
        # liste 2 np array ###############################################
        hand_lm = np.array(hand_lm)
    else:
        hand_lm = np.zeros((21, 3))

    return hand_lm


def lm_drawing (image,hand_resultats):
    for hand_landmarks in hand_resultats.multi_hand_landmarks:
                #printing des landmarks sur l'image
                mp_drawing.draw_landmarks(image,hand_landmarks, mp_hands.HAND_CONNECTIONS,
                                          landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=1),
                                          connection_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2, circle_radius=2)
                                         )




def angle_process(hand_lm):
    vect = []
    angle = []
    # Format (base,point1,p2) note(le p1 est le plus sur l'extrémité)
    # (base de l'articulation)
    # Angle de l'index #################
    vect.append([5, 8, 0])
    vect.append([5, 6, 0])
    vect.append([6, 8, 5])

    # Angle du majeur #################
    vect.append([9, 12, 0])
    vect.append([9, 10, 0])
    vect.append([10, 12, 9])

    # Angle l'anulaire #################
    vect.append([13, 16, 0])
    vect.append([13, 14, 0])
    vect.append([14, 16, 13])

    # Angle de l'oriculaire #################
    vect.append([17, 20, 0])
    vect.append([17, 18, 0])
    vect.append([18, 20, 17])

    # Angle ddu pouce #################
    vect.append([2, 4, 0])
    vect.append([2, 3, 0])
    vect.append([1, 4, 5])

    for b, p1, p2 in vect:
        vec1 = hand_lm[p1] - hand_lm[b]
        vec2 = hand_lm[p2] - hand_lm[b]
        angle.append(np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))))
    angle = np.array(angle)
    return angle




def distance_process(lm):
    dist=np.zeros((21))
    # ditance de la pomme
    p1 = 9
    p2 = 0
    dist[0] = np.linalg.norm(lm[p1] - lm[p2])

    #pouce  (p1=4  p2=2)
    p1=4
    p2=2
    dist[1]=np.linalg.norm(lm[p1] - lm[p2])

    # index  (p1=8  p2=5)
    p1 = 8
    p2 = 5
    dist[2] = np.linalg.norm(lm[p1] - lm[p2])

    # mùajeur  (p1=12  p2=9)
    p1 = 12
    p2 = 9
    dist[3] = np.linalg.norm(lm[p1] - lm[p2])

    # annulaire  (p1=16  p2=13)
    p1 = 16
    p2 = 13
    dist[4] = np.linalg.norm(lm[p1] - lm[p2])

    # auriculaire  (p1=20  p2=17)
    p1 = 20
    p2 = 17
    dist[5] = np.linalg.norm(lm[p1] - lm[p2])


    #DISTANCE ENTRE LES DOIGTS
    #liens au pouce (p1=4)
    p1=4
    p2=8
    dist[6] = np.linalg.norm(lm[p1] - lm[p2])
    p2 = 12
    dist[7] = np.linalg.norm(lm[p1] - lm[p2])
    p2 = 16
    dist[8] = np.linalg.norm(lm[p1] - lm[p2])
    p2 = 20
    dist[9] = np.linalg.norm(lm[p1] - lm[p2])
    p2 = 0
    dist[10] = np.linalg.norm(lm[p1] - lm[p2])

    # liens a l'index (p1=8)
    p1=8
    p2 = 12
    dist[11] = np.linalg.norm(lm[p1] - lm[p2])
    p2 = 16
    dist[12] = np.linalg.norm(lm[p1] - lm[p2])
    p2 = 20
    dist[13] = np.linalg.norm(lm[p1] - lm[p2])
    p2 = 0
    dist[14] = np.linalg.norm(lm[p1] - lm[p2])

    # liens au majeur (p1=12)
    p1=12
    p2 = 16
    dist[15] = np.linalg.norm(lm[p1] - lm[p2])
    p2 = 20
    dist[16] = np.linalg.norm(lm[p1] - lm[p2])
    p2 = 0
    dist[17] = np.linalg.norm(lm[p1] - lm[p2])

    # liens a l'anulaire (p1=16)
    p1 = 16
    p2 = 20
    dist[18] = np.linalg.norm(lm[p1] - lm[p2])
    p2 = 0
    dist[19] = np.linalg.norm(lm[p1] - lm[p2])

    # liens a l'oriculair (p1=20)
    p1 = 20
    p2 = 0
    dist[20] = np.linalg.norm(lm[p1] - lm[p2])

    return dist


def save_lm(lm, folder):

    if not os.path.exists(folder):
        os.makedirs(folder)
    file_list = [f for f in os.listdir(folder) if f.startswith("lm_") and f.endswith(".npy")]
    file_count = len(file_list)
    filename = folder + "/lm_" + str(file_count) + ".npy"
    np.save(filename, lm)
    return 0

def load_lm(folder):
    file_list = [f for f in os.listdir(folder) if (f.startswith("lm_") or f.startswith("data_") ) and f.endswith(".npy")]
    file_list.sort()
    data = [np.load(folder + "/" + f) for f in file_list]
    return data


def proba_viz(vect_prob,image):
    import csv
    with open('./classe.csv') as mon_fichier:
        mon_fichier_reader = csv.reader(mon_fichier, delimiter=',', quotechar='"')
        classe_list = [x for x in mon_fichier_reader]
    classe_list=classe_list[0]
        
    for idx,prob in enumerate(vect_prob) :
        cv2.rectangle(image, (0,idx*25), (1+int(200*1),25+idx*25), (15,15,15), -1)
        cv2.rectangle(image, (0,idx*25), (1+int(200*prob),25+idx*25), (80,130,80), -1)
        cv2.putText(image, str( classe_list[idx] )+' : '+str(round(prob,2)), org=(0,20+idx*25), fontFace=cv2.FONT_HERSHEY_SIMPLEX ,
                                      fontScale=0.6, color=(255,255,255), thickness=1)
        
        
        

def plot_log(filename, show=None):
    """ Plot log of training / validation learning curve
    # Arguments
        :param filename: str, csv log file name
        :param show: None / str, show graph if none or save to 'show' directory
    """
    # Load csv file
    keys, values, idx = [], [], None
    with open(filename, 'r') as f:
        reader = DictReader(f)
        for row in reader:
            if len(keys) == 0:
                for key, value in row.items():
                    keys.append(key)
                    values.append(float(value))
                idx = keys.index('epoch')
                continue
            for _, value in row.items():
                values.append(float(value))
        values = np.reshape(values, newshape=(-1, len(keys)))
        values[:, idx] += 1
    # Plot
    fig = plt.figure(figsize=(8,6))
    fig.subplots_adjust(top=0.95, bottom=0.05, right=0.95)
    fig.add_subplot(211)
    for i, key in enumerate(keys):
        # training loss
        if key.find('loss') >= 0:   # and not key.find('val') >= 0:
            plt.plot(values[:, idx], values[:, i], label=key)
    plt.legend()
    plt.title('Training loss')
    fig.add_subplot(212)
    for i, key in enumerate(keys):
        # acc
        if key.find('acc') >= 0:
            plt.plot(values[:, idx], values[:, i], label=key)
    plt.legend()
    plt.title('Training and validation accuracy')
    if show is not None:
        fig.savefig(join(show, 'log.png'))
    else:
        plt.show()
