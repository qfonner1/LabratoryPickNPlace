import numpy as np


def RotX(angle):
    Rx = np.array([[1, 0, 0],[0, np.cos(angle), -np.sin(angle)],[0, np.sin(angle),  np.cos(angle)]])
    return Rx

def RotY(angle):
    Ry = np.array([[ np.cos(angle), 0, np.sin(angle)],[ 0,1,0],[-np.sin(angle), 0, np.cos(angle)]])
    return Ry

def RotZ(angle):
    Rz = np.array([[ np.cos(angle), -np.sin(angle), 0],[np.sin(angle), np.cos(angle),0],[ 0,0,1]])
    return Rz