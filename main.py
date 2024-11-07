import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import pyautogui
import uiautomation as auto
import tkinter as tk
from threading import Thread
from PIL import Image, ImageTk

pyautogui.FAILSAFE = False

# Configuración del motor de voz
engine = pyttsx3.init()
engine.setProperty('rate', 150)

# Inicializar MediaPipe para seguimiento ocular
mp_face_mesh = mp.solutions.face_mesh.FaceMesh(refine_landmarks=True)

# Variables globales
is_tracking = False  # Controla si el seguimiento está activado o no
last_spoken = ""  # Recuerda el último elemento anunciado
cap = cv2.VideoCapture(0)  # Iniciar la captura de video

# Función para calcular el punto de la mirada en la pantalla
def get_gaze_position(landmarks, img_shape):
    ih, iw, _ = img_shape
    
    # Puntos clave alrededor de los ojos
    left_eye_points = [landmarks[362], landmarks[385], landmarks[386], landmarks[387], landmarks[263]]
    right_eye_points = [landmarks[33], landmarks[160], landmarks[159], landmarks[158], landmarks[133]]
    
    # Calcular los centros de cada ojo
    left_eye_center = np.mean(left_eye_points, axis=0).astype(int)
    right_eye_center = np.mean(right_eye_points, axis=0).astype(int)
    
    # Calcular el punto central de la mirada
    gaze_point = ((left_eye_center[0] + right_eye_center[0]) // 2,
                  (left_eye_center[1] + right_eye_center[1]) // 2)
    
    # Ajustar las coordenadas de pantalla aplicando sensibilidad en ambos ejes
    screen_x = int(((iw - gaze_point[0]) / iw * pyautogui.size().width))
    screen_y = int(gaze_point[1] / ih * pyautogui.size().height)
    return (screen_x, screen_y)

# Función para mostrar la previsualización de la cámara en la ventana de Tkinter
def update_camera_preview():
    ret, frame = cap.read()
    if ret:
        # Convertir el frame a formato RGB y luego a una imagen compatible con Tkinter
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)
    if is_tracking:
        camera_label.after(10, update_camera_preview)  # Llamar a sí mismo continuamente

# Función para suavizar el movimiento del cursor
def smooth_cursor(current_position, previous_position, smoothing_factor=0.2):
    new_x = int(previous_position[0] * (1 - smoothing_factor) + current_position[0] * smoothing_factor)
    new_y = int(previous_position[1] * (1 - smoothing_factor) + current_position[1] * smoothing_factor)
    return new_x, new_y

# Función para explorar elementos en la interfaz y detectar el elemento en foco
def get_element_in_focus(screen_point):
    global last_spoken
    focused_element = None

    # Explorar elementos en la pantalla usando uiautomation
    root = auto.GetRootControl()
    for control in root.GetChildren():
        rect = control.BoundingRectangle
        if rect and rect.left <= screen_point[0] <= rect.right and rect.top <= screen_point[1] <= rect.bottom:
            focused_element = control.Name or control.ControlTypeName
            break

    # Retroalimentación en voz del elemento en foco
    if focused_element and focused_element != last_spoken:
        engine.say(f"Mirando {focused_element}")
        engine.runAndWait()
        last_spoken = focused_element

# Función para realizar el seguimiento ocular
def start_tracking():
    global is_tracking
    previous_position = (0, 0)
    
    while is_tracking:
        ret, frame = cap.read()
        if ret:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ih, iw, _ = rgb_frame.shape
            results = mp_face_mesh.process(rgb_frame)
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    landmarks = [(int(landmark.x * iw), int(landmark.y * ih)) for landmark in face_landmarks.landmark]
                    screen_point = get_gaze_position(landmarks, rgb_frame.shape)
                    
                    # Suavizar el movimiento del cursor
                    smooth_point = smooth_cursor(screen_point, previous_position)
                    pyautogui.moveTo(smooth_point[0], smooth_point[1])
                    previous_position = smooth_point

                    # Detectar el elemento en el punto de mirada
                    get_element_in_focus(smooth_point)

# Función para iniciar el seguimiento en un hilo separado
def initiate_tracking():
    global is_tracking
    if not is_tracking:
        is_tracking = True
        tracking_thread = Thread(target=start_tracking)
        tracking_thread.start()
        update_camera_preview()  # Iniciar la previsualización de la cámara

# Función para detener el seguimiento
def stop_tracking():
    global is_tracking
    is_tracking = False

# Interfaz gráfica con Tkinter
root = tk.Tk()
root.title("Control de Seguimiento Ocular")
window_width = 600
window_height = 400

# Calcular la posición para centrar la ventana en la pantalla
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

x_position = (screen_width // 2) - (window_width // 2)
y_position = (screen_height // 2) - (window_height // 2)

# Configurar tamaño y posición de la ventana
root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

# Label para mostrar la previsualización de la cámara
camera_label = tk.Label(root)
camera_label.pack()

# Botones para iniciar y detener el seguimiento
start_button = tk.Button(root, text="Iniciar Seguimiento", command=initiate_tracking)
start_button.pack(pady=10)

stop_button = tk.Button(root, text="Detener Seguimiento", command=stop_tracking)
stop_button.pack(pady=10)

# Iniciar la interfaz de Tkinter
root.mainloop()

# Liberar la cámara al cerrar la ventana
cap.release()
cv2.destroyAllWindows()
