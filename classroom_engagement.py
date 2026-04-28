import cv2
import dlib
import numpy as np
from scipy.spatial import distance
import time
import pandas as pd
from collections import OrderedDict
import copy

# ================================
# 1. HELPER FUNCTIONS
# ================================
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    EAR = (A + B) / (2.0 * C)
    return EAR

def lips_aspect_ratio(lips):
    A = distance.euclidean(lips[4], lips[8])
    B = distance.euclidean(lips[2], lips[10])
    C = distance.euclidean(lips[0], lips[6])
    LAR = (A + B + C) / 3.0
    return LAR

def get_head_pose(shape, size):
    # 3D model points.
    model_points = np.array([
        (0.0, 0.0, 0.0),             # Nose tip 30
        (0.0, -330.0, -65.0),        # Chin 8
        (-225.0, 170.0, -135.0),     # Left eye left corner 36
        (225.0, 170.0, -135.0),      # Right eye right corne 45
        (-150.0, -150.0, -125.0),    # Left Mouth corner 48
        (150.0, -150.0, -125.0)      # Right mouth corner 54
    ])
    
    # 2D image points from the facial landmarks
    image_points = np.array([
        (shape.part(30).x, shape.part(30).y),     # Nose tip
        (shape.part(8).x, shape.part(8).y),       # Chin
        (shape.part(36).x, shape.part(36).y),     # Left eye left corner
        (shape.part(45).x, shape.part(45).y),     # Right eye right corner
        (shape.part(48).x, shape.part(48).y),     # Left Mouth corner
        (shape.part(54).x, shape.part(54).y)      # Right mouth corner
    ], dtype="double")
    
    # Camera internals
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array(
                         [[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype = "double"
                         )
    
    dist_coeffs = np.zeros((4,1)) # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    
    # Using Rodriguez to get rotation matrix
    rmat, jac = cv2.Rodrigues(rotation_vector)
    
    # Get the angles
    angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
    x = angles[0] * 360 # Pitch (Up/Down)
    y = angles[1] * 360 # Yaw (Left/Right)
    z = angles[2] * 360 # Roll
    
    status = "Focused"
    if y < -10:
        status = "Looking Right"
    elif y > 10:
        status = "Looking Left"
    elif x < -10:
        status = "Looking Down"
    
    return status

# ================================
# 2. CENTROID TRACKER
# ================================
class CentroidTracker:
    def __init__(self, maxDisappeared=50):
        self.nextObjectID = 0
        self.objects = OrderedDict()
        self.disappeared = OrderedDict()
        self.maxDisappeared = maxDisappeared

    def register(self, centroid):
        self.objects[self.nextObjectID] = centroid
        self.disappeared[self.nextObjectID] = 0
        self.nextObjectID += 1

    def deregister(self, objectID):
        del self.objects[objectID]
        del self.disappeared[objectID]

    def update(self, rects):
        if len(rects) == 0:
            for objectID in list(self.disappeared.keys()):
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)
            return self.objects

        inputCentroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (startX, startY, endX, endY)) in enumerate(rects):
            cX = int((startX + endX) / 2.0)
            cY = int((startY + endY) / 2.0)
            inputCentroids[i] = (cX, cY)

        if len(self.objects) == 0:
            for i in range(0, len(inputCentroids)):
                self.register(inputCentroids[i])
        else:
            objectIDs = list(self.objects.keys())
            objectCentroids = list(self.objects.values())
            D = distance.cdist(np.array(objectCentroids), inputCentroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            usedRows = set()
            usedCols = set()

            for (row, col) in zip(rows, cols):
                if row in usedRows or col in usedCols:
                    continue
                objectID = objectIDs[row]
                self.objects[objectID] = inputCentroids[col]
                self.disappeared[objectID] = 0
                usedRows.add(row)
                usedCols.add(col)

            unusedRows = set(range(0, D.shape[0])).difference(usedRows)
            unusedCols = set(range(0, D.shape[1])).difference(usedCols)

            for row in unusedRows:
                objectID = objectIDs[row]
                self.disappeared[objectID] += 1
                if self.disappeared[objectID] > self.maxDisappeared:
                    self.deregister(objectID)

            for col in unusedCols:
                self.register(inputCentroids[col])

        return self.objects

# ================================
# 3. MAIN CLASSROOM SYSTEM
# ================================
def main():
    print("[INFO] Loading facial landmark predictor...")
    cap = cv2.VideoCapture(0)
    hog_face_detector = dlib.get_frontal_face_detector()
    dlib_facelandmark = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat/shape_predictor_68_face_landmarks.dat")
    
    tracker = CentroidTracker(maxDisappeared=40)
    
    # Configs
    EAR_THRESH = 0.26
    LAR_THRESH = 0.35 # Custom threshold for talking
    
    # Store analytics data per student ID
    # data format: student_id: {'sleep_frames': 0, 'talk_frames': 0, 'distracted_frames': 0, 'focus_frames': 0}
    session_data = {}

    print("[INFO] Starting Classroom Engagement Monitor...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.resize(frame, (800, 600))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # 1. Detect multiple faces
        faces = hog_face_detector(gray)
        rects = []
        
        for face in faces:
            # Map for tracker
            rects.append((face.left(), face.top(), face.right(), face.bottom()))
            
        # 2. Assign unique IDs to faces
        objects = tracker.update(rects)
        
        # Build mapping from centroid back to face rect (naive mapping)
        for (objectID, centroid) in objects.items():
            if objectID not in session_data:
                session_data[objectID] = {
                    'sleep_frames': 0, 
                    'talk_frames': 0, 
                    'distracted_frames': 0, 
                    'focus_frames': 0
                }
                
            # Find closest face rect to this centroid
            closest_face = None
            min_dist = float('inf')
            for face in faces:
                fc = (int((face.left() + face.right()) / 2.0), int((face.top() + face.bottom()) / 2.0))
                d = distance.euclidean(centroid, fc)
                if d < min_dist:
                    min_dist = d
                    closest_face = face
                    
            if closest_face and min_dist < 50:
                face = closest_face
                face_landmarks = dlib_facelandmark(gray, face)
                
                # --- A. Sleepiness Detection (EAR) ---
                leftEye = [(face_landmarks.part(n).x, face_landmarks.part(n).y) for n in range(36, 42)]
                rightEye = [(face_landmarks.part(n).x, face_landmarks.part(n).y) for n in range(42, 48)]
                ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
                
                # --- B. Talking/Whispering Detection (LAR) ---
                lips = [(face_landmarks.part(n).x, face_landmarks.part(n).y) for n in range(48, 68)]
                lar = lips_aspect_ratio(lips)
                
                # --- C. Attention Focus (Head Pose) ---
                size = frame.shape
                focus_status = get_head_pose(face_landmarks, size)
                
                # Update Session Data
                if ear < EAR_THRESH:
                    session_data[objectID]['sleep_frames'] += 1
                    status_text = "Sleeping!"
                    color = (0, 0, 255)
                elif lar > LAR_THRESH:
                    session_data[objectID]['talk_frames'] += 1
                    status_text = "Talking"
                    color = (0, 165, 255) # Orange
                elif focus_status != "Focused":
                    session_data[objectID]['distracted_frames'] += 1
                    status_text = focus_status
                    color = (0, 255, 255) # Yellow
                else:
                    session_data[objectID]['focus_frames'] += 1
                    status_text = "Focused"
                    color = (0, 255, 0) # Green
                
                # Draw on frame
                cv2.rectangle(frame, (face.left(), face.top()), (face.right(), face.bottom()), color, 2)
                cv2.putText(frame, f"Seat {objectID}: {status_text}", (face.left(), face.top() - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                            
        # Info panel
        cv2.putText(frame, "Press 'Esc' to exit and generate report.", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Classroom Engagement Monitor", frame)

        key = cv2.waitKey(1)
        if key == 27: # Esc key
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
    # 4. Generate Reporting Dashboard / CSV
    print("\n[INFO] Generating Session Engagement Report...")
    report_list = []
    fps = 10 # approximate fps
    for obj_id, data in session_data.items():
        total = data['focus_frames'] + data['distracted_frames'] + data['sleep_frames'] + data['talk_frames']
        if total == 0:
            continue
            
        focus_pct = round((data['focus_frames'] / total) * 100, 1)
        # Convert frames to approximate seconds
        report_list.append({
            "Student/Seat ID": f"Seat {obj_id}",
            "Focus Score (%)": focus_pct,
            "Total Focused (sec)": data['focus_frames'] // fps,
            "Total Distracted (sec)": data['distracted_frames'] // fps,
            "Total Sleeping (sec)": data['sleep_frames'] // fps,
            "Total Talking (sec)": data['talk_frames'] // fps
        })
        
    df = pd.DataFrame(report_list)
    df.to_csv("classroom_report.csv", index=False)
    print(df.to_string(index=False))
    print("\n[SUCCESS] Report generated: 'classroom_report.csv'")

if __name__ == "__main__":
    main()
