import cv2, sys
import jetson.inference
import jetson.utils


# Called every time a mouse event happen
def on_mouse(event, x, y, flags, userdata):
    """
    Possible States
    ---------------
    0- No points selected, nothing should be done
    1- First point selected, rectangle should be continuously drawn
    2- Second point selected, final rectangle should be drawn and coordinates yielded
    """
    global state, p1, p2, FLAG, temp
    
    # Left click
    if event == cv2.EVENT_LBUTTONDOWN:
        # Select first point
        if state == 0:
            p1 = (x,y)
            state += 1
            FLAG = True
    if event == cv2.EVENT_LBUTTONUP:
        if state == 1:
            p2 = (x,y)
            state += 1
            FLAG = False    
            
    # Right click (erase current ROI)
    if event == cv2.EVENT_RBUTTONUP:
        p1, p2 = None, None
        state = 0
        FLAG = False
     
    temp = (x,y)
    

def captureROI():
    global p1, p2, state, frame, COLOR, WEIGHT, FLAG, temp
    COLOR = (255,0,0)
    WEIGHT = 10
    
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Frame', cv2.WINDOW_NORMAL)

    # Our ROI, defined by two points
    p1, p2 = None, None
    state = 0
    FLAG = False
    # Register the mouse callback
    cv2.setMouseCallback('Frame', on_mouse)

    while cap.isOpened():
        val, frame = cap.read()
        #frame = cv2.flip(frame, 1)
        
        # If a ROI is selected, draw it
        if FLAG:
            cv2.rectangle(frame, p1, temp, COLOR, WEIGHT)
        elif not FLAG and p2:
            cv2.rectangle(frame, p1, p2, COLOR, WEIGHT)
        # Show image
        cv2.imshow('Frame', frame)
        
        # Let OpenCV manage window events
        key = cv2.waitKey(50)
        # If ESCAPE key pressed, stop
        if key == 27:
            cap.release()
        if key == 13:
            rect = (p1[0], p1[1], p2[0]-p1[0], p2[1]-p1[1])
            print(rect)
            cap.release()
            return frame, rect
            
            
def get_frames(video_name, warmup=5):
    if not video_name:
        cap = cv2.VideoCapture(0)
        # warmup
        for i in range(warmup):
            print(f"Warming up... ({i}/5)")
            cap.read()
            print("Reading...")
        print("-------Camera hot-------")
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    elif video_name.endswith('avi') or \
        video_name.endswith('mp4'):
        cap = cv2.VideoCapture(video_name)
        while True:
            ret, frame = cap.read()
            if ret:
                yield frame
            else:
                break
    else:
        images = glob(os.path.join(video_name, '*.jp*'))
        images = sorted(images,
                        key=lambda x: int(x.split('/')[-1].split('.')[0]))
        for img in images:
            frame = cv2.imread(img)
            yield frame


def contains_person(frame, net):
    """
    Checks for the existence of a person in the frame
    """
    
    width, height = frame.shape[1], frame.shape[2]
    detections = net.Detect(frame, width, height)
    for detection in detections:
        if (detection.ClassID == 1): 
            print("""
                  ----------------------------------------------------
                  ----------WARNING!!! TARGET NEAR PERSONNEL----------
                  ----------------------------------------------------
                  ----------PRESS ENTER TO CONFIRM, ELSE ANY KEY------
                  ----------------------------------------------------
                  """) 
            return True
    return False
               

