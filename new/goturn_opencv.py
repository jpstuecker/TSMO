import cv2

tracker = cv2.TrackerGOTURN_create()

cap = cv2.VideoCapture(0)
_, frame = cap.read()

bbox = cv2.selectROI(frame, False)

ok = tracker.init(frame, bbox)

while cap.isOpened():
	ok, frame = cap.read()

	timer = cv2.getTickCount()

	ok, bbox = tracker.update(frame)

	fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer)

	if ok:
		p1 = (int(bbox[0]), int(bbox[1]))
		p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
		cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
	else:
		cv2.putText(frame, "TRACKING FAILURE", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)

	cv2.putText(frame, "GOTURN Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)
	cv2.putText(frame, f"FPS: {fps}", (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2)

	cv2.imshow("Tracking", frame)

	k = cv2.waitKey(1) & 0xff
	if k == 27: break