import cv2
import supervision as sv
import numpy as np
from ultralytics import YOLO
import time
import math

h=540
w=960

poly_points=[(328,90),(0,190),(0,h),(w,h),(w,198),(642,90)]
#poly_points=[(320,120),(0,270),(210,270),(610,120)]

poly_points=np.array(poly_points,dtype=np.int32)

img_points=np.array([(328,90),(-410,540),(746,540),(642,90)],dtype=np.float32)
rec_points=np.array([(0,0),(0,300),(40,300),(40,0)],dtype=np.float32)


last_center_point=[
    [],
    []
]
last_time=[
    [],
    []
]
last_tracking_id=[]

def convert_real_coordinate(source,target,img_coordinate):
    source=source.astype(np.float32)
    target=target.astype(np.float32)
    img_coordinate=np.array(img_coordinate,dtype=np.float32)
    img_coordinate=img_coordinate.astype(np.float32)
    
    m_matrix=cv2.getPerspectiveTransform(source,target)
    img_coordinate=img_coordinate.reshape(-1,1,2)
    transfrom_coordinate=cv2.perspectiveTransform(img_coordinate,m_matrix)
    transfrom_coordinate=np.array(transfrom_coordinate,dtype=np.float32)
    #print("Matrix: ",m_matrix)
    #print("real: ",transfrom_coordinate)
    return transfrom_coordinate

def click_event(event, x, y, flags, params): 
   
    # checking for left mouse clicks 
    if event == cv2.EVENT_LBUTTONDBLCLK: 
        val=(x,y)
        #poly_points.append(val)
        print(x, ' ', y) 

    time.sleep(1)
    
def get_processing_region(img,vertices):
    region=np.zeros_like(frame)
    mask=np.zeros_like(img)
    match_mask_color=(255,255,255)
    cv2.fillPoly(mask,vertices,match_mask_color)
    mask_img=cv2.bitwise_and(img,mask)
    region=cv2.add(region,mask_img)
    return region

def speed(coordinate_1,coordinate_2,previous_time):
    
    x1,y1=coordinate_1[0]
    x2,y2=coordinate_2
    
    print("x1 y1",x1, y1)
    print("x2 y2",x2, y2)
    
    current_time=time.time()
    distance=math.sqrt(pow(x1-x2,2)+pow(y1-y2,2))
    duration=current_time-previous_time
    
    print("x1-x2", x1-x2)
    print("y1-y2", y1-y2)
    
    print("Previous_time:",previous_time)
    print("current_time: ",current_time)
    
    print("Distance: ",distance)
    print("Duration: ",duration)
    
    return round(distance*3.6/duration,2)

def get_element_value(list,tracking_id):
    
    for index,val in enumerate(list[1]):
        print(val,tracking_id,index)
        if val==tracking_id: return list[0][index],index
    return None
   
        

model=YOLO("yolov8n.pt")
tracker=sv.ByteTrack()

cap=cv2.VideoCapture("C:/Users/ASUS/Downloads/highway_traffic (online-video-cutter.com) (2).mp4")

while cap.isOpened():
    _,frame=cap.read()
    frame=cv2.resize(frame,(0,0),fx=0.5,fy=0.5)
    
    print(frame.shape)
    
    detect_frame=get_processing_region(frame,[poly_points])
    
    results=model(detect_frame)[0]
    coordinates=results.boxes.xyxy
    
    detections=sv.Detections.from_ultralytics(results)
    detections=tracker.update_with_detections(detections)
    
    class_names_id=detections.class_id
    trackings_id=detections.tracker_id
    print(trackings_id)
    confidents=results.boxes.conf
    
    lables=[f'{class_name_id} #{tracking_id}' for class_name_id,tracking_id in zip(class_names_id,trackings_id)]
            
    frame=cv2.polylines(frame,[poly_points],True,(100,200,0),1)
    
    i=0
    for coordinate, class_name_id, tracking_id,lable,confident in zip(coordinates,class_names_id,trackings_id,lables,confidents):
        x1=int(coordinate[0])
        y1=int(coordinate[1])
            
        x2=int(coordinate[2])
        y2=int(coordinate[3])
        
        center=((x1+x2)/2,y2)
        cv2.circle(frame,(int((x1+x2)/2),y2),1,(0,122,30),3)
        center=convert_real_coordinate(img_points,rec_points,center)
        print(center)

        #check= check_pos(poly_points,center)
        if results.names[int(class_name_id)]=="car" and float(confident)>=0.45:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(211,111,109),2)
            cv2.rectangle(detect_frame,(x1,y1),(x2,y2),(211,111,109),2)
            
            previous_center_point=None
            previous_time=None
            
            if tracking_id in last_tracking_id:
                
                
                previous_center_point,ct_index=get_element_value(last_center_point,tracking_id)
                previous_center_point=previous_center_point[0]
               
                previous_time,t_index=get_element_value(last_time,tracking_id)
                print("==================================")
                print("Id",tracking_id)
                print("Pre_point",previous_center_point)
                #print("Center",center[0])
                print("==================================")
               
                val=speed(center[0],previous_center_point[0],previous_time)
                cv2.putText(frame,f'{val} {tracking_id}',(x1,y1-10),cv2.FONT_HERSHEY_COMPLEX,0.5,(100,21,109),1)
            
            if previous_center_point is not None and previous_time is not None:
                last_tracking_id[ct_index]=tracking_id
                last_center_point[0][ct_index]=center
                
                last_time[0][t_index]=time.time()
                last_time[1][t_index]=tracking_id
                 
            else:
                last_center_point[0].append(center)
                last_center_point[1].append(tracking_id)

                last_time[0].append(time.time())
                last_time[1].append(tracking_id)
                
                last_tracking_id.append(tracking_id)
                
            print("nav",last_center_point)
            i=i+1
    cv2.imshow("Camera",frame)
    cv2.imshow("Camera_detect",detect_frame)
    cv2.setMouseCallback("Camera",click_event)
    
    if cv2.waitKey(1)==ord('q'): break
        
    
print("Completed!")
cap.release()
cv2.destroyAllWindows()

