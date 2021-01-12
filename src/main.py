import cv2
import numpy as np

def detect(img, maskr, maskg, masky, md, sr, br, param1, param2):
    
    # Set font for CV2 GUI
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Store dimensions of input image as a variable
    size = img.shape

    # Detect circles present in mask with Hough Circle Transform
    r_circles = cv2.HoughCircles(maskr, cv2.HOUGH_GRADIENT, 1, md,
                               param1=param1, param2=param2, minRadius=sr, maxRadius=br)

    g_circles = cv2.HoughCircles(maskg, cv2.HOUGH_GRADIENT, 1, md,
                               param1=param1, param2=param2, minRadius=sr, maxRadius=br)
    

    y_circles = cv2.HoughCircles(masky, cv2.HOUGH_GRADIENT, 1, md,
                               param1=param1, param2=param2, minRadius=sr, maxRadius=br)

    # Detect traffic lights
    r = 5
    bound = 4.0 / 10
    if r_circles is not None:
        r_circles = np.uint16(np.around(r_circles))

        for i in r_circles[0, :]:
            if i[0] > size[1] or i[1] > size[0]or i[1] > size[0]*bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):

                    if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                        continue
                    h += maskr[i[1]+m, i[0]+n]
                    s += 1
            if h / s > 50:
                cv2.circle(img, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                cv2.circle(maskr, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                cv2.putText(img,'RED',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)

    if g_circles is not None:
        g_circles = np.uint16(np.around(g_circles))

        for i in g_circles[0, :]:
            if i[0] > size[1] or i[1] > size[0] or i[1] > size[0]*bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):

                    if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                        continue
                    h += maskg[i[1]+m, i[0]+n]
                    s += 1
            if h / s > 100:
                cv2.circle(img, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                cv2.circle(maskg, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                cv2.putText(img,'GREEN',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)

    if y_circles is not None:
        y_circles = np.uint16(np.around(y_circles))

        for i in y_circles[0, :]:
            if i[0] > size[1] or i[1] > size[0] or i[1] > size[0]*bound:
                continue

            h, s = 0.0, 0.0
            for m in range(-r, r):
                for n in range(-r, r):

                    if (i[1]+m) >= size[0] or (i[0]+n) >= size[1]:
                        continue
                    h += masky[i[1]+m, i[0]+n]
                    s += 1
            if h / s > 50:
                cv2.circle(img, (i[0], i[1]), i[2]+10, (0, 255, 0), 2)
                cv2.circle(masky, (i[0], i[1]), i[2]+30, (255, 255, 255), 2)
                cv2.putText(img,'YELLOW',(i[0], i[1]), font, 1,(255,0,0),2,cv2.LINE_AA)
                
    return img

if __name__ == '__main__':
    
    # Dummy function to pass as parameter when needed
    def nothing(x):
        pass

    # Name each window with String
    redWindow = 'Red'
    greenWindow = 'Green'
    yellowWindow = 'Yellow'
    paramWindow = 'Hough Circles Parameters'
    
    # Name each slider
    rhl = 'H Low'
    rhh = 'H High'
    rsl = 'S Low'
    rsh = 'S High'
    rvl = 'V Low'
    rvh = 'V High'
    
    ghl = 'H Low'
    ghh = 'H High'
    gsl = 'S Low'
    gsh = 'S High'
    gvl = 'V Low'
    gvh = 'V High'
    
    yhl = 'H Low'
    yhh = 'H High'
    ysl = 'S Low'
    ysh = 'S High'
    yvl = 'V Low'
    yvh = 'V High'
    
    md = 'Min Dist'
    sr = 'Min Radius'
    br = 'Max Radius'
    p1 = 'Param 1'
    p2 = 'Param 2'
    
    # Create separate window for each colour and the hough circle transform parameters
    cv2.namedWindow(redWindow, flags = cv2.WINDOW_NORMAL)
    cv2.namedWindow(greenWindow, flags = cv2.WINDOW_NORMAL)
    cv2.namedWindow(yellowWindow, flags = cv2.WINDOW_NORMAL)
    cv2.namedWindow(paramWindow, flags = cv2.WINDOW_NORMAL)
    
    # Create trackbars
    cv2.createTrackbar(rhl, redWindow, 0, 255, nothing)
    cv2.createTrackbar(rhh, redWindow, 0, 255, nothing)
    cv2.createTrackbar(rsl, redWindow, 0, 255, nothing)
    cv2.createTrackbar(rsh, redWindow, 0, 255, nothing)
    cv2.createTrackbar(rvl, redWindow, 0, 255, nothing)
    cv2.createTrackbar(rvh, redWindow, 0, 255, nothing)
    
    cv2.createTrackbar(ghl, greenWindow, 0, 255, nothing)
    cv2.createTrackbar(ghh, greenWindow, 0, 255, nothing)
    cv2.createTrackbar(gsl, greenWindow, 0, 255, nothing)
    cv2.createTrackbar(gsh, greenWindow, 0, 255, nothing)
    cv2.createTrackbar(gvl, greenWindow, 0, 255, nothing)
    cv2.createTrackbar(gvh, greenWindow, 0, 255, nothing)
    
    cv2.createTrackbar(yhl, yellowWindow, 0, 255, nothing)
    cv2.createTrackbar(yhh, yellowWindow, 0, 255, nothing)
    cv2.createTrackbar(ysl, yellowWindow, 0, 255, nothing)
    cv2.createTrackbar(ysh, yellowWindow, 0, 255, nothing)
    cv2.createTrackbar(yvl, yellowWindow, 0, 255, nothing)
    cv2.createTrackbar(yvh, yellowWindow, 0, 255, nothing)
    
    cv2.createTrackbar(md, paramWindow, 1, 255, nothing)
    cv2.createTrackbar(sr, paramWindow, 1, 255, nothing)
    cv2.createTrackbar(br, paramWindow, 1, 255, nothing)
    cv2.createTrackbar(p1, paramWindow, 1, 255, nothing)
    cv2.createTrackbar(p2, paramWindow, 1, 255, nothing)
    
    # Set initial value for each trackbar
    cv2.setTrackbarPos(rhl, redWindow, 5)
    cv2.setTrackbarPos(rhh, redWindow, 40)
    cv2.setTrackbarPos(rsl, redWindow, 180)
    cv2.setTrackbarPos(rsh, redWindow, 255)
    cv2.setTrackbarPos(rvl, redWindow, 180)
    cv2.setTrackbarPos(rvh, redWindow, 255)
    
    cv2.setTrackbarPos(ghl, greenWindow, 40)
    cv2.setTrackbarPos(ghh, greenWindow, 100)
    cv2.setTrackbarPos(gsl, greenWindow, 30)
    cv2.setTrackbarPos(gsh, greenWindow, 255)
    cv2.setTrackbarPos(gvl, greenWindow, 40)
    cv2.setTrackbarPos(gvh, greenWindow, 255)
    
    cv2.setTrackbarPos(yhl, yellowWindow, 30)
    cv2.setTrackbarPos(yhh, yellowWindow, 50)
    cv2.setTrackbarPos(ysl, yellowWindow, 100)
    cv2.setTrackbarPos(ysh, yellowWindow, 255)
    cv2.setTrackbarPos(yvl, yellowWindow, 240)
    cv2.setTrackbarPos(yvh, yellowWindow, 255)
    
    cv2.setTrackbarPos(md, paramWindow, 50)
    cv2.setTrackbarPos(sr, paramWindow, 4)
    cv2.setTrackbarPos(br, paramWindow, 14)
    cv2.setTrackbarPos(p1, paramWindow, 50)
    cv2.setTrackbarPos(p2, paramWindow, 9)
    
    # Capture video from device
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1360)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 800)
    
    while(True):
        # Extract current frame from capture device
        ret, frame = cap.read()
        
        # Convert colour space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
        # Get slider position
        rhlv = cv2.getTrackbarPos(rhl, redWindow)
        rhhv = cv2.getTrackbarPos(rhh, redWindow)
        rslv = cv2.getTrackbarPos(rsl, redWindow)
        rshv = cv2.getTrackbarPos(rsh, redWindow)
        rvlv = cv2.getTrackbarPos(rvl, redWindow)
        rvhv = cv2.getTrackbarPos(rvh, redWindow)
        
        ghlv = cv2.getTrackbarPos(ghl, greenWindow)
        ghhv = cv2.getTrackbarPos(ghh, greenWindow)
        gslv = cv2.getTrackbarPos(gsl, greenWindow)
        gshv = cv2.getTrackbarPos(gsh, greenWindow)
        gvlv = cv2.getTrackbarPos(gvl, greenWindow)
        gvhv = cv2.getTrackbarPos(gvh, greenWindow)
        
        yhlv = cv2.getTrackbarPos(yhl, yellowWindow)
        yhhv = cv2.getTrackbarPos(yhh, yellowWindow)
        yslv = cv2.getTrackbarPos(ysl, yellowWindow)
        yshv = cv2.getTrackbarPos(ysh, yellowWindow)
        yvlv = cv2.getTrackbarPos(yvl, yellowWindow)
        yvhv = cv2.getTrackbarPos(yvh, yellowWindow)
        
        mdv = cv2.getTrackbarPos(md, paramWindow)
        srv = cv2.getTrackbarPos(sr, paramWindow)
        brv = cv2.getTrackbarPos(br, paramWindow)
        p1v = cv2.getTrackbarPos(p1, paramWindow)
        p2v = cv2.getTrackbarPos(p2, paramWindow)
    
        # Parse slider position as array to make it neat
        RHSVLOW = np.array([rhlv, rslv, rvlv])
        RHSVHIGH = np.array([rhhv, rshv, rvhv])
        GHSVLOW = np.array([ghlv, gslv, gvlv])
        GHSVHIGH = np.array([ghhv, gshv, gvhv])
        YHSVLOW = np.array([yhlv, yslv, yvlv])
        YHSVHIGH = np.array([yhhv, yshv, yvhv])
    
        # Colour segmentation
        maskr = cv2.inRange(hsv, RHSVLOW, RHSVHIGH)
        maskg = cv2.inRange(hsv, GHSVLOW, GHSVHIGH)
        masky = cv2.inRange(hsv, YHSVLOW, YHSVHIGH)
        after = detect(frame, maskr, maskg, masky, mdv, srv, brv, p1v, p2v)
        
        # Output the final composite, along with each individual masks (for debugging)
        cv2.imshow('Output', after)
        cv2.imshow('Red Mask', maskr)
        cv2.imshow('Green Mask', maskg)
        cv2.imshow('Yellow Mask', masky)

        # Terminate program if user hits q
        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    # Release resources before exiting
    cap.release()
    cv2.destroyAllWindows()
