import cv2

fs = cv2.FileStorage("C:\\Users\\mkouk\\OneDrive\\画像\\カメラ ロール\\WIN_20230915_16_32_46_Pro.jpg", cv2.FILE_STORAGE_READ)
if fs.isOpened():
        width = (int)(camera_fs.getNode("cameraResolution").at(0).real())
        height = (int)(camera_fs.getNode("cameraResolution").at(1).real())
        camera_matrix = camera_fs.getNode("cameraMatrix").mat()

fovx, fovy, focal_length, principal_point, aspect_ratio = cv2.calibrationMatrixValues(camera_matrix,
            (width, height), aperture_width, aperture_height)

print("  FOVX = ", "{:.6f}".format(fovx))
print("  FOVY = ", "{:.6f}".format(fovy))
print("  Focal length    = ", "{:.6f}".format(focal_length))
print("  Principal point = ", "{:.6f}, {:6f}".format(*principal_point))
print("  Aspect ratio    = ", "{:.6f}".format(aspect_ratio))


