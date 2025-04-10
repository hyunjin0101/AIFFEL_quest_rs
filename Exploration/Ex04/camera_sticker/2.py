# Load face image and whiskers image
face_path = r"C:\Users\hjin0\Desktop\aiffel\camera_sticker\1.png"  # previously uploaded
whiskers_path = r'C:\Users\hjin0\AppData\Roaming\SPB_Data\aiffel\camera_sticker_test\images\cat.png'
face_img = cv2.imread(face_path)
whiskers_img = cv2.imread(whiskers_path, cv2.IMREAD_UNCHANGED)

# Simulated landmark for nose tip from dlib (based on visual estimation for this face)
face_h, face_w, _ = face_img.shape
x_nose = face_w // 2
y_nose = face_h // 2 + 25  # adjusted for natural nose position in uploaded image

# Resize whiskers based on face width
whiskers_width = face_w // 2
aspect_ratio = whiskers_img.shape[0] / whiskers_img.shape[1]
whiskers_height = int(whiskers_width * aspect_ratio)
whiskers_resized = cv2.resize(whiskers_img, (whiskers_width, whiskers_height))

# Position whiskers just below the nose
x_offset = x_nose - whiskers_width // 2
y_offset = y_nose + 10

# Overlay transparent PNG
def overlay_transparent(background, overlay, x, y):
    b, g, r, a = cv2.split(overlay)
    overlay_rgb = cv2.merge((b, g, r))
    mask = a / 255.0

    h, w = overlay_rgb.shape[:2]
    roi = background[y:y+h, x:x+w]

    for c in range(3):
        roi[:, :, c] = roi[:, :, c] * (1 - mask) + overlay_rgb[:, :, c] * mask

    background[y:y+h, x:x+w] = roi
    return background

# Apply overlay
output_img = overlay_transparent(face_img.copy(), whiskers_resized, x_offset, y_offset)

# Save and display result
result_path = "/mnt/data/face_with_whiskers_dlib_simulated.png"
cv2.imwrite(result_path, output_img)

Image.fromarray(cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB))