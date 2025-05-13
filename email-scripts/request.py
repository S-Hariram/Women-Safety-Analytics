import cv2
import datetime
import os
import smtplib
from geopy.geocoders import Nominatim
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders

# Step 1: Capture snapshot from the camera
camera = cv2.VideoCapture(0)
ret, image = camera.read()
image_path = 'snapshot.jpg'

if ret:
    cv2.imwrite(image_path, image)
    print("📸 Snapshot saved successfully!")
else:
    print("❌ Failed to capture image! Exiting...")
    exit()

camera.release()

# Step 2: Get current time
current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Step 3: Get location safely
geolocator = Nominatim(user_agent="my_geolocation_app")
try:
    location = geolocator.geocode("12A State Highway, CGC Jhanjeri, Mohali, Punjab, India", timeout=10)
    location_details = f"{location.address}, Lat: {location.latitude}, Lon: {location.longitude}" if location else "Location not found"
except Exception as e:
    location_details = f"Error retrieving location: {e}"

# Step 4: Record 10-second MP4 video
video_path = 'alert_video.mp4'
camera = cv2.VideoCapture(0)

if not camera.isOpened():
    print("❌ Failed to open the camera!")
    exit()

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
out = cv2.VideoWriter(video_path, fourcc, 20.0, (640, 480))

start_time = datetime.datetime.now()
while (datetime.datetime.now() - start_time).seconds < 10:
    ret, frame = camera.read()
    if ret:
        out.write(frame)
    else:
        print("⚠ Warning: Failed to capture frame, stopping recording...")
        break

camera.release()
out.release()
print("🎥 Video saved successfully!")

# Step 5: Get Email Credentials
from_email = input("Enter your email address: ")
password = input("Enter your App Password (not regular password): ")
to_email = input("Enter the recipient's email address: ")

# Step 6: Create Email
msg = MIMEMultipart()
msg['From'] = from_email
msg['To'] = to_email
msg['Subject'] = "🚨 Alert Message"

body = f"⚠ Alert!\n🕒 Time: {current_time}\n📍 Location: {location_details}"
msg.attach(MIMEText(body, 'plain'))

# Step 7: Attach snapshot
if os.path.exists(image_path):
    with open(image_path, "rb") as attachment:
        part = MIMEBase('application', 'octet-stream')
        part.set_payload(attachment.read())
        encoders.encode_base64(part)
        part.add_header('Content-Disposition', f"attachment; filename={os.path.basename(image_path)}")
        msg.attach(part)
else:
    print("❌ Image file missing, skipping attachment.")

# Step 8: Attach video
if os.path.exists(video_path):
    with open(video_path, "rb") as video_attachment:
        video_part = MIMEBase('application', 'octet-stream')
        video_part.set_payload(video_attachment.read())
        encoders.encode_base64(video_part)
        video_part.add_header('Content-Disposition', f"attachment; filename={os.path.basename(video_path)}")
        msg.attach(video_part)
else:
    print("❌ Video file missing, skipping attachment.")

# Step 9: Send Email
try:
    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(from_email, password)
    text = msg.as_string()
    server.sendmail(from_email, to_email, text)
    server.quit()
    print("✅ Email sent successfully!")
except smtplib.SMTPAuthenticationError:
    print("❌ Authentication failed. Use an App Password instead of your real password!")
except Exception as e:
    print(f"❌ An error occurred: {e}")

# Step 10: Clean up files
for file in [image_path, video_path]:
    try:
        if os.path.exists(file):
            os.remove(file)
            print(f"🗑 Deleted {file}")
    except PermissionError as e:
        print(f"⚠ Permission Error: {e}")
