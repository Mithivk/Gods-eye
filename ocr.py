import cv2
import pytesseract
from gtts import gTTS
from playsound import playsound
import os

# Set the tesseract_cmd path if it's not in the system PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Function to play audio
def play_audio(text):
    tts = gTTS(text=text, lang='en')
    audio_file = "audio.mp3"
    tts.save(audio_file)
    playsound(audio_file)
    os.remove(audio_file)

# Function to load and process image from file
def load_and_process_image(image_path):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file {image_path} not found.")

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray

# Function to perform OCR on the processed image
def perform_ocr(image):
    # Use Tesseract to do OCR on the image
    text = pytesseract.image_to_string(image)
    return text

# Main function
def main(image_path):
    print("Loading and processing image...")
    processed_image = load_and_process_image(image_path)

    print("Performing OCR on the image...")
    extracted_text = perform_ocr(processed_image)
    print("Extracted Text:\n", extracted_text)

    if extracted_text.strip():
        print("Converting text to speech...")
        play_audio(extracted_text)
    else:
        print("No text found in the image.")

if __name__ == "__main__":
    # Path to the image file
    image_path = 'Screenshot 2024-07-17 185903.png'
    main(image_path)
