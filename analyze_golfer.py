import base64
import json
import os
import requests

API_URL = "http://localhost:11435/api/generate"
MODEL = "llava:latest"

IMAGE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "AT.jpg")

PROMPT = (
    "You are an expert sports image analyst. "
    "Look at this photo from a professional women's golf tournament. "
    "Identify the female pro golfer who appears LARGEST in the frame "
    "(ignore caddies, spectators, and any other people). "
    "Describe ONLY her physical characteristics EXCLUDING her face. "
    "Be as detailed as possible about ALL of the following:\n"
    "1. HAIR: style, length, color, any hair accessories (visor, cap, headband, etc.)\n"
    "2. TOP: type of shirt/polo/jacket, color(s), pattern, collar style, sleeve length\n"
    "3. LOGOS/BRANDING: any visible logos, sponsor names, or brand marks on her clothing, "
    "hat, or accessories — describe their location, color, and text if readable\n"
    "4. BOTTOM: type of pants/skirt/shorts, color, length, fit\n"
    "5. BELT: color, style, buckle details if visible\n"
    "6. SHOES: type, color, brand if visible\n"
    "7. GLOVE: which hand, color\n"
    "8. ACCESSORIES: watch, bracelet, necklace, sunglasses (on head or hanging), "
    "earrings, or any other visible accessories\n"
    "9. BUILD/POSTURE: approximate height impression, body build, stance\n"
    "10. ANY OTHER distinguishing non-facial physical features\n\n"
    "Do NOT describe her face or facial features. "
    "Provide your answer in English."
)


def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def analyze_image(image_path: str) -> str:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image_b64 = encode_image(image_path)

    payload = {
        "model": MODEL,
        "prompt": PROMPT,
        "images": [image_b64],
        "stream": False,
    }

    print(f"Sending image '{os.path.basename(image_path)}' to {MODEL}...")
    print(f"API endpoint: {API_URL}")
    print("-" * 60)

    response = requests.post(API_URL, json=payload, timeout=300)
    response.raise_for_status()

    result = response.json()
    return result.get("response", "No response received from model.")


def main():
    try:
        description = analyze_image(IMAGE_FILE)
        print("\n=== Female Pro Golfer Physical Description (Excluding Face) ===\n")
        print(description)
        print("\n" + "=" * 60)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Please place 'AT.jpg' in the same directory as this script: "
              f"{os.path.dirname(os.path.abspath(__file__))}")
    except requests.exceptions.ConnectionError:
        print(f"Error: Could not connect to the LLM server at {API_URL}")
        print("Please make sure the Ollama server is running on port 11435.")
    except requests.exceptions.Timeout:
        print("Error: Request timed out. The model may be taking too long to respond.")
    except requests.exceptions.HTTPError as e:
        print(f"Error: HTTP error from LLM server: {e}")
    except json.JSONDecodeError:
        print("Error: Invalid JSON response from the LLM server.")


if __name__ == "__main__":
    main()
