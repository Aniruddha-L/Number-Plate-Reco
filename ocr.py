import os
import cv2
import pytesseract
import xml.etree.ElementTree as ET
from sklearn.metrics import classification_report, accuracy_score

# Set Tesseract path (Windows only, update if installed elsewhere)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Paths
xml_dir = r"D:\Projects\Project phase 1\ground truth annotes"
output_dir = r"D:\Projects\Project phase 1\output_images"
report_path = r"D:\Projects\Project phase 1\classification_report.txt"

# Create output folder
os.makedirs(output_dir, exist_ok=True)

# Store results
y_true = []
y_pred = []
files_checked = []

# Loop through all XML files
for xml_file in os.listdir(xml_dir):
    if xml_file.lower().endswith(".xml"):
        xml_path = os.path.join(xml_dir, xml_file)

        # Parse XML
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            # Get image path and filename
            img_path = root.find("path").text
            filename = root.find("filename").text

            # Ground truth label
            gt_label = root.find("object/name").text.strip()

            # Bounding box
            bbox = root.find("object/bndbox")
            xmin = int(bbox.find("xmin").text)
            ymin = int(bbox.find("ymin").text)
            xmax = int(bbox.find("xmax").text)
            ymax = int(bbox.find("ymax").text)

            # Load image
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Image not found for {xml_file}, skipping...")
                continue

            # Crop plate
            plate_crop = img[ymin:ymax, xmin:xmax]

            # OCR
            ocr_result = pytesseract.image_to_string(plate_crop, config="--psm 7").strip()

            # Normalize labels (remove spaces, dashes, uppercase)
            clean_gt = gt_label.replace(" ", "").replace("-", "").upper()
            clean_ocr = ocr_result.replace(" ", "").replace("-", "").upper()

            # Save results
            y_true.append(clean_gt)
            y_pred.append(clean_ocr)
            files_checked.append(filename)

            # Draw bounding box + OCR result
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 3)
            cv2.putText(img, f"OCR: {ocr_result}", (xmin, ymin - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            # Save annotated image
            output_path = os.path.join(output_dir, f"annotated_{filename}")
            cv2.imwrite(output_path, img)

        except Exception as e:
            print(f"❌ Error processing {xml_file}: {e}")
            continue

# Generate classification report
# Generate classification report
if y_true:
    report = classification_report(y_true, y_pred, zero_division=0)
    acc = accuracy_score(y_true, y_pred)

    total_images = len(files_checked)
    correct = sum(1 for gt, pr in zip(y_true, y_pred) if gt == pr)
    incorrect = total_images - correct

    with open(report_path, "w") as f:
        f.write("Number Plate OCR Evaluation\n")
        f.write("============================\n\n")
        f.write(f"Total Images Checked: {total_images}\n")
        f.write(f"Accuracy: {acc:.4f}\n\n")
        f.write("Detailed Classification Report:\n")
        f.write(report)
        f.write("\n\nSummary:\n")
        f.write(f"Total Images   : {total_images}\n")
        f.write(f"Correctly Classified   : {correct}\n")
        f.write(f"Incorrectly Classified : {incorrect}\n")

    print("✅ Classification report saved at:", report_path)
    print("✅ Annotated images saved in:", output_dir)
else:
    print("⚠️ No valid data processed!")
