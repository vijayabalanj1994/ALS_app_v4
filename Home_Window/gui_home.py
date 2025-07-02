import sys
from PIL import Image

from PySide6 import QtCore as qtc
from PySide6 import QtWidgets as qtw
from PySide6.QtWidgets import QFileDialog

from Home_Window.UI.home_window import Ui_MainWindow
from Home_Window.utils.pil2pixmap import pil2pixmap
from ALS_Diagnostic_Model.src.utils.predict_image import predict_image
from ALS_Diagnostic_Model.src.utils.load_model import load_model
from ALS_Diagnostic_Model.src.utils.gradcam import extract_output_cam_and_image, generate_gradcam_overlay, generate_gradcam_circles, generate_gradcam_focus_mask, extract_color_masks, quantify_focus
from Dataset.utils import get_metadata

class GuiHome(qtw.QMainWindow, Ui_MainWindow):

    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.input_image_path = ""
        self.q_path_image_path = r"C:\Users\vijay\Neuro_BioMark\ALS_app_v4\Dataset\ALS_QuPath_Images"
        self.q_path_img = None
        self.a_browse_image.triggered.connect(self.browse_image)

        self.cam, self.reconstructed_image = None, None

        #self.a_run_analysis.triggered.connect(self.run_analysis)
        self.s_focus_slider.valueChanged.connect(self.update_cam_mask)
        self.le_focus_toggle.textChanged.connect(self.update_slider)
        #self.a_clear.triggered.connect(self.clear_GUI)

    @qtc.Slot()
    def browse_image(self):

        print("called.")

        self.clear_GUI()

        self.input_image_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select an Image",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif)"
        )

        if self.input_image_path:
            try:
                pil_image = Image.open(self.input_image_path)
                pixmap = pil2pixmap(pil_image)
                self.lb_input_image.setPixmap(
                    pixmap.scaled(
                        self.lb_input_image.size()
                    )
                )

                image_no, case_id, region = get_metadata(self.input_image_path)
                self.lb_image_no.setText(image_no)
                self.lb_case_id.setText(case_id)
                self.lb_region.setText(region)

                self.q_path_image_path = self.q_path_image_path + rf"\{int(image_no)}.tif"
                self.q_path_img = Image.open(self.q_path_image_path)
                pixmap = pil2pixmap(self.q_path_img)
                self.lb_qu_path_image.setPixmap(
                    pixmap.scaled(
                        self.lb_input_image.size()
                    )
                )

                self.run_analysis()

            except Exception as e:
                print(e)
                self.lb_input_image.setText("Unable to load image.")


    def update_progress_bar(self, results:list):

        class_labels = ["control", "concordant", "discordant"]

        for label, value in zip(class_labels, results):
            bar = getattr(self, f"p_bar_{label}")
            bar.setValue(int(value*100))

        ranked = sorted(
            zip(class_labels, results),
            key=lambda x: x[1],
            reverse=True  # Highest confidence first
        )

        rank_colors = ["#4CAF50", "#FFA726", "#FFEB3B"]  # Green, Orange, Yellow

        for i, (label, _) in enumerate(ranked):
            bar = getattr(self, f"p_bar_{label}")
            color = rank_colors[i]
            bar.setStyleSheet(f"""
                QProgressBar {{
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    text-align: center;
                }}
                QProgressBar::chunk {{
                    background-color: {color};
                    width: 10px;
                }}
            """)

    @qtc.Slot()
    def run_analysis(self):

        weights_path = r"C:\Users\vijay\Neuro_BioMark\ALS_app_v1\ALS_Diagnostic_Model\src\model\weights.pth"
        model = load_model(weights_path)

        predicted_class, probs, image_tensor, class_idx = predict_image(model, self.input_image_path)

        self.update_progress_bar(probs)

        self.lb_prediction.setText(predicted_class)

        self.model_raw_output, self.cam, self.reconstructed_image = extract_output_cam_and_image(model, image_tensor, class_idx)
        cam_image = generate_gradcam_overlay(self.cam, self.reconstructed_image)
        pixmap = pil2pixmap(cam_image)
        self.lb_cam_image.setPixmap(
            pixmap.scaled(
                self.lb_cam_image.size()
            )
        )

        self.s_focus_slider.setValue(80)
        filtered_image, circles = generate_gradcam_circles(self.cam, self.q_path_img.copy(), threshold=0.8)
        pixmap = pil2pixmap(filtered_image)
        self.lb_cam_circle_image.setPixmap(
            pixmap.scaled(
                self.lb_cam_circle_image.size()
            )
        )

        cells_mask = extract_color_masks(self.q_path_img)["yellow"]
        hits, misses, precision = quantify_focus(circles, cells_mask)
        self.le_areas_focused.setText(str(hits+misses))
        self.le_ROIs_focused.setText(str(hits))
        self.le_focus_relevance.setText(f"{precision:.2f}")

        self.s_focus_slider.setValue(80)
        masked_image = generate_gradcam_focus_mask(self.cam, self.reconstructed_image.copy(), threshold=0.8)
        pixmap = pil2pixmap(masked_image)
        self.lb_selected_focus_image.setPixmap(
            pixmap.scaled(
                self.lb_selected_focus_image.size()
            )
        )

    @qtc.Slot()
    def update_slider(self):

        value = int(self.le_focus_toggle.text())
        self.s_focus_slider.setValue(value)

    @qtc.Slot()
    def update_cam_mask(self):

        self.lb_cam_circle_image.clear()
        self.lb_selected_focus_image.clear()
        self.le_focus_toggle.setText(f"{self.s_focus_slider.value()}")
        threshold = self.s_focus_slider.value()/100


        filtered_image, circles = generate_gradcam_circles(self.cam, self.q_path_img.copy(), threshold=threshold)
        pixmap = pil2pixmap(filtered_image)
        self.lb_cam_circle_image.setPixmap(
            pixmap.scaled(
                self.lb_cam_circle_image.size()
            )
        )

        qpath_img = Image.open(self.q_path_image_path)
        cells_mask = extract_color_masks(qpath_img)["yellow"]
        hits, misses, precision = quantify_focus(circles, cells_mask)
        self.le_areas_focused.setText(str(hits+misses))
        self.le_ROIs_focused.setText(str(hits))
        self.le_focus_relevance.setText(f"{precision:.2f}")

        masked_image = generate_gradcam_focus_mask(self.cam, self.reconstructed_image.copy(), threshold=threshold)
        pixmap = pil2pixmap(masked_image)
        self.lb_selected_focus_image.setPixmap(
            pixmap.scaled(
                self.lb_selected_focus_image.size()
            )
        )

    @qtc.Slot()
    def clear_GUI(self):
        self.input_image_path = ""
        self.q_path_image_path = r"C:\Users\vijay\Neuro_BioMark\ALS_app_v4\Dataset\ALS_QuPath_Images"
        self.qpath_img = None
        self.lb_input_image.setText("Upload Image")
        self.lb_cam_image.setText("Waiting for model to run...")
        self.lb_cam_circle_image.setText("Waiting for model to run...")
        self.lb_selected_focus_image.setText("Waiting for model to run...")
        self.p_bar_control.setValue(0)
        self.p_bar_concordant.setValue(0)
        self.p_bar_discordant.setValue(0)
        self.lb_prediction.setText("Waiting...")

        self.lb_image_no.setText("")
        self.lb_case_id.setText("")
        self.lb_region.setText("")



if __name__ == "__main__":
    app = qtw.QApplication(sys.argv)
    window = GuiHome()
    window.show()
    sys.exit(app.exec())