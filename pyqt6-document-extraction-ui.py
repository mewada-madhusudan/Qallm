import sys
import os
import logging
import threading
from pathlib import Path

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QLabel, QLineEdit, QPushButton, QComboBox, QCheckBox, 
    QTextEdit, QTreeWidget, QTreeWidgetItem, QFileDialog, 
    QProgressBar, QFrame, QGroupBox, QMessageBox, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal, QObject

# Import the main extraction system
# Assuming the previous code is saved in extraction_system.py
from extraction_system import LocalLLMProcessor, DataProcessor, ResultsExporter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Custom logger that emits signals for the UI
class LogHandler(QObject, logging.Handler):
    new_log = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        logging.Handler.__init__(self)
        self.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        self.setFormatter(formatter)
    
    def emit(self, record):
        log_message = self.format(record)
        self.new_log.emit(log_message)

class DocumentExtractionUI(QMainWindow):
    """PyQt6 UI for the document extraction system"""
    
    progress_update = pyqtSignal(int)
    
    def __init__(self):
        super().__init__()
        
        # Set up the main window
        self.setWindowTitle("Intelligent Document Extraction System")
        self.setGeometry(100, 100, 1000, 800)
        
        # Set up central widget and main layout
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QVBoxLayout(self.central_widget)
        
        # Initialize UI components
        self.create_input_frame()
        self.create_model_selection_frame()
        self.create_log_frame()
        self.create_results_frame()
        self.create_button_frame()
        
        # State variables
        self.input_file_path = None
        self.output_file_path = None
        self.selected_model = "mistralai/Mistral-7B-Instruct-v0.2"
        self.processing_thread = None
        self.llm_processor = None
        self.data_processor = None
        
        # Set up signal connections
        self.progress_update.connect(self.update_progress)
        
        # Set up custom logging
        self.log_handler = LogHandler()
        self.log_handler.new_log.connect(self.append_log)
        logging.getLogger().addHandler(self.log_handler)
    
    def create_input_frame(self):
        """Create frame for input/output file selection"""
        input_group = QGroupBox("File Selection")
        input_layout = QVBoxLayout()
        input_group.setLayout(input_layout)
        
        # Input file selection
        input_file_layout = QHBoxLayout()
        input_file_label = QLabel("Input Excel File:")
        self.input_file_entry = QLineEdit()
        self.input_file_entry.setReadOnly(True)
        input_file_browse = QPushButton("Browse...")
        input_file_browse.clicked.connect(self.browse_input_file)
        
        input_file_layout.addWidget(input_file_label)
        input_file_layout.addWidget(self.input_file_entry)
        input_file_layout.addWidget(input_file_browse)
        
        # Output file selection
        output_file_layout = QHBoxLayout()
        output_file_label = QLabel("Output Excel File:")
        self.output_file_entry = QLineEdit()
        self.output_file_entry.setReadOnly(True)
        output_file_browse = QPushButton("Browse...")
        output_file_browse.clicked.connect(self.browse_output_file)
        
        output_file_layout.addWidget(output_file_label)
        output_file_layout.addWidget(self.output_file_entry)
        output_file_layout.addWidget(output_file_browse)
        
        # Add layouts to group
        input_layout.addLayout(input_file_layout)
        input_layout.addLayout(output_file_layout)
        
        # Add group to main layout
        self.main_layout.addWidget(input_group)
    
    def create_model_selection_frame(self):
        """Create frame for model selection"""
        model_group = QGroupBox("Model Selection")
        model_layout = QVBoxLayout()
        model_group.setLayout(model_layout)
        
        # Model selection dropdown
        model_selection_layout = QHBoxLayout()
        model_label = QLabel("LLM Model:")
        
        self.model_dropdown = QComboBox()
        models = [
            "mistralai/Mistral-7B-Instruct-v0.2",
            "meta-llama/Llama-2-7b-chat-hf",
            "THUDM/chatglm2-6b",
            "mosaicml/mpt-7b-instruct"
        ]
        self.model_dropdown.addItems(models)
        self.model_dropdown.setCurrentText(self.selected_model)
        self.model_dropdown.currentTextChanged.connect(self.update_selected_model)
        
        model_selection_layout.addWidget(model_label)
        model_selection_layout.addWidget(self.model_dropdown)
        model_selection_layout.addStretch()
        
        # Advanced options
        advanced_layout = QHBoxLayout()
        self.use_8bit_checkbox = QCheckBox("Use 8-bit Quantization (reduces memory usage)")
        self.use_8bit_checkbox.setChecked(True)
        advanced_layout.addWidget(self.use_8bit_checkbox)
        advanced_layout.addStretch()
        
        # Add layouts to group
        model_layout.addLayout(model_selection_layout)
        model_layout.addLayout(advanced_layout)
        
        # Add group to main layout
        self.main_layout.addWidget(model_group)
    
    def create_log_frame(self):
        """Create frame for logging output"""
        log_group = QGroupBox("Log")
        log_layout = QVBoxLayout()
        log_group.setLayout(log_layout)
        
        # Log text area
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        
        log_layout.addWidget(self.log_text)
        
        # Add group to main layout
        self.main_layout.addWidget(log_group)
    
    def create_results_frame(self):
        """Create frame for displaying results"""
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout()
        results_group.setLayout(results_layout)
        
        # Create treeview for results
        self.results_tree = QTreeWidget()
        self.results_tree.setColumnCount(5)
        self.results_tree.setHeaderLabels(["ID", "Element", "Document", "Extracted Value", "Confidence"])
        
        # Set column widths
        self.results_tree.setColumnWidth(0, 50)
        self.results_tree.setColumnWidth(1, 150)
        self.results_tree.setColumnWidth(2, 250)
        self.results_tree.setColumnWidth(3, 200)
        self.results_tree.setColumnWidth(4, 100)
        
        results_layout.addWidget(self.results_tree)
        
        # Add group to main layout
        self.main_layout.addWidget(results_group)
    
    def create_button_frame(self):
        """Create frame for action buttons"""
        button_layout = QVBoxLayout()
        
        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        button_layout.addWidget(self.progress_bar)
        
        # Buttons layout
        buttons_row = QHBoxLayout()
        
        # Action buttons
        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self.start_processing)
        
        self.stop_button = QPushButton("Stop")
        self.stop_button.clicked.connect(self.stop_processing)
        self.stop_button.setEnabled(False)
        
        self.clear_button = QPushButton("Clear Results")
        self.clear_button.clicked.connect(self.clear_results)
        
        self.exit_button = QPushButton("Exit")
        self.exit_button.clicked.connect(self.close)
        
        # Add buttons to layout
        buttons_row.addWidget(self.start_button)
        buttons_row.addWidget(self.stop_button)
        buttons_row.addWidget(self.clear_button)
        buttons_row.addStretch()
        buttons_row.addWidget(self.exit_button)
        
        button_layout.addLayout(buttons_row)
        
        # Add button layout to main layout
        self.main_layout.addLayout(button_layout)
    
    def browse_input_file(self):
        """Open file dialog to select input Excel file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Input Excel File",
            "",
            "Excel files (*.xlsx *.xls);;All files (*.*)"
        )
        
        if file_path:
            self.input_file_path = file_path
            self.input_file_entry.setText(file_path)
            
            # Auto-suggest output path
            if not self.output_file_entry.text():
                output_dir = os.path.dirname(file_path)
                input_name = os.path.splitext(os.path.basename(file_path))[0]
                output_path = os.path.join(output_dir, f"{input_name}_results.xlsx")
                
                self.output_file_path = output_path
                self.output_file_entry.setText(output_path)
    
    def browse_output_file(self):
        """Open file dialog to select output Excel file"""
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Select Output Excel File",
            "",
            "Excel files (*.xlsx);;All files (*.*)"
        )
        
        if file_path:
            # Ensure file has .xlsx extension
            if not file_path.lower().endswith('.xlsx'):
                file_path += '.xlsx'
                
            self.output_file_path = file_path
            self.output_file_entry.setText(file_path)
    
    def update_selected_model(self, model_name):
        """Update the selected model when dropdown selection changes"""
        self.selected_model = model_name
    
    def update_progress(self, value):
        """Update progress bar value"""
        self.progress_bar.setValue(value)
    
    def append_log(self, message):
        """Append message to log text box"""
        self.log_text.append(message)
        # Auto-scroll to bottom
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
    
    def update_results_tree(self, results):
        """Update results tree with extraction results"""
        # Clear existing results
        self.results_tree.clear()
        
        # Add new results
        try:
            row_id = 1
            for element_result in results.get("results", []):
                element_name = element_result.get("element_name", "")
                
                for doc_result in element_result.get("document_results", []):
                    doc_path = os.path.basename(doc_result.get("document_path", ""))
                    
                    item = QTreeWidgetItem()
                    item.setText(0, str(row_id))  # ID
                    item.setText(1, element_name)  # Element
                    item.setText(2, doc_path)  # Document
                    
                    if doc_result.get("success", False) and "extraction" in doc_result:
                        extraction = doc_result["extraction"]
                        value = extraction.get("value", "N/A")
                        found = extraction.get("found", False)
                        confidence = extraction.get("confidence", "N/A") if "confidence" in extraction else "N/A"
                        
                        item.setText(3, str(value))  # Extracted Value
                        item.setText(4, str(confidence))  # Confidence
                    else:
                        error = doc_result.get("error", "Extraction failed")
                        item.setText(3, "ERROR")  # Extracted Value
                        item.setText(4, "N/A")  # Confidence
                    
                    self.results_tree.addTopLevelItem(item)
                    row_id += 1
        except Exception as e:
            logger.error(f"Error updating results tree: {str(e)}")
    
    def start_processing(self):
        """Start the document processing in a separate thread"""
        # Validate inputs
        if not self.input_file_path or not os.path.exists(self.input_file_path):
            logging.error("Input file not found or not specified")
            QMessageBox.critical(self, "Input Error", "Please select a valid input Excel file")
            return
            
        if not self.output_file_path:
            logging.error("Output file not specified")
            QMessageBox.critical(self, "Output Error", "Please specify an output Excel file")
            return
        
        # Update UI state
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.progress_bar.setValue(0)
        
        # Clear previous results
        self.clear_results()
        
        # Start processing in a separate thread
        self.processing_thread = threading.Thread(target=self.process_data)
        self.processing_thread.daemon = True
        self.processing_thread.start()
    
    def process_data(self):
        """Process the input data - runs in a separate thread"""
        try:
            # Initialize LLM processor
            model_name = self.selected_model
            logging.info(f"Initializing LLM processor with model: {model_name}")
            
            self.progress_update.emit(10)
            use_8bit = self.use_8bit_checkbox.isChecked()
            self.llm_processor = LocalLLMProcessor(model_name=model_name, use_8bit=use_8bit)
            
            # Initialize data processor
            logging.info("Initializing data processor")
            self.progress_update.emit(20)
            self.data_processor = DataProcessor(self.llm_processor)
            
            # Process Excel file
            logging.info(f"Processing Excel file: {self.input_file_path}")
            self.progress_update.emit(30)
            results = self.data_processor.process_excel(self.input_file_path)
            
            if not results["success"]:
                error_msg = results.get('error', 'Unknown error')
                logging.error(f"Failed to process Excel file: {error_msg}")
                # Using signals to update UI from thread
                QApplication.instance().processEvents()
                QMessageBox.critical(self, "Processing Error", f"Failed to process Excel file: {error_msg}")
                self.progress_update.emit(0)
                self.start_button.setEnabled(True)
                self.stop_button.setEnabled(False)
                return
            
            # Update progress
            self.progress_update.emit(80)
            
            # Update results tree (use signals to update UI from thread)
            QApplication.instance().processEvents()
            self.update_results_tree(results)
            
            # Export results
            logging.info(f"Exporting results to: {self.output_file_path}")
            export_success = ResultsExporter.export_to_excel(results, self.output_file_path)
            
            if not export_success:
                logging.error("Failed to export results")
                QMessageBox.critical(self, "Export Error", "Failed to export results to Excel file")
            else:
                logging.info("Processing completed successfully")
                QMessageBox.information(self, "Success", 
                                      f"Processing completed successfully. Results exported to {self.output_file_path}")
            
            # Update progress
            self.progress_update.emit(100)
            
        except Exception as e:
            logging.error(f"Error in processing thread: {str(e)}")
            QApplication.instance().processEvents()
            QMessageBox.critical(self, "Processing Error", f"An error occurred: {str(e)}")
        
        finally:
            # Reset UI state using signals
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
    
    def stop_processing(self):
        """Stop the processing thread"""
        if self.processing_thread and self.processing_thread.is_alive():
            logging.info("Processing stopped by user")
            
            # Disable stop button
            self.stop_button.setEnabled(False)
            
            # Show message to user
            QMessageBox.information(self, "Processing Stopped", 
                                  "Processing stop requested. The current operation will terminate after the current document completes.")
    
    def clear_results(self):
        """Clear results from the tree widget"""
        self.results_tree.clear()
        
        # Reset progress bar
        self.progress_bar.setValue(0)


# Main entry point
if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application style (optional)
    # app.setStyle("Fusion")  # More consistent cross-platform look
    
    window = DocumentExtractionUI()
    window.show()
    
    sys.exit(app.exec())
