import tkinter as tk
from tkinter import filedialog, ttk, scrolledtext
import threading
import os
import sys
import logging
from pathlib import Path
import pandas as pd

# Import the main extraction system
# Assuming the previous code is saved in extraction_system.py
from extraction_system import LocalLLMProcessor, DataProcessor, ResultsExporter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentExtractionUI:
    """Simple Tkinter UI for the document extraction system"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("Intelligent Document Extraction System")
        self.root.geometry("900x700")
        
        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Initialize UI components
        self.create_input_frame()
        self.create_model_selection_frame()
        self.create_log_frame()
        self.create_results_frame()
        self.create_button_frame()
        
        # State variables
        self.input_file_path = None
        self.output_file_path = None
        self.selected_model = tk.StringVar(value="mistralai/Mistral-7B-Instruct-v0.2")
        self.processing_thread = None
        self.llm_processor = None
        self.data_processor = None
        
    def create_input_frame(self):
        """Create frame for input/output file selection"""
        input_frame = ttk.LabelFrame(self.main_frame, text="File Selection", padding="10")
        input_frame.pack(fill=tk.X, pady=5)
        
        # Input file selection
        ttk.Label(input_frame, text="Input Excel File:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.input_file_entry = ttk.Entry(input_frame, width=50)
        self.input_file_entry.grid(row=0, column=1, padx=5, pady=5)
        ttk.Button(input_frame, text="Browse...", command=self.browse_input_file).grid(row=0, column=2, padx=5, pady=5)
        
        # Output file selection
        ttk.Label(input_frame, text="Output Excel File:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.output_file_entry = ttk.Entry(input_frame, width=50)
        self.output_file_entry.grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(input_frame, text="Browse...", command=self.browse_output_file).grid(row=1, column=2, padx=5, pady=5)
    
    def create_model_selection_frame(self):
        """Create frame for model selection"""
        model_frame = ttk.LabelFrame(self.main_frame, text="Model Selection", padding="10")
        model_frame.pack(fill=tk.X, pady=5)
        
        # Model selection dropdown
        ttk.Label(model_frame, text="LLM Model:").grid(row=0, column=0, sticky=tk.W, pady=5)
        
        models = [
            "mistralai/Mistral-7B-Instruct-v0.2",
            "meta-llama/Llama-2-7b-chat-hf",
            "THUDM/chatglm2-6b",
            "mosaicml/mpt-7b-instruct"
        ]
        
        model_dropdown = ttk.Combobox(model_frame, textvariable=self.selected_model, values=models, width=48)
        model_dropdown.grid(row=0, column=1, padx=5, pady=5, sticky=tk.W)
        
        # Advanced options
        advanced_frame = ttk.Frame(model_frame)
        advanced_frame.grid(row=1, column=0, columnspan=3, sticky=tk.W, pady=5)
        
        self.use_8bit = tk.BooleanVar(value=True)
        ttk.Checkbutton(advanced_frame, text="Use 8-bit Quantization (reduces memory usage)", 
                        variable=self.use_8bit).pack(side=tk.LEFT, padx=5)
    
    def create_log_frame(self):
        """Create frame for logging output"""
        log_frame = ttk.LabelFrame(self.main_frame, text="Log", padding="10")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Log text area
        self.log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.config(state=tk.DISABLED)
        
        # Custom logging handler to redirect logs to the text widget
        class TextHandler(logging.Handler):
            def __init__(self, text_widget):
                logging.Handler.__init__(self)
                self.text_widget = text_widget
                
            def emit(self, record):
                msg = self.format(record)
                
                def append():
                    self.text_widget.config(state=tk.NORMAL)
                    self.text_widget.insert(tk.END, msg + '\n')
                    self.text_widget.see(tk.END)
                    self.text_widget.config(state=tk.DISABLED)
                    
                self.text_widget.after(0, append)
        
        # Add the handler to the logger
        text_handler = TextHandler(self.log_text)
        text_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        text_handler.setFormatter(formatter)
        logging.getLogger().addHandler(text_handler)
    
    def create_results_frame(self):
        """Create frame for displaying results"""
        results_frame = ttk.LabelFrame(self.main_frame, text="Results", padding="10")
        results_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create treeview for results
        self.results_tree = ttk.Treeview(results_frame, columns=("element", "document", "value", "confidence"))
        self.results_tree.heading("#0", text="ID")
        self.results_tree.heading("element", text="Element")
        self.results_tree.heading("document", text="Document")
        self.results_tree.heading("value", text="Extracted Value")
        self.results_tree.heading("confidence", text="Confidence")
        
        self.results_tree.column("#0", width=50)
        self.results_tree.column("element", width=150)
        self.results_tree.column("document", width=250)
        self.results_tree.column("value", width=200)
        self.results_tree.column("confidence", width=100)
        
        # Add scrollbar
        scrollbar = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.results_tree.yview)
        self.results_tree.configure(yscrollcommand=scrollbar.set)
        
        # Pack components
        self.results_tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    
    def create_button_frame(self):
        """Create frame for action buttons"""
        button_frame = ttk.Frame(self.main_frame, padding="10")
        button_frame.pack(fill=tk.X, pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(button_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        # Buttons
        self.start_button = ttk.Button(button_frame, text="Start Processing", command=self.start_processing)
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_processing, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Clear Results", command=self.clear_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Exit", command=self.root.destroy).pack(side=tk.RIGHT, padx=5)
    
    def browse_input_file(self):
        """Open file dialog to select input Excel file"""
        file_path = filedialog.askopenfilename(
            title="Select Input Excel File",
            filetypes=[("Excel files", "*.xlsx *.xls"), ("All files", "*.*")]
        )
        
        if file_path:
            self.input_file_path = file_path
            self.input_file_entry.delete(0, tk.END)
            self.input_file_entry.insert(0, file_path)
            
            # Auto-suggest output path
            if not self.output_file_entry.get():
                output_dir = os.path.dirname(file_path)
                input_name = os.path.splitext(os.path.basename(file_path))[0]
                output_path = os.path.join(output_dir, f"{input_name}_results.xlsx")
                
                self.output_file_path = output_path
                self.output_file_entry.delete(0, tk.END)
                self.output_file_entry.insert(0, output_path)
    
    def browse_output_file(self):
        """Open file dialog to select output Excel file"""
        file_path = filedialog.asksaveasfilename(
            title="Select Output Excel File",
            defaultextension=".xlsx",
            filetypes=[("Excel files", "*.xlsx"), ("All files", "*.*")]
        )
        
        if file_path:
            self.output_file_path = file_path
            self.output_file_entry.delete(0, tk.END)
            self.output_file_entry.insert(0, file_path)
    
    def update_progress(self, value):
        """Update progress bar value"""
        self.progress_var.set(value)
    
    def update_results_tree(self, results):
        """Update results treeview with extraction results"""
        # Clear existing results
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Add new results
        try:
            row_id = 1
            for element_result in results.get("results", []):
                element_name = element_result.get("element_name", "")
                
                for doc_result in element_result.get("document_results", []):
                    doc_path = os.path.basename(doc_result.get("document_path", ""))
                    
                    if doc_result.get("success", False) and "extraction" in doc_result:
                        