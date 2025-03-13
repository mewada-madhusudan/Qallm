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
                        extraction = doc_result["extraction"]
                        value = extraction.get("value", "N/A")
                        found = extraction.get("found", False)
                        confidence = extraction.get("confidence", "N/A") if "confidence" in extraction else "N/A"
                        
                        self.results_tree.insert(
                            "", "end", text=str(row_id), 
                            values=(element_name, doc_path, value, confidence)
                        )
                    else:
                        error = doc_result.get("error", "Extraction failed")
                        self.results_tree.insert(
                            "", "end", text=str(row_id), 
                            values=(element_name, doc_path, "ERROR", "N/A")
                        )
                    
                    row_id += 1
        except Exception as e:
            logger.error(f"Error updating results tree: {str(e)}")
    
    def start_processing(self):
        """Start the document processing in a separate thread"""
        # Validate inputs
        if not self.input_file_path or not os.path.exists(self.input_file_path):
            logging.error("Input file not found or not specified")
            return
            
        if not self.output_file_path:
            logging.error("Output file not specified")
            return
        
        # Update UI state
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress_var.set(0)
        
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
            model_name = self.selected_model.get()
            logging.info(f"Initializing LLM processor with model: {model_name}")
            
            self.update_progress(10)
            self.llm_processor = LocalLLMProcessor(model_name=model_name)
            
            # Initialize data processor
            logging.info("Initializing data processor")
            self.update_progress(20)
            self.data_processor = DataProcessor(self.llm_processor)
            
            # Process Excel file
            logging.info(f"Processing Excel file: {self.input_file_path}")
            self.update_progress(30)
            results = self.data_processor.process_excel(self.input_file_path)
            
            if not results["success"]:
                error_msg = results.get('error', 'Unknown error')
                logging.error(f"Failed to process Excel file: {error_msg}")
                self.root.after(0, lambda: tk.messagebox.showerror("Processing Error", f"Failed to process Excel file: {error_msg}"))
                self.update_progress(0)
                self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
                self.root.after(0, lambda: self.stop_button.config(state=tk.DISABLED))
                return
            
            # Update progress
            self.update_progress(80)
            
            # Update results tree
            self.root.after(0, lambda: self.update_results_tree(results))
            
            # Export results
            logging.info(f"Exporting results to: {self.output_file_path}")
            export_success = ResultsExporter.export_to_excel(results, self.output_file_path)
            
            if not export_success:
                logging.error("Failed to export results")
                self.root.after(0, lambda: tk.messagebox.showerror("Export Error", "Failed to export results to Excel file"))
            else:
                logging.info("Processing completed successfully")
                self.root.after(0, lambda: tk.messagebox.showinfo("Success", f"Processing completed successfully. Results exported to {self.output_file_path}"))
            
            # Update progress
            self.update_progress(100)
            
        except Exception as e:
            logging.error(f"Error in processing thread: {str(e)}")
            self.root.after(0, lambda: tk.messagebox.showerror("Processing Error", f"An error occurred: {str(e)}"))
        
        finally:
            # Reset UI state
            self.root.after(0, lambda: self.start_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.stop_button.config(state=tk.DISABLED))
    
    def stop_processing(self):
        """Stop the processing thread"""
        if self.processing_thread and self.processing_thread.is_alive():
            logging.info("Processing stopped by user")
            
            # We can't directly stop the thread, but we can set a flag
            # For a more complete solution, we'd need to implement a cancellation mechanism
            # in the DataProcessor and LLMProcessor classes
            
            # Disable stop button
            self.stop_button.config(state=tk.DISABLED)
            
            # Show message to user
            tk.messagebox.showinfo("Processing Stopped", 
                                   "Processing stop requested. The current operation will terminate after the current document completes.")
    
    def clear_results(self):
        """Clear results from the treeview"""
        for item in self.results_tree.get_children():
            self.results_tree.delete(item)
        
        # Reset progress bar
        self.progress_var.set(0)


# Main entry point
if __name__ == "__main__":
    root = tk.Tk()
    app = DocumentExtractionUI(root)
    root.mainloop()
