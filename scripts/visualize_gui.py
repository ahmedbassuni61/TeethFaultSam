import os
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import trimesh
import numpy as np
import plotly.graph_objects as go
from pathlib import Path

class SyntheticDataViewer(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("🦷 3DTeethSAM - Synthetic Faults Viewer")
        self.geometry("900x600")
        
        self.data_files = []
        
        self.create_widgets()
        
    def create_widgets(self):
        # Top frame for folder selection
        top_frame = tk.Frame(self)
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.dir_var = tk.StringVar(value="synthetic_data")
        tk.Label(top_frame, text="Dataset Root:", font=("Arial", 10, "bold")).pack(side=tk.LEFT)
        tk.Entry(top_frame, textvariable=self.dir_var, width=60, font=("Arial", 10)).pack(side=tk.LEFT, padx=10)
        tk.Button(top_frame, text="Browse", command=self.browse_dir).pack(side=tk.LEFT, padx=5)
        tk.Button(top_frame, text="Load Data", command=self.load_data, bg="#2196F3", fg="white").pack(side=tk.LEFT, padx=5)
        
        # Treeview for displaying available scans
        tree_frame = tk.Frame(self)
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        columns = ("fault_type", "jaw", "patient_id", "obj_file", "json_file")
        self.tree = ttk.Treeview(tree_frame, columns=columns, show="headings")
        self.tree.heading("fault_type", text="Fault Type")
        self.tree.heading("jaw", text="Jaw")
        self.tree.heading("patient_id", text="Patient ID")
        self.tree.heading("obj_file", text="OBJ Path")
        self.tree.heading("json_file", text="JSON Path")
        
        self.tree.column("fault_type", width=100)
        self.tree.column("jaw", width=80)
        self.tree.column("patient_id", width=120)
        self.tree.column("obj_file", width=250)
        self.tree.column("json_file", width=250)
        
        scrollbar = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Bottom frame for action buttons
        bottom_frame = tk.Frame(self)
        bottom_frame.pack(fill=tk.X, padx=10, pady=10)
        
        tk.Button(
            bottom_frame, 
            text="👁️ Visualize Selected Mesh (Plotly)", 
            command=self.visualize_selected, 
            bg="#4CAF50", 
            fg="white", 
            font=("Arial", 12, "bold")
        ).pack(pady=5)
        
        tk.Label(
            bottom_frame, 
            text="Note: Visualizations open instantly in your default web browser for high-performance 3D rendering.", 
            fg="gray"
        ).pack()

    def browse_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            self.dir_var.set(directory)
            
    def load_data(self):
        root_dir = Path(self.dir_var.get())
        if not root_dir.exists():
            messagebox.showerror("Error", f"Directory '{root_dir}' does not exist.")
            return
            
        self.tree.delete(*self.tree.get_children())
        self.data_files = []
        
        # We assume structure: <fault_type>/<jaw>/<patient_id>/<obj_and_json>
        # but we also support flat or other structures as long as there is an obj and json pair
        for obj_file in root_dir.rglob("*.obj"):
            # Exclude keypoints or alternative jsons, look for matching name or fallback
            base_name = obj_file.stem
            
            # Look for exact match json or first available non-kpt json
            expected_json = obj_file.parent / f"{base_name}.json"
            if expected_json.exists():
                json_file = expected_json
            else:
                jsons = [f for f in obj_file.parent.glob("*.json") if "__kpt" not in f.name]
                if not jsons:
                    continue
                json_file = jsons[0]
                
            patient_id = obj_file.parent.name
            jaw = 'lower' if 'lower' in obj_file.name.lower() else 'upper'
            
            # Try to guess fault type from path if it follows our script's convention
            parts = obj_file.parts
            fault_type = "unknown"
            if len(parts) >= 4:
                # E.g. synthetic_data / cavity / lower / patient_id
                fault_type = parts[-4]
                
            item_id = self.tree.insert("", tk.END, values=(fault_type.capitalize(), jaw.capitalize(), patient_id, str(obj_file), str(json_file)))
            self.data_files.append({
                'id': item_id,
                'obj': str(obj_file),
                'json': str(json_file),
                'title': f"{patient_id} - {jaw.capitalize()} ({fault_type.capitalize()})"
            })
            
        if not self.data_files:
            messagebox.showinfo("Info", f"No matching data pairs (.obj and .json) found in {root_dir}")
            
    def visualize_selected(self):
        selected_items = self.tree.selection()
        if not selected_items:
            messagebox.showwarning("Warning", "Please select an item from the list first.")
            return
            
        item_id = selected_items[0]
        data = next((x for x in self.data_files if x['id'] == item_id), None)
        if not data:
            return
            
        # Run visualization
        self.plot_mesh(data['obj'], data['json'], data['title'])
        
    def plot_mesh(self, obj_path, json_path, title):
        try:
            print(f"Loading {obj_path}...")
            mesh = trimesh.load(obj_path, process=False)
            
            with open(json_path, 'r') as f:
                label_data = json.load(f)
                
            labels = np.array(label_data.get('labels', np.zeros(len(mesh.vertices))))
            fault_labels = np.array(label_data.get('fault_labels', np.zeros(len(mesh.vertices))))
            
            # Initialize colors (default white)
            colors = np.ones((len(mesh.vertices), 3)) * 255
            
            # 1. Healthy Gum (Pink)
            colors[labels == 0] = [230, 150, 160]
            
            # 2. Healthy Teeth (Light Gray)
            colors[labels > 0] = [240, 240, 240]
            
            # 3. Fault Regions (Bright Red)
            colors[fault_labels == 1] = [255, 30, 30]
            
            vertex_colors = [f'rgb({int(c[0])}, {int(c[1])}, {int(c[2])})' for c in colors]
            
            print("Rendering visualization...")
            # Create Plotly figure
            fig = go.Figure(data=[
                go.Mesh3d(
                    x=mesh.vertices[:, 0], 
                    y=mesh.vertices[:, 1], 
                    z=mesh.vertices[:, 2],
                    i=mesh.faces[:, 0], 
                    j=mesh.faces[:, 1], 
                    k=mesh.faces[:, 2],
                    vertexcolor=vertex_colors, 
                    showscale=False,
                    lighting=dict(ambient=0.5, diffuse=0.8, specular=0.1, roughness=0.6)
                )
            ])
            
            fig.update_layout(
                title=dict(text=title, font=dict(size=24)), 
                scene=dict(
                    xaxis=dict(visible=False), 
                    yaxis=dict(visible=False), 
                    zaxis=dict(visible=False), 
                    aspectmode='data'
                ),
                margin=dict(l=0, r=0, b=0, t=50)
            )
            
            # Opens in default browser instantly
            fig.show()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to visualize mesh:\n{str(e)}")

if __name__ == "__main__":
    app = SyntheticDataViewer()
    app.mainloop()
