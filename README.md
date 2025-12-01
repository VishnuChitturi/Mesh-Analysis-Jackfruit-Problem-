#  3D Cloth Simulation & Cutting – OpenGL + Python

This project is an **interactive real-time cloth simulation** using **Verlet integration physics** and **OpenGL rendering**.  
Users can **cut the fabric using mouse drawing**, apply **random/horizontal cuts**, and **orbit around the cloth in 3D**.

 **Built using Python, NumPy, and PyOpenGL.**

---

##  Features

✔️ Real-time cloth simulation using physics  
✔️ Mouse-based cutting interaction  
✔️ Camera orbit controls (right-click drag)  
✔️ Color designed cloth (alternating pattern)  
✔️ Smooth animations using Verlet integration  
✔️ Interactive GUI using OpenGL + GLUT  
✔️ Handles tearing and spring constraint updates  

---

python3 mesh-analysis.py

---

##  Controls

| Key / Mouse     | Action |
|----------------|--------|
| **Left Mouse** | Draw and perform cutting |
| **Right Mouse** | Orbit/rotate camera |
| **SPACE**      | Perform horizontal cut |
| **C**          | Perform random cut |
| **R**          | Reset cloth |
| **Q / ESC**    | Quit simulation |

---

##  Installation & Setup

### 1️⃣ Install Python dependencies
```bash
pip install numpy PyOpenGL PyOpenGL_accelerate

If you are on Linux/macOS and see GLUT errors, install FreeGLUT

Ubuntu:
sudo apt-get install freeglut3-dev

Mac:
brew install freeglut

```
### 2️⃣ Run the simulation
```
python cloth_simulation.py

Project Structure
cloth-simulation/
│── cloth_simulation.py   # Main simulation code
│── README.md             # This file
