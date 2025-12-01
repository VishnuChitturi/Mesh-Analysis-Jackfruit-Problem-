import time
import numpy as np

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


# =========================
#   LOW-LEVEL PHYSICS
# =========================

class MassPoint:
    """
    Single mass node of the fabric.
    Stores position, previous position (for Verlet), velocity and forces.
    """

    def __init__(self, x, y, z, mass=1.0):
        self.pos = np.array([x, y, z], dtype=float)
        self.prev_pos = np.array([x, y, z], dtype=float)

        # These are mostly for debugging / future tweaks
        self.vel = np.zeros(3, dtype=float)
        self.force = np.zeros(3, dtype=float)

        self.mass = float(mass)
        self.locked = False      # pinned in space
        self.removed = False     # if we ever want to disable a point

    def add_force(self, f):
        """Accumulate an external force (gravity, wind, etc.)."""
        if not self.locked and not self.removed:
            self.force += f

    def integrate(self, dt, damping=0.98):
        """
        Verlet integration step.

        x_new = x + (x - x_prev) * damping + a * dt^2
        """
        if self.locked or self.removed:
            # Clear forces so they don't explode when re-enabled
            self.force[:] = 0.0
            return

        accel = self.force / self.mass
        next_pos = self.pos + (self.pos - self.prev_pos) * damping + accel * (dt * dt)

        self.prev_pos = self.pos.copy()
        self.pos = next_pos

        # Approximate velocity (not needed for sim, but useful to have)
        self.vel = (self.pos - self.prev_pos) / max(dt, 1e-6)

        # Clear forces for next frame
        self.force[:] = 0.0


class SpringLink:
    """
    Distance constraint between two MassPoint objects.
    Acts like a simple spring that tries to keep a rest length.
    """

    def __init__(self, a: MassPoint, b: MassPoint, stiffness=0.99):
        self.a = a
        self.b = b
        self.rest_len = np.linalg.norm(a.pos - b.pos)
        self.stiffness = float(stiffness)
        self.enabled = True

    def project(self):
        """
        Enforce the distance constraint between a and b.
        Uses a simple position correction step.
        """
        if not self.enabled:
            return

        if (self.a.removed or self.b.removed) or (self.a.locked and self.b.locked):
            return

        delta = self.b.pos - self.a.pos
        dist = np.linalg.norm(delta)

        if dist < 1e-5:
            return

        # How much it's stretched / compressed
        stretch = (dist - self.rest_len) / dist
        corr = delta * stretch * self.stiffness

        if not self.a.locked and not self.b.locked:
            self.a.pos += 0.5 * corr
            self.b.pos -= 0.5 * corr
        elif self.a.locked:
            self.b.pos -= corr
        elif self.b.locked:
            self.a.pos += corr


class ClothFace:
    """
    Triangle used for rendering the fabric surface
    and for hit-testing cuts.
    """

    def __init__(self, p0: MassPoint, p1: MassPoint, p2: MassPoint):
        self.nodes = [p0, p1, p2]
        self.visible = True

    def surface_normal(self):
        """Compute normal using cross product of triangle edges."""
        if not self.visible:
            return np.array([0.0, 0.0, 1.0], dtype=float)

        v0 = self.nodes[0].pos
        v1 = self.nodes[1].pos
        v2 = self.nodes[2].pos

        e1 = v1 - v0
        e2 = v2 - v0
        n = np.cross(e1, e2)
        ln = np.linalg.norm(n)
        if ln < 1e-6:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        return n / ln

    # ------- Cutting helpers -------

    @staticmethod
    def _seg_intersect_2d(a0, a1, b0, b1):
        """Return True if 2D segments (a0,a1) and (b0,b1) intersect."""
        def ccw(p, q, r):
            return (r[1] - p[1]) * (q[0] - p[0]) > (q[1] - p[1]) * (r[0] - p[0])

        return (ccw(a0, b0, b1) != ccw(a1, b0, b1)) and (ccw(a0, a1, b0) != ccw(a0, a1, b1))

    @staticmethod
    def _point_in_tri_2d(p, a, b, c):
        """Barycentric sign test in 2D."""
        def sign(p1, p2, p3):
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        d1 = sign(p, a, b)
        d2 = sign(p, b, c)
        d3 = sign(p, c, a)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
        return not (has_neg and has_pos)

    def intersects_segment(self, p_start, p_end):
        """
        Rough test: does the segment pass through or across this triangle?
        We:
          1. check Z-layer distance (to avoid cutting wrong depth)
          2. project everything to XY
          3. test against edges + interior
        """
        if not self.visible:
            return False, None

        v0, v1, v2 = [n.pos for n in self.nodes]

        # Layer check in Z so we don't cut far layers accidentally
        tri_z = (v0[2] + v1[2] + v2[2]) / 3.0
        seg_z = (p_start[2] + p_end[2]) / 2.0
        if abs(tri_z - seg_z) > 0.1:
            return False, None

        # Project to XY
        s0 = p_start[:2]
        s1 = p_end[:2]
        t0 = v0[:2]
        t1 = v1[:2]
        t2 = v2[:2]

        # Check intersection with each triangle edge
        edges = [(t0, t1), (t1, t2), (t2, t0)]
        for e0, e1 in edges:
            if self._seg_intersect_2d(s0, s1, e0, e1):
                hit = 0.5 * (p_start + p_end)  # approximate hit point
                return True, hit

        # Or segment endpoints inside triangle
        if self._point_in_tri_2d(s0, t0, t1, t2) or self._point_in_tri_2d(s1, t0, t1, t2):
            hit = 0.5 * (p_start + p_end)
            return True, hit

        return False, None


# =========================
#   FABRIC SYSTEM
# =========================

class FabricSystem:
    """
    Manages the entire cloth: points, springs, faces and cutting.
    """

    def __init__(self,
                 cols=20,
                 rows=20,
                 spacing=0.1,
                 layer_count=2,
                 layer_gap=0.05):

        self.cols = int(cols)
        self.rows = int(rows)
        self.spacing = float(spacing)
        self.layer_count = int(layer_count)
        self.layer_gap = float(layer_gap)

        self.points = []
        self.springs = []
        self.faces = []

        self.gravity = np.array([0.0, -9.81, 0.0], dtype=float)
        self.damping = 0.98
        self.relax_loops = 5

        self._build_mesh()

    # ------- Construction -------

    def _build_mesh(self):
        """Create grid of points, springs, and faces for all layers."""
        print(f"Creating fabric grid: {self.cols}x{self.rows}, layers={self.layer_count}")

        # Reference grid for each layer: [layer][row][col]
        grid = []

        for layer_idx in range(self.layer_count):
            z_offset = layer_idx * self.layer_gap
            layer_grid = []

            for r in range(self.rows + 1):
                row_nodes = []
                for c in range(self.cols + 1):
                    # Center the cloth around origin
                    x = (c - self.cols / 2.0) * self.spacing
                    y = (self.rows / 2.0 - r) * self.spacing
                    z = z_offset

                    node = MassPoint(x, y, z)

                    # Only pin top corners of the first (front) layer
                    if layer_idx == 0 and r == 0 and (c == 0 or c == self.cols):
                        node.locked = True

                    self.points.append(node)
                    row_nodes.append(node)
                layer_grid.append(row_nodes)
            grid.append(layer_grid)

        # Create structural & diagonal springs in each layer
        for layer_idx in range(self.layer_count):
            layer_grid = grid[layer_idx]
            for r in range(self.rows + 1):
                for c in range(self.cols + 1):
                    p = layer_grid[r][c]

                    # horizontal
                    if c < self.cols:
                        self.springs.append(SpringLink(p, layer_grid[r][c + 1]))

                    # vertical
                    if r < self.rows:
                        self.springs.append(SpringLink(p, layer_grid[r + 1][c]))

                    # diagonals
                    if r < self.rows and c < self.cols:
                        self.springs.append(SpringLink(p, layer_grid[r + 1][c + 1]))
                        self.springs.append(SpringLink(layer_grid[r][c + 1],
                                                       layer_grid[r + 1][c]))

        # Connect layers vertically (same (r,c) across layers)
        if self.layer_count > 1:
            for layer_idx in range(self.layer_count - 1):
                upper = grid[layer_idx]
                lower = grid[layer_idx + 1]
                for r in range(self.rows + 1):
                    for c in range(self.cols + 1):
                        self.springs.append(SpringLink(upper[r][c], lower[r][c], stiffness=0.99))

        # Build triangle faces for each cell in each layer
        for layer_idx in range(self.layer_count):
            layer_grid = grid[layer_idx]
            for r in range(self.rows):
                for c in range(self.cols):
                    p00 = layer_grid[r][c]
                    p10 = layer_grid[r][c + 1]
                    p01 = layer_grid[r + 1][c]
                    p11 = layer_grid[r + 1][c + 1]

                    # Two triangles per quad
                    self.faces.append(ClothFace(p00, p10, p01))
                    self.faces.append(ClothFace(p10, p11, p01))

        print(f"Fabric created: {len(self.points)} points, {len(self.springs)} springs, {len(self.faces)} faces")

    # ------- Simulation -------

    def step(self, dt):
        """Advance physics by dt seconds."""
        # 1) Add gravity
        for p in self.points:
            if not p.locked:
                p.add_force(self.gravity * p.mass)

        # 2) Integrate all points
        for p in self.points:
            p.integrate(dt, self.damping)

        # 3) Relax springs (project constraints)
        for _ in range(self.relax_loops):
            for s in self.springs:
                s.project()

        # 4) Extra stretch clamp (prevent crazy explosions)
        for s in self.springs:
            if not s.enabled:
                continue

            diff = s.b.pos - s.a.pos
            dist = np.linalg.norm(diff)
            if dist < 1e-6:
                continue

            max_len = 1.5 * s.rest_len
            if dist > max_len:
                # Pull them closer together
                excess = dist - max_len
                corr = diff * (excess / dist)
                if not s.a.locked and not s.b.locked:
                    s.a.pos += 0.5 * corr
                    s.b.pos -= 0.5 * corr
                elif s.a.locked:
                    s.b.pos -= corr
                elif s.b.locked:
                    s.a.pos += corr

    # ------- Cutting -------

    def _segments_intersect_2d(self, a0, a1, b0, b1):
        """2D segment intersection helper."""
        def ccw(p, q, r):
            return (r[1] - p[1]) * (q[0] - p[0]) > (q[1] - p[1]) * (r[0] - p[0])
        return (ccw(a0, b0, b1) != ccw(a1, b0, b1)) and (ccw(a0, a1, b0) != ccw(a0, a1, b1))

    def cut_with_segment(self, start_world, end_world):
        """
        Cut the cloth along a 3D segment.
        Deactivates intersected faces and disables springs that cross the segment.
        """
        print(f"Cut request: from {start_world} to {end_world}")

        cut_faces = 0
        touched_points = set()

        # 1) Disable faces intersected by the cut
        for f in self.faces:
            hit, _ = f.intersects_segment(start_world, end_world)
            if hit:
                if f.visible:
                    cut_faces += 1
                f.visible = False
                for p in f.nodes:
                    touched_points.add(p)

        # 2) Disable springs that cross the same 2D segment
        s0 = start_world[:2]
        s1 = end_world[:2]

        for s in self.springs:
            if not s.enabled:
                continue

            a2 = s.a.pos[:2]
            b2 = s.b.pos[:2]

            if self._segments_intersect_2d(a2, b2, s0, s1):
                s.enabled = False

        print(f"Cut completed: {cut_faces} faces hidden, {len(touched_points)} points affected.")
        return cut_faces > 0


# =========================
#   OPENGL VIEWER
# =========================

class FabricViewer:
    """
    GLUT-based viewer for interacting with the FabricSystem.
    Left drag = draw cut line
    Right drag = orbit camera
    Keys: SPACE = horizontal cut, 'c' = random cut, 'r' = reset.
    """

    def __init__(self):
        # Simulation
        self.fabric = None
        self.last_time = time.time()

        # Camera
        self.cam_pitch = 25.0
        self.cam_yaw = -20.0
        self.cam_dist = 12.0

        # Mouse
        self._last_mouse_x = 0
        self._last_mouse_y = 0
        self._active_button = None

        # Cutting line
        self._cut_start = None
        self._cut_end = None
        self._drawing_cut = False

    # ------- GL Setup & Callbacks -------

    def _init_gl(self):
        """Initial GL state + create fabric."""
        glClearColor(0.1, 0.4, 0.65, 1.0)
        glEnable(GL_DEPTH_TEST)

        # Lighting setup
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_COLOR_MATERIAL)

        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

        glLightfv(GL_LIGHT0, GL_POSITION, [3.0, 5.0, 2.0, 1.0])
        glLightfv(GL_LIGHT0, GL_AMBIENT, [0.15, 0.15, 0.2, 1.0])
        glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.9, 0.9, 0.9, 1.0])

        # Slightly more "silky" color
        self.fabric = FabricSystem(cols=15, rows=15, spacing=0.5, layer_count=2, layer_gap=0.05)

    def _display(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Camera transform
        gluLookAt(0.0, 0.0, self.cam_dist,
                  0.0, 0.0, 0.0,
                  0.0, 1.0, 0.0)
        glRotatef(self.cam_pitch, 1, 0, 0)
        glRotatef(self.cam_yaw, 0, 1, 0)

        # Draw fabric
        self._draw_fabric()

        # Draw cut preview
        if self._cut_start is not None and self._cut_end is not None:
            self._draw_cut_preview()

        glutSwapBuffers()

    def _draw_fabric(self):
        """Draw faces + edge springs."""
        # Filled surface
        glEnable(GL_LIGHTING)
        glColor3f(0.0, 1.0, 0.0)

        glBegin(GL_TRIANGLES)
        for face in self.fabric.faces:
            if not face.visible:
                continue
            n = face.surface_normal()
            glNormal3fv(n)
            for p in face.nodes:
                glVertex3fv(p.pos)
        glEnd()

        # Wireframe overlay
        glDisable(GL_LIGHTING)
        glLineWidth(1.0)
        glColor3f(0.15, 0.2, 0.35)

        glBegin(GL_LINES)
        for s in self.fabric.springs:
            if not s.enabled:
                continue
            glVertex3fv(s.a.pos)
            glVertex3fv(s.b.pos)
        glEnd()

        glEnable(GL_LIGHTING)

    def _draw_cut_preview(self):
        glDisable(GL_LIGHTING)
        glLineWidth(3.0)
        glColor3f(1.0, 0.3, 0.87)

        glBegin(GL_LINES)
        glVertex3fv(self._cut_start)
        glVertex3fv(self._cut_end)
        glEnd()

        glEnable(GL_LIGHTING)

    def _idle(self):
        """Physics tick + redraw."""
        now = time.time()
        dt = now - self.last_time
        # Limit dt so simulation stays stable if window stalls
        dt = min(dt, 0.016)
        self.last_time = now

        if self.fabric is not None:
            self.fabric.step(dt)

        glutPostRedisplay()

    def _reshape(self, w, h):
        if h == 0:
            h = 1
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, float(w) / float(h), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)

    # ------- Input -------

    def _keyboard(self, key, x, y):
        if key in (b'q', b'\x1b'):
            glutLeaveMainLoop()
        elif key == b'c':
            self._random_cut()
        elif key == b'r':
            self.fabric = FabricSystem(cols=15, rows=15, spacing=0.15, layer_count=2)
            print("Fabric reset.")
        elif key == b' ':
            self._horizontal_cut()

    def _mouse(self, button, state, x, y):
        self._last_mouse_x = x
        self._last_mouse_y = y

        if button == GLUT_LEFT_BUTTON:
            if state == GLUT_DOWN:
                self._active_button = GLUT_LEFT_BUTTON
                self._drawing_cut = True
                self._cut_start = self._screen_to_world(x, y)
                self._cut_end = self._cut_start.copy()
            else:
                # Finish cut
                if self._drawing_cut and self._cut_start is not None and self._cut_end is not None:
                    self.fabric.cut_with_segment(self._cut_start, self._cut_end)
                self._drawing_cut = False
                self._active_button = None

        elif button == GLUT_RIGHT_BUTTON:
            if state == GLUT_DOWN:
                self._active_button = GLUT_RIGHT_BUTTON
            else:
                self._active_button = None

    def _motion(self, x, y):
        dx = x - self._last_mouse_x
        dy = y - self._last_mouse_y

        if self._active_button == GLUT_RIGHT_BUTTON:
            # Orbit camera
            self.cam_yaw += dx * 0.4
            self.cam_pitch += dy * 0.4
        elif self._drawing_cut:
            # Update cut line
            self._cut_end = self._screen_to_world(x, y)

        self._last_mouse_x = x
        self._last_mouse_y = y

    # ------- Helpers -------

    def _screen_to_world(self, sx, sy):
        """
        Very rough mapping of screen coords to a plane
        in front of the camera (around the cloth).
        Good enough for drawing a cutting line.
        """
        viewport = glGetIntegerv(GL_VIEWPORT)
        w = viewport[2]
        h = viewport[3]

        # Normalized device coords in [-1, 1]
        nx = (sx / float(w)) * 2.0 - 1.0
        ny = 1.0 - (sy / float(h)) * 2.0

        # Map to some fixed plane around origin
        world_x = nx * 2.0
        world_y = ny * 2.0
        world_z = 0.025   # roughly between the two fabric layers

        return np.array([world_x, world_y, world_z], dtype=float)

    def _random_cut(self):
        """Cut with a random short segment through the cloth."""
        s = np.array([
            np.random.uniform(-1.5, 1.5),
            np.random.uniform(-1.5, 1.5),
            0.025
        ], dtype=float)
        e = np.array([
            np.random.uniform(-1.5, 1.5),
            np.random.uniform(-1.5, 1.5),
            0.025
        ], dtype=float)

        self.fabric.cut_with_segment(s, e)
        print("Random cut executed.")

    def _horizontal_cut(self):
        """Straight horizontal cut across center."""
        s = np.array([-2.0, 0.0, 0.025], dtype=float)
        e = np.array([2.0, 0.0, 0.025], dtype=float)
        self.fabric.cut_with_segment(s, e)
        print("Middle horizontal cut executed.")

    # ------- Entry point -------

    def run(self):
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(800, 600)
        glutCreateWindow(b"Fabric Cutter - Cloth Simulation")

        self._init_gl()

        glutDisplayFunc(self._display)
        glutReshapeFunc(self._reshape)
        glutKeyboardFunc(self._keyboard)
        glutMouseFunc(self._mouse)
        glutMotionFunc(self._motion)
        glutIdleFunc(self._idle)

        print("\n=== CONTROLS ===")
        print("Left Mouse  : drag to draw a cut line")
        print("Right Mouse : drag to orbit camera")
        print("SPACE       : horizontal cut across the cloth")
        print("C           : random cut")
        print("R           : reset fabric")
        print("Q / ESC     : quit")
        print("================\n")

        glutMainLoop()


if __name__ == "__main__":
    viewer = FabricViewer()
    viewer.run()
