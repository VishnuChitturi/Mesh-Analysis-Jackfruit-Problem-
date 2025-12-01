
import time
import numpy as np

from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *


# =========================
#   LOW-LEVEL PHYSICS
# =========================

class MP:
    """
    Single mass node of the fabric.
    Stores position, previous position (for Verlet), velocity and forces.
    """

    def __init__(self, x, y, z, mass=1.0):
        self.p = np.array([x, y, z], dtype=float)
        self.pp = np.array([x, y, z], dtype=float)

        # These are mostly for debugging / future tweaks
        self.v = np.zeros(3, dtype=float)
        self.f = np.zeros(3, dtype=float)

        self.mass = float(mass)
        self.lk = False      # pinned in space
        self.rm = False     # if we ever want to disable a point

    def af(self, f):
        """Accumulate an external f (g, wind, etc.)."""
        if not self.lk and not self.rm:
            self.f += f

    def upd(self, dt, dp=0.98):
        """
        Verlet integration st.

        x_new = x + (x - x_prev) * dp + a * dt^2
        """
        if self.lk or self.rm:
            # Clear forces so they don't explode when re-en
            self.f[:] = 0.0
            return

        accel = self.f / self.mass
        next_pos = self.p + (self.p - self.pp) * dp + accel * (dt * dt)

        self.pp = self.p.copy()
        self.p = next_pos

        # Approximate velocity (not needed for sim, but useful to have)
        self.v = (self.p - self.pp) / max(dt, 1e-6)

        # Clear forces for next frame
        self.f[:] = 0.0


class SL:
    """
    Distance constraint between two MP objects.
    Acts like a simple spring that tries to keep a rest length.
    """

    def __init__(self, a: MP, b: MP, stiffness=0.99):
        self.a = a
        self.b = b
        self.rl = np.linalg.norm(a.p - b.p)
        self.stiffness = float(stiffness)
        self.en = True

    def project(self):
        """
        Enforce the distance constraint between a and b.
        Uses a simple position correction st.
        """
        if not self.en:
            return

        if (self.a.rm or self.b.rm) or (self.a.lk and self.b.lk):
            return

        delta = self.b.p - self.a.p
        dist = np.linalg.norm(delta)

        if dist < 1e-5:
            return

        # How much it's stretched / compressed
        stretch = (dist - self.rl) / dist
        corr = delta * stretch * self.stiffness

        if not self.a.lk and not self.b.lk:
            self.a.p += 0.5 * corr
            self.b.p -= 0.5 * corr
        elif self.a.lk:
            self.b.p -= corr
        elif self.b.lk:
            self.a.p += corr


class CF:
    """
    Triangle used for rendering the fabric surface
    and for hit-testing cuts.
    """

    def __init__(self, p0: MP, p1: MP, p2: MP):
        self.nodes = [p0, p1, p2]
        self.vs = True

    def sn(self):
        """Compute normal using cross product of triangle edges."""
        if not self.vs:
            return np.array([0.0, 0.0, 1.0], dtype=float)

        v0 = self.nodes[0].p
        v1 = self.nodes[1].p
        v2 = self.nodes[2].p

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
        if not self.vs:
            return False, None

        v0, v1, v2 = [n.p for n in self.nodes]

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

class FS:
    """
    Manages the entire cloth: points, springs, faces and cutting.
    """

    def __init__(self,
                 c=20,
                 r=20,
                 sp=0.1,
                 lc=2,
                 lg=0.05):

        self.c = int(c)
        self.r = int(r)
        self.sp = float(sp)
        self.lc = int(lc)
        self.lg = float(lg)

        self.points = []
        self.springs = []
        self.faces = []

        self.g = np.array([0.0, -9.81, 0.0], dtype=float)
        self.dp = 0.98
        self.rl = 5

        self._build_mesh()

    # ------- Construction -------

    def _build_mesh(self):
        """Create grid of points, springs, and faces for all layers."""
        print(f"Creating fabric grid: {self.c}x{self.r}, layers={self.lc}")

        # Reference grid for each layer: [layer][row][col]
        grid = []

        for layer_idx in range(self.lc):
            z_offset = layer_idx * self.lg
            layer_grid = []

            for r in range(self.r + 1):
                row_nodes = []
                for c in range(self.c + 1):
                    # Center the cloth around origin
                    x = (c - self.c / 2.0) * self.sp
                    y = (self.r / 2.0 - r) * self.sp
                    z = z_offset

                    node = MP(x, y, z)

                    # Only pin top corners of the first (front) layer
                    if layer_idx == 0 and r == 0 and (c == 0 or c == self.c):
                        node.lk = True

                    self.points.append(node)
                    row_nodes.append(node)
                layer_grid.append(row_nodes)
            grid.append(layer_grid)

        # Create structural & diagonal springs in each layer
        for layer_idx in range(self.lc):
            layer_grid = grid[layer_idx]
            for r in range(self.r + 1):
                for c in range(self.c + 1):
                    p = layer_grid[r][c]

                    # horizontal
                    if c < self.c:
                        self.springs.append(SL(p, layer_grid[r][c + 1]))

                    # vertical
                    if r < self.r:
                        self.springs.append(SL(p, layer_grid[r + 1][c]))

                    # diagonals
                    if r < self.r and c < self.c:
                        self.springs.append(SL(p, layer_grid[r + 1][c + 1]))
                        self.springs.append(SL(layer_grid[r][c + 1],
                                                       layer_grid[r + 1][c]))

        # Connect layers vertically (same (r,c) across layers)
        if self.lc > 1:
            for layer_idx in range(self.lc - 1):
                upper = grid[layer_idx]
                lower = grid[layer_idx + 1]
                for r in range(self.r + 1):
                    for c in range(self.c + 1):
                        self.springs.append(SL(upper[r][c], lower[r][c], stiffness=0.99))

        # Build triangle faces for each cell in each layer
        for layer_idx in range(self.lc):
            layer_grid = grid[layer_idx]
            for r in range(self.r):
                for c in range(self.c):
                    p00 = layer_grid[r][c]
                    p10 = layer_grid[r][c + 1]
                    p01 = layer_grid[r + 1][c]
                    p11 = layer_grid[r + 1][c + 1]

                    # Two triangles per quad
                    self.faces.append(CF(p00, p10, p01))
                    self.faces.append(CF(p10, p11, p01))

        print(f"Fabric created: {len(self.points)} points, {len(self.springs)} springs, {len(self.faces)} faces")

    # ------- Simulation -------

    def st(self, dt):
        """Advance physics by dt seconds."""
        # 1) Add g
        for p in self.points:
            if not p.lk:
                p.af(self.g * p.mass)

        # 2) Integrate all points
        for p in self.points:
            p.upd(dt, self.dp)

        # 3) Relax springs (project constraints)
        for _ in range(self.rl):
            for s in self.springs:
                s.project()

        # 4) Extra stretch clamp (prevent crazy explosions)
        for s in self.springs:
            if not s.en:
                continue

            diff = s.b.p - s.a.p
            dist = np.linalg.norm(diff)
            if dist < 1e-6:
                continue

            max_len = 1.5 * s.rl
            if dist > max_len:
                # Pull them closer together
                excess = dist - max_len
                corr = diff * (excess / dist)
                if not s.a.lk and not s.b.lk:
                    s.a.p += 0.5 * corr
                    s.b.p -= 0.5 * corr
                elif s.a.lk:
                    s.b.p -= corr
                elif s.b.lk:
                    s.a.p += corr

    # ------- Cutting -------

    def _segments_intersect_2d(self, a0, a1, b0, b1):
        """2D segment intersection helper."""
        def ccw(p, q, r):
            return (r[1] - p[1]) * (q[0] - p[0]) > (q[1] - p[1]) * (r[0] - p[0])
        return (ccw(a0, b0, b1) != ccw(a1, b0, b1)) and (ccw(a0, a1, b0) != ccw(a0, a1, b1))

    def cut(self, start_world, end_world):
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
                if f.vs:
                    cut_faces += 1
                f.vs = False
                for p in f.nodes:
                    touched_points.add(p)

        # 2) Disable springs that cross the same 2D segment
        s0 = start_world[:2]
        s1 = end_world[:2]

        for s in self.springs:
            if not s.en:
                continue

            a2 = s.a.p[:2]
            b2 = s.b.p[:2]

            if self._segments_intersect_2d(a2, b2, s0, s1):
                s.en = False

        print(f"Cut completed: {cut_faces} faces hidden, {len(touched_points)} points affected.")
        return cut_faces > 0


# =========================
#   OPENGL VIEWER
# =========================

class FV:
    """
    GLUT-based viewer for interacting with the FS.
    Left drag = draw cut line
    Right drag = orbit camera
    Keys: SPACE = horizontal cut, 'c' = random cut, 'r' = reset.
    """

    def __init__(self):
        # Simulation
        self.fabric = None
        self.lt = time.time()

        # Camera
        self.cp = 25.0
        self.cy = -20.0
        self.cd = 12.0

        # Mouse
        self._lmx = 0
        self._lmy = 0
        self._ab = None

        # Cutting line
        self._cs = None
        self._ce = None
        self._dcf = False

    # ------- GL Setup & Callbacks -------

    def _ig(self):
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
        self.fabric = FS(c=15, r=15, sp=0.5, lc=2, lg=0.05)

    def _ds(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        # Camera transform
        gluLookAt(0.0, 0.0, self.cd,
                  0.0, 0.0, 0.0,
                  0.0, 1.0, 0.0)
        glRotatef(self.cp, 1, 0, 0)
        glRotatef(self.cy, 0, 1, 0)

        # Draw fabric
        self._df()

        # Draw cut preview
        if self._cs is not None and self._ce is not None:
            self._dc()

        glutSwapBuffers()

    def _df(self):
        """Draw faces + edge springs."""
        # Filled surface
        glEnable(GL_LIGHTING)
        glColor3f(0.0, 1.0, 0.0)

        glBegin(GL_TRIANGLES)
        for face in self.fabric.faces:
            if not face.vs:
                continue
            n = face.sn()
            glNormal3fv(n)
            for p in face.nodes:
                glVertex3fv(p.p)
        glEnd()

        # Wireframe overlay
        glDisable(GL_LIGHTING)
        glLineWidth(1.0)
        glColor3f(0.15, 0.2, 0.35)

        glBegin(GL_LINES)
        for s in self.fabric.springs:
            if not s.en:
                continue
            glVertex3fv(s.a.p)
            glVertex3fv(s.b.p)
        glEnd()

        glEnable(GL_LIGHTING)

    def _dc(self):
        glDisable(GL_LIGHTING)
        glLineWidth(3.0)
        glColor3f(1.0, 0.3, 0.87)

        glBegin(GL_LINES)
        glVertex3fv(self._cs)
        glVertex3fv(self._ce)
        glEnd()

        glEnable(GL_LIGHTING)

    def _id(self):
        """Physics tick + redraw."""
        now = time.time()
        dt = now - self.lt
        # Limit dt so simulation stays stable if window stalls
        dt = min(dt, 0.016)
        self.lt = now

        if self.fabric is not None:
            self.fabric.st(dt)

        glutPostRedisplay()

    def _rp(self, w, h):
        if h == 0:
            h = 1
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, float(w) / float(h), 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)

    # ------- Input -------

    def _kb(self, key, x, y):
        if key in (b'q', b'\x1b'):
            glutLeaveMainLoop()
        elif key == b'c':
            self._rc()
        elif key == b'r':
            self.fabric = FS(c=15, r=15, sp=0.15, lc=2)
            print("Fabric reset.")
        elif key == b' ':
            self._hc()

    def _ms(self, button, state, x, y):
        self._lmx = x
        self._lmy = y

        if button == GLUT_LEFT_BUTTON:
            if state == GLUT_DOWN:
                self._ab = GLUT_LEFT_BUTTON
                self._dcf = True
                self._cs = self._screen_to_world(x, y)
                self._ce = self._cs.copy()
            else:
                # Finish cut
                if self._dcf and self._cs is not None and self._ce is not None:
                    self.fabric.cut(self._cs, self._ce)
                self._dcf = False
                self._ab = None

        elif button == GLUT_RIGHT_BUTTON:
            if state == GLUT_DOWN:
                self._ab = GLUT_RIGHT_BUTTON
            else:
                self._ab = None

    def _mt(self, x, y):
        dx = x - self._lmx
        dy = y - self._lmy

        if self._ab == GLUT_RIGHT_BUTTON:
            # Orbit camera
            self.cy += dx * 0.4
            self.cp += dy * 0.4
        elif self._dcf:
            # Update cut line
            self._ce = self._screen_to_world(x, y)

        self._lmx = x
        self._lmy = y

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

    def _rc(self):
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

        self.fabric.cut(s, e)
        print("Random cut executed.")

    def _hc(self):
        """Straight horizontal cut across center."""
        s = np.array([-2.0, 0.0, 0.025], dtype=float)
        e = np.array([2.0, 0.0, 0.025], dtype=float)
        self.fabric.cut(s, e)
        print("Middle horizontal cut executed.")

    # ------- Entry point -------

    def start(self):
        glutInit()
        glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH)
        glutInitWindowSize(800, 600)
        glutCreateWindow(b"3D Mesh Simulation - Final Version")

        self._ig()

        glutDisplayFunc(self._ds)
        glutReshapeFunc(self._rp)
        glutKeyboardFunc(self._kb)
        glutMouseFunc(self._ms)
        glutMotionFunc(self._mt)
        glutIdleFunc(self._id)

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
    viewer = FV()
    viewer.start()
