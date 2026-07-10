import numpy as np
import matplotlib.pyplot as plt

# Create figure
fig = plt.figure(figsize=(11, 6))
ax = fig.add_subplot(111, projection='3d')

# Ambient curved surface (wireframe)
x = np.linspace(-1.4, 1.4, 40)
y = np.linspace(-1.2, 1.2, 40)
X, Y = np.meshgrid(x, y)
Z = 0.25*(X**2 - 0.7*Y**2)  # gentle saddle-like surface
ax.plot_wireframe(X, Y, Z, rstride=3, cstride=3, linewidth=0.5)

# Helper: parallelogram corners
def parallelogram(center, u, v):
    c = np.array(center)
    u = np.array(u)
    v = np.array(v)
    corners = [c - u - v, c + u - v, c + u + v, c - u + v]
    return np.array(corners)

left_plane = parallelogram(center=(-2.2, 0.0, -0.2), u=(0.0, 1.2, 0.2), v=(0.0, 0.0, 0.9))
right_plane = parallelogram(center=( 2.2, 0.0, -0.2), u=(0.0, 1.2, 0.2), v=(0.0, 0.0, 0.9))

# Draw plane outlines + internal dashed lines
def draw_plane_grid(plane, lw=2):
    loop = np.vstack([plane, plane[0]])
    ax.plot(loop[:,0], loop[:,1], loop[:,2], linewidth=lw)
    for t in np.linspace(-0.8, 0.8, 5):
        p1 = plane[0]*(0.5 - t/2) + plane[1]*(0.5 + t/2)
        p2 = plane[3]*(0.5 - t/2) + plane[2]*(0.5 + t/2)
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], [p1[2], p2[2]], linewidth=0.8, linestyle='--', alpha=0.7)

draw_plane_grid(left_plane)
draw_plane_grid(right_plane)

# Points: joints, projections (KL projections), and generic product candidates
p_xz = np.array([-1.0, 0.6, 0.55])      # p_phi(x,z)
proj_xz = np.array([-2.2, 0.6, 0.05])   # Pi_{I_XZ}(p_phi) = p(x)p_phi(z)
qxr = np.array([-2.2, -0.2, -0.15])     # generic q(x)r(z) on I_XZ

p_yz = np.array([ 1.0, -0.5, 0.55])     # p_phi(y,z)
proj_yz = np.array([ 2.2, -0.5, 0.05])  # Pi_{I_YZ}(p_phi) = p(y)p_phi(z)
qyr = np.array([ 2.2, 0.25, -0.15])     # generic q'(y)r'(z) on I_YZ

# Plot points
ax.scatter(*p_xz, s=70, marker='o')
ax.scatter(*p_yz, s=80, marker='^')

# Projections highlighted with star markers
ax.scatter(*proj_xz, s=140, marker='*')
ax.scatter(*proj_yz, s=140, marker='*')

# Generic candidate product points
ax.scatter(*qxr, s=55, marker='o')
ax.scatter(*qyr, s=55, marker='^')

# --- Left triangle: KL(p||qr) decomposes into I(X;Z) + KL(proj||qr) ---
# I(X;Z) (distance to independence, minimized by projection)
ax.plot([p_xz[0], proj_xz[0]], [p_xz[1], proj_xz[1]], [p_xz[2], proj_xz[2]], linestyle='--', linewidth=1.8)
# KL(proj || q r)
ax.plot([proj_xz[0], qxr[0]], [proj_xz[1], qxr[1]], [proj_xz[2], qxr[2]], linestyle='--', linewidth=1.2, alpha=0.9)
# KL(p || q r) (shown schematically as a dotted connector)
ax.plot([p_xz[0], qxr[0]], [p_xz[1], qxr[1]], [p_xz[2], qxr[2]], linestyle=':', linewidth=1.6)

# Push arrow (compression): toward projection
vec_push = proj_xz - p_xz
ax.quiver(p_xz[0], p_xz[1], p_xz[2], vec_push[0], vec_push[1], vec_push[2],
          arrow_length_ratio=0.15, linewidth=2)

# --- Right triangle: KL(p||q'r') decomposes into I(Z;Y) + KL(proj||q'r') ---
ax.plot([p_yz[0], proj_yz[0]], [p_yz[1], proj_yz[1]], [p_yz[2], proj_yz[2]], linestyle='--', linewidth=1.8)
ax.plot([proj_yz[0], qyr[0]], [proj_yz[1], qyr[1]], [proj_yz[2], qyr[2]], linestyle='--', linewidth=1.2, alpha=0.9)
ax.plot([p_yz[0], qyr[0]], [p_yz[1], qyr[1]], [p_yz[2], qyr[2]], linestyle=':', linewidth=1.6)

# Pull arrow (prediction): away from independence manifold (lengthen I(Z;Y))
vec_pull = p_yz - proj_yz
ax.quiver(proj_yz[0], proj_yz[1], proj_yz[2], vec_pull[0], vec_pull[1], vec_pull[2],
          arrow_length_ratio=0.15, linewidth=2)

# Labels near points
ax.text(p_xz[0]-0.20, p_xz[1]+0.06, p_xz[2]+0.06, r'$p_\phi(x,z)$', fontsize=11)
#ax.text(proj_xz[0]-0.52, proj_xz[1]+0.06, proj_xz[2]-0.07, r'$\Pi_{\mathcal{I}_{XZ}}(p_\phi)=p(x)p_\phi(z)$', fontsize=10)
ax.text(qxr[0]-0.40, qxr[1]-0.05, qxr[2]-0.06, r'$q(x)r(z)$', fontsize=11)

ax.text(p_yz[0]-0.20, p_yz[1]-0.10, p_yz[2]+0.06, r'$p_\phi(y,z)$', fontsize=11)
#ax.text(proj_yz[0]-0.52, proj_yz[1]-0.10, proj_yz[2]-0.07, r'$\Pi_{\mathcal{I}_{YZ}}(p_\phi)=p(y)p_\phi(z)$', fontsize=10)
ax.text(qyr[0]-0.45, qyr[1]-0.03, qyr[2]-0.06, r"$q'(y)r'(z)$", fontsize=11)

# Side labels (placed roughly at midpoints)
def midpoint(a, b):
    return (a + b) / 2

m1 = midpoint(p_xz, proj_xz)
ax.text(m1[0]-0.15, m1[1]+0.05, m1[2]+0.02, r'$I_\phi(X;Z)$', fontsize=12)

m2 = midpoint(proj_xz, qxr)
#ax.text(m2[0]-0.10, m2[1]-0.03, m2[2]-0.03, r'$\mathrm{KL}\!\left(p(x)p_\phi(z)\,\|\,q(x)r(z)\right)$', fontsize=9)

m3 = midpoint(p_xz, qxr)
#ax.text(m3[0]-0.05, m3[1]+0.02, m3[2]+0.02, r'$\mathrm{KL}\!\left(p_\phi(x,z)\,\|\,q(x)r(z)\right)$', fontsize=9)

m4 = midpoint(p_yz, proj_yz)
ax.text(m4[0]-0.10, m4[1]-0.02, m4[2]+0.02, r'$I_\phi(Z;Y)$', fontsize=12)

m5 = midpoint(proj_yz, qyr)
#ax.text(m5[0]-0.10, m5[1]-0.02, m5[2]-0.03, r'$\mathrm{KL}\!\left(p(y)p_\phi(z)\,\|\,q^\prime(y)r^\prime(z)\right)$', fontsize=9)

m6 = midpoint(p_yz, qyr)
#ax.text(m6[0]-0.05, m6[1]-0.02, m6[2]+0.02, r'$\mathrm{KL}\!\left(p_\phi(y,z)\,\|\,q^\prime(y)r^\prime(z)\right)$', fontsize=9)

# Manifold labels
ax.text(-2.55, -1.15, -0.48, r'$\mathcal{I}_{XZ}=\{q(x)r(z)\}$', fontsize=12)
ax.text( 1.90, -1.15, -0.48, r'$\mathcal{I}_{YZ}=\{q^\prime(y)r^\prime(z)\}$', fontsize=12)
ax.text(-0.45,  1.05,  0.80, r'ambient space $\mathcal{P}$', fontsize=12)

# Titles and loss
#fig.text(0.12, 0.93, 'Compression term', fontsize=14, weight='bold')
#fig.text(0.12, 0.89, r'Minimize $\beta\,I_\phi(X;Z)$ (shorten dashed segment)', fontsize=11)

#fig.text(0.70, 0.93, 'Prediction term', fontsize=14, weight='bold')
#fig.text(0.70, 0.89, r'Maximize $I_\phi(Z;Y)$ (lengthen dashed segment)', fontsize=11)

#fig.text(0.35, 0.05, r'$\mathcal{L}_{\mathrm{IB}}(\phi)=\beta\,I_\phi(X;Z)-I_\phi(Z;Y)$', fontsize=14)

# Clean style
ax.set_axis_off()
ax.view_init(elev=22, azim=-60)

# Save

pdf_path = "geometric_ib_schematic_with_KL_projections.pdf"
plt.tight_layout()
#plt.savefig(png_path, dpi=300, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.show()

pdf_path