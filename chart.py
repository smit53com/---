import io
import matplotlib.pyplot as plt

def render_wheel_png(longs):
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': 'polar'})
    ax.set_theta_direction(-1)
    ax.set_theta_offset(3.14159/2.0)
    for planet, lon in longs.items():
        theta = (lon / 180.0) * 3.14159
        ax.plot(theta, 1, 'o', label=planet)
    ax.legend(bbox_to_anchor=(1.1, 1.05))
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    plt.close(fig)
    return buf
