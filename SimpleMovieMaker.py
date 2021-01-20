import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.animation as animation


PATH = "Chronic/scratch/PML1/Movie/Ex%d.png"

def update(i):
	p = PATH.format(i)
	image = mpimg.imread(p)
	plt.gca().clear()
	plt.imshow(image)

# Set up formatting for the movie files
Writer = animation.writers['ffmpeg']
writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)

ani = animation.FuncAnimation(plt.gcf(), update, range(1,4), interval = 1000, repeat=False)
ani.save("diapole.mp4", writer = writer )