import matplotlib.pyplot as plt
import numpy as np

# Apply the gorgeous style
plt.style.use('/home/mila/d/dereck.piche/llm_negotiation/dedestyle.mplstyle')

# Create sample data
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x + np.pi/4)
y4 = np.cos(x + np.pi/4)



# Plot multiple lines to see the color cycle
plt.plot(x, y1, label='Orange (sin)')
plt.plot(x, y2, label='Green (cos)')
plt.plot(x, y3, label='Blue')
plt.plot(x, y4, label='Purple')



plt.title('Dede Style')
plt.xlabel('X Values', fontsize=14)
plt.ylabel('Y Values', fontsize=14)

plt.legend()
plt.savefig('/home/mila/d/dereck.piche/llm_negotiation/style_test.png')
plt.show()


